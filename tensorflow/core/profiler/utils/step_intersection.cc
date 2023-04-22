/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/profiler/utils/step_intersection.h"

#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

namespace {

// Returns the timespan in this step (across all cores).
Timespan StepTimespan(const PerCoreStepInfo& percore_stepinfo) {
  uint64 min_ps = kuint64max;
  uint64 max_ps = 0;
  for (const auto& core_stepinfo : percore_stepinfo.step_info_per_core()) {
    const auto& stepinfo = core_stepinfo.second;
    uint64 begin_ps = stepinfo.begin_ps();
    uint64 end_ps = begin_ps + stepinfo.duration_ps();
    min_ps = std::min(min_ps, begin_ps);
    max_ps = std::max(max_ps, end_ps);
  }
  return (min_ps < max_ps) ? Timespan::FromEndPoints(min_ps, max_ps)
                           : Timespan();
}

// Returns the timespan across all steps in the given step_db.
Timespan AllStepsTimespan(const StepDatabaseResult& step_db) {
  uint64 min_ps = kuint64max;
  uint64 max_ps = 0;
  for (const auto& step : step_db.step_sequence()) {
    Timespan timespan = StepTimespan(step);
    uint64 begin_ps = timespan.begin_ps();
    uint64 end_ps = timespan.end_ps();
    min_ps = std::min(min_ps, begin_ps);
    max_ps = std::max(max_ps, end_ps);
  }
  return (min_ps < max_ps) ? Timespan::FromEndPoints(min_ps, max_ps)
                           : Timespan();
}

struct AlignmentInfo {
  StepsAlignment alignment;
  double similarity;
};

// Computes the similarity between the given two steps. The closer their
// timespans are, the larger is the similarity.
double StepSimilarity(const PerCoreStepInfo& subordinate_step,
                      const PerCoreStepInfo& chief_step) {
  Timespan subordinate_timespan = StepTimespan(subordinate_step);
  Timespan chief_timespan = StepTimespan(chief_step);
  return chief_timespan.OverlappedDurationPs(subordinate_timespan);
}

// If the subordinate steps and the chief steps are aligned at the given anchor
// points (i.e. at the subordinate_anchor step on the subordinate sequence, at
// the chief_anchor step on the chief sequence), returns the corresponding
// AlignmentInfo.
AlignmentInfo ComputeAlignmentInfo(const StepDatabaseResult& subordinate,
                                   uint32 subordinate_anchor,
                                   const StepDatabaseResult& chief,
                                   uint32 chief_anchor) {
  // Assumes that the step at subordinate_anchor on the subordinate sequence is
  // aligned with the step at the chief_anchor on the chief sequence. Then the
  // number of steps before the anchor is the minimum of the number of steps
  // before the anchor in the subordinate and that before the anchor in the
  // chief. Similarly, the number of steps after the anchor is the minimum of
  // the number of steps after the anchor in the subordinate and that after the
  // anchor in the chief.
  uint32 pre_anchor_steps = std::min(subordinate_anchor, chief_anchor);
  uint32 post_anchor_steps =
      std::min(subordinate.step_sequence_size() - subordinate_anchor,
               chief.step_sequence_size() - chief_anchor);
  // total number of steps aligned = pre_anchor_steps + post_anchor_steps.
  uint32 alignment_steps = pre_anchor_steps + post_anchor_steps;

  double similarity = 0;
  // Where the aligned steps begin on the subordinate sequence.
  uint32 begin_subordinate_idx = subordinate_anchor - pre_anchor_steps;
  // Where the aligned steps begin on the chief sequence.
  uint32 begin_chief_idx = chief_anchor - pre_anchor_steps;

  for (uint32 i = 0; i < alignment_steps; i++) {
    // Accumulates the similarity at each step.
    similarity +=
        StepSimilarity(subordinate.step_sequence(begin_subordinate_idx + i),
                       chief.step_sequence(begin_chief_idx + i));
  }
  StepsAlignment alignment = {begin_subordinate_idx, begin_chief_idx,
                              alignment_steps};
  return {alignment, similarity};
}

// Returns the best alignment for aligning subordinate against chief.
StepsAlignment FindStepsAlignment(const StepDatabaseResult& subordinate,
                                  const StepDatabaseResult& chief) {
  double max_similarity = -1;
  StepsAlignment alignment = {0, 0, 0};
  if (subordinate.step_sequence_size() == 0 || chief.step_sequence_size() == 0)
    return alignment;
  for (auto c = 0; c < chief.step_sequence_size(); c++) {
    AlignmentInfo info =
        ComputeAlignmentInfo(subordinate, /*subordinate_anchor=*/0, chief, c);
    if (info.similarity <= max_similarity) continue;
    max_similarity = info.similarity;
    alignment = info.alignment;
  }
  for (auto s = 1; s < subordinate.step_sequence_size(); s++) {
    // s starts at 1 instead of 0, because the loop above already considers
    // (s=0, c=0).
    AlignmentInfo info =
        ComputeAlignmentInfo(subordinate, s, chief, /*chief_anchor=*/0);
    if (info.similarity <= max_similarity) continue;
    max_similarity = info.similarity;
    alignment = info.alignment;
  }
  return alignment;
}

std::string StringStepsAlignment(const StepsAlignment& alignment) {
  return absl::StrCat(
      "[begin_subordinate_idx: ", alignment.begin_subordinate_idx,
      ", begin_chief_idx: ", alignment.begin_chief_idx,
      ", num_steps: ", alignment.num_steps, "]");
}

std::string StringDstStepNumbers(const std::vector<uint32>& step_numbers) {
  std::string str;
  absl::StrAppend(&str, "[");
  for (auto i = 0; i < step_numbers.size(); i++) {
    if (i > 0) absl::StrAppend(&str, ", ");
    absl::StrAppend(&str, step_numbers[i]);
  }
  absl::StrAppend(&str, "]");
  return str;
}

std::string StringSrcToDstIndexMap(uint32 src_first_step_idx,
                                   uint32 num_steps) {
  std::string str;
  absl::StrAppend(&str, "[");
  for (auto i = 0; i < num_steps; i++) {
    if (i > 0) absl::StrAppend(&str, ", ");
    absl::StrAppend(&str, src_first_step_idx + i, ":", i);
  }
  absl::StrAppend(&str, "]");
  return str;
}

}  // namespace

StepIntersection::StepIntersection(
    uint32 max_steps,
    const absl::flat_hash_map<uint32, const StepDatabaseResult*>&
        perhost_stepdb) {
  empty_intersect_ = false;

  // Figures out the host with the shortest timespan among their steps (called
  // this host the "chief").
  chief_host_id_ = kuint32max;
  uint64 min_duration_ps = kuint64max;
  const StepDatabaseResult* chief_step_db = nullptr;
  for (const auto& hostid_stepdb : perhost_stepdb) {
    auto host_id = hostid_stepdb.first;
    const auto& step_db = hostid_stepdb.second;
    Timespan timespan = AllStepsTimespan(*step_db);
    if (timespan.duration_ps() < min_duration_ps) {
      chief_host_id_ = host_id;
      chief_step_db = step_db;
      min_duration_ps = timespan.duration_ps();
    }
  }
  if (chief_host_id_ == kuint32max) {
    // There is no step at all on any host.
    steps_dropped_ = 0;
    begin_chief_idx_ = 0;
    end_chief_idx_ = 0;
    return;
  }

  uint32 max_begin_chief_idx = 0;
  uint32 min_end_chief_idx = kuint32max;
  // Aligns the steps in all hosts with those in the chief.
  for (const auto& hostid_stepdb : perhost_stepdb) {
    auto host_id = hostid_stepdb.first;
    const auto& step_db = hostid_stepdb.second;
    if (host_id == chief_host_id_) {
      // Simply aligns with itself.
      perhost_alignment_[host_id] = {
          /*begin_subordinate_idx=*/0, /*begin_chief_idx=*/0,
          static_cast<uint32>(step_db->step_sequence_size())};
    } else {
      perhost_alignment_[host_id] =
          FindStepsAlignment(*step_db, *chief_step_db);
    }
    // Intersects this host's alignment with other hosts' alignments.
    uint32 host_begin_chief_idx = perhost_alignment_[host_id].begin_chief_idx;
    max_begin_chief_idx = std::max(max_begin_chief_idx, host_begin_chief_idx);
    uint32 host_end_chief_idx = perhost_alignment_[host_id].begin_chief_idx +
                                perhost_alignment_[host_id].num_steps;
    min_end_chief_idx = std::min(min_end_chief_idx, host_end_chief_idx);
  }
  if (max_begin_chief_idx > min_end_chief_idx) {
    // The intersection is empty.
    steps_dropped_ = 0;
    begin_chief_idx_ = 0;
    end_chief_idx_ = 0;
    empty_intersect_ = true;
    return;
  }

  begin_chief_idx_ = max_begin_chief_idx;

  // Takes max_steps into account.
  uint32 num_steps = min_end_chief_idx - max_begin_chief_idx;
  if (num_steps > max_steps) {
    steps_dropped_ = num_steps - max_steps;
    // TODO(ckluk): Drops from both ends to avoid incomplete steps at the
    // beginning and end of the profile.
    end_chief_idx_ = max_begin_chief_idx + max_steps;
  } else {
    steps_dropped_ = 0;
    end_chief_idx_ = min_end_chief_idx;
  }
}

std::vector<uint32> StepIntersection::DstStepNumbers() const {
  // TODO(ckluk): Honors training-loop boundaries (if more than one loop
  // sampled).
  std::vector<uint32> result;
  result.reserve(NumSteps());
  for (uint32 i = 0; i < NumSteps(); i++) {
    result.push_back(i);
  }
  return result;
}

uint32 StepIntersection::FirstStepIndex(uint32 host_id) const {
  const auto* alignment = gtl::FindOrNull(perhost_alignment_, host_id);
  if (alignment == nullptr) return 0;
  DCHECK(alignment->begin_chief_idx <= begin_chief_idx_);
  uint32 shift = begin_chief_idx_ - alignment->begin_chief_idx;
  uint32 begin_subordinate_idx = alignment->begin_subordinate_idx + shift;
  return begin_subordinate_idx;
}

std::string StepIntersection::DebugString() const {
  std::string str;
  absl::StrAppend(&str, "chief host id_: ", chief_host_id_, "\n");
  absl::StrAppend(&str, "begin_chief_idx_: ", begin_chief_idx_,
                  ", num_steps: ", NumSteps(), "\n");
  absl::StrAppend(
      &str, "DstStepNumbers(): ", StringDstStepNumbers(DstStepNumbers()), "\n");

  std::vector<uint32> host_ids;
  host_ids.reserve(perhost_alignment_.size());
  for (const auto& hostid_alignment : perhost_alignment_) {
    auto host_id = hostid_alignment.first;
    host_ids.push_back(host_id);
  }
  absl::c_sort(host_ids);

  absl::StrAppend(&str, "perhost_alignment:\n");
  for (const auto host_id : host_ids) {
    const auto* ptr = gtl::FindOrNull(perhost_alignment_, host_id);
    if (ptr == nullptr) continue;
    absl::StrAppend(&str, "host: ", host_id,
                    ", step-alignment: ", StringStepsAlignment(*ptr), "\n");
  }
  absl::StrAppend(&str, "SrcToDstIndexMap():\n");
  for (const auto host_id : host_ids) {
    absl::StrAppend(&str, "host: ", host_id, ", src-to-dst-index-map: ",
                    StringSrcToDstIndexMap(FirstStepIndex(host_id), NumSteps()),
                    "\n");
  }
  return str;
}

}  // namespace profiler
}  // namespace tensorflow
