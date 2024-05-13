/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/memory_space_assignment/buffer_interval_comparator.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {
namespace {

bool DoesResultMatchFilter(const HloPositionMatcher& filter,
                           const MsaBufferInterval& buffer_interval) {
  HloInstruction* instruction = buffer_interval.buffer->instruction();
  if (filter.has_instruction_regex() &&
      !RE2::FullMatch(instruction->ToString(), filter.instruction_regex())) {
    return false;
  }
  if (filter.has_instruction_name_regex() &&
      !RE2::FullMatch(instruction->name(), filter.instruction_name_regex())) {
    return false;
  }
  if (filter.has_tuple_index() &&
      buffer_interval.buffer->index() !=
          ShapeIndex(filter.tuple_index().index().begin(),
                     filter.tuple_index().index().end())) {
    return false;
  }
  if (filter.has_size_gte() && filter.size_gte() > buffer_interval.size) {
    return false;
  }
  if (filter.has_size_lte() && filter.size_lte() < buffer_interval.size) {
    return false;
  }
  return true;
}

// Returns an integer representing the priority of a MsaBufferInterval during
// assignment, a smaller number indicates a higher priority.
int64_t GetBufferIntervalOverridePriority(
    const MsaSortOrderOverrides& msa_sort_order_overrides,
    const MsaBufferInterval& buffer_interval) {
  if (msa_sort_order_overrides.overrides_size() == 0) {
    return 0;
  }
  for (int64_t i = 0; i < msa_sort_order_overrides.overrides_size(); ++i) {
    const auto& override = msa_sort_order_overrides.overrides(i);
    if (!DoesResultMatchFilter(override.hlo_position_matcher(),
                               buffer_interval)) {
      continue;
    }
    LOG(INFO) << "Override Sort Order Config " << i << " matches "
              << buffer_interval.buffer->instruction()->ToString();
    switch (override.override_options().options_case()) {
      case MsaSortOrderOverrideOptions::kAssignFirst:
        return std::numeric_limits<int64_t>::lowest() + i;
      case MsaSortOrderOverrideOptions::kAssignLast:
        return std::numeric_limits<int64_t>::max() - i;
      case MsaSortOrderOverrideOptions::OPTIONS_NOT_SET:
        continue;
    }
  }
  return 0;
}

}  // namespace

MemoryBoundednessBufferIntervalComparator::
    MemoryBoundednessBufferIntervalComparator(
        const CostAnalysis& cost_analysis,
        CostAnalysis::Cache* cost_analysis_cache)
    : BufferIntervalComparator(),
      cost_analysis_(cost_analysis),
      cost_analysis_cache_(cost_analysis_cache) {}

MemoryBoundednessBufferIntervalComparator::
    MemoryBoundednessBufferIntervalComparator(
        const CostAnalysis& cost_analysis,
        CostAnalysis::Cache* cost_analysis_cache,
        MsaSortOrderOverrides msa_sort_order_overrides)
    : BufferIntervalComparator(),
      cost_analysis_(cost_analysis),
      cost_analysis_cache_(cost_analysis_cache),
      msa_sort_order_overrides_(msa_sort_order_overrides) {}

std::string
MemoryBoundednessBufferIntervalComparator::DescribeComparisonCriteria() const {
  return "[override priority, -memory boundedness, -size, -buffer duration, "
         "latest use time, (inclusive) start time, instruction id ]";
}

std::string MemoryBoundednessBufferIntervalComparator::CriteriaToString(
    const MsaBufferInterval& buffer_interval) {
  return absl::StrCat("[ ", absl::StrJoin(GetTuple(buffer_interval), ", "),
                      " ]");
}

bool MemoryBoundednessBufferIntervalComparator::LessThan(
    const MsaBufferInterval& lhs, const MsaBufferInterval& rhs) {
  return GetTuple(lhs) < GetTuple(rhs);
}

int64_t MemoryBoundednessBufferIntervalComparator::GetLatestUseTime(
    const MsaBufferInterval& buffer_interval) {
  auto latest_use_it = buffer_to_latest_use_.find(buffer_interval.buffer);
  if (latest_use_it == buffer_to_latest_use_.end()) {
    int64_t latest_use_time = 0;
    for (const HloUse& use : buffer_interval.buffer->GetUses()) {
      auto it = cost_analysis_.hlo_live_range().instruction_schedule().find(
          use.instruction);
      if (it != cost_analysis_.hlo_live_range().instruction_schedule().end()) {
        latest_use_time = std::max(latest_use_time, it->second);
      }
    }
    latest_use_it =
        buffer_to_latest_use_
            .insert(std::make_pair(buffer_interval.buffer, latest_use_time))
            .first;
  }
  return latest_use_it->second;
}

MemoryBoundednessBufferIntervalComparator::ComparisonTuple
MemoryBoundednessBufferIntervalComparator::GetTuple(
    const MsaBufferInterval& buffer_interval) {
  int64_t priority = GetBufferIntervalOverridePriority(
      msa_sort_order_overrides_, buffer_interval);
  float inverse_memory_boundedness =
      -1.0 * cost_analysis_.GetMemoryBoundedness(buffer_interval,
                                                 cost_analysis_cache_);
  int64_t inverse_buffer_size = -1 * buffer_interval.size;
  int64_t inverse_buffer_duration = buffer_interval.start - buffer_interval.end;
  int64_t latest_use_time = GetLatestUseTime(buffer_interval);
  int64_t buffer_start_time = buffer_interval.start;
  auto buffer_id = buffer_interval.buffer->id();
  return std::make_tuple(priority, inverse_memory_boundedness,
                         inverse_buffer_size, inverse_buffer_duration,
                         latest_use_time, buffer_start_time, buffer_id);
}

DefaultCrossProgramPrefetchBufferIntervalComparator::
    DefaultCrossProgramPrefetchBufferIntervalComparator(
        const HloLiveRange& hlo_live_range)
    : BufferIntervalComparator(), hlo_live_range_(hlo_live_range) {}

std::string DefaultCrossProgramPrefetchBufferIntervalComparator::
    DescribeComparisonCriteria() const {
  return "[ -size, -cumulative use size, latest use, instruction id]";
}

std::string
DefaultCrossProgramPrefetchBufferIntervalComparator::CriteriaToString(
    const MsaBufferInterval& buffer_interval) {
  return absl::StrCat("[ ", absl::StrJoin(GetTuple(buffer_interval), ", "),
                      " ]");
}

bool DefaultCrossProgramPrefetchBufferIntervalComparator::LessThan(
    const MsaBufferInterval& lhs, const MsaBufferInterval& rhs) {
  return GetTuple(lhs) < GetTuple(rhs);
}

DefaultCrossProgramPrefetchBufferIntervalComparator::ComparisonTuple
DefaultCrossProgramPrefetchBufferIntervalComparator::GetTuple(
    const MsaBufferInterval& buffer_interval) {
  auto sort_data_it = additional_sort_data_.find(buffer_interval.buffer);
  if (sort_data_it == additional_sort_data_.end()) {
    AdditionalSortData sort_data;
    absl::c_for_each(buffer_interval.buffer->GetUses(), [&](const HloUse& use) {
      auto it = hlo_live_range_.instruction_schedule().find(use.instruction);
      if (it == hlo_live_range_.instruction_schedule().end()) {
        return;
      }
      sort_data.latest_use = std::max(sort_data.latest_use, it->second);
      sort_data.cumulative_use_size +=
          ShapeUtil::ElementsInRecursive(use.instruction->shape());
    });
    sort_data_it =
        additional_sort_data_.try_emplace(buffer_interval.buffer, sort_data)
            .first;
  }

  return std::make_tuple(
      -1 * buffer_interval.size, -1 * sort_data_it->second.cumulative_use_size,
      sort_data_it->second.latest_use, buffer_interval.buffer->id());
}

}  // namespace memory_space_assignment
}  // namespace xla
