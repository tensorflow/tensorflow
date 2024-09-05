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

#include "xla/service/memory_space_assignment/slice.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/time_utils.h"
#include "xla/shape.h"

namespace xla::memory_space_assignment {

std::tuple<const HeapSimulator::Chunk&, int64_t, const SliceProposal&, float>
SliceDecisionToTuple(const SliceDecision& decision) {
  return std::make_tuple(
      std::ref(decision.chunk), decision.exclusive_start_time,
      std::ref(decision.sizing), decision.copy_resource_consumed);
}

std::string SliceDecision::ToString() const {
  return absl::StrCat("{ chunk: ", chunk.ToString(),
                      ", (exclusive) start_time: ", exclusive_start_time,
                      ", sizing: ", sizing.ToString(),
                      ", copy_resource_consumed: ", copy_resource_consumed,
                      " }");
}

bool SliceDecision::operator==(const SliceDecision& other) const {
  return SliceDecisionToTuple(*this) == SliceDecisionToTuple(other);
}

std::string SliceProposal::ToString() const {
  return absl::StrCat(
      "{ slice_shape: ", slice_shape.ToString(true), ", slice_params: { ",
      absl::StrJoin(slice_params, ", ",
                    [](std::string* out, const SliceParam& param) {
                      absl::StrAppend(out, param.ToString());
                    }),
      " }, slice_size: ", slice_size, " }");
}

std::ostream& operator<<(std::ostream& os, const SliceProposal& proposal) {
  os << proposal.ToString();
  return os;
}

std::tuple<const Shape&, const std::vector<SliceParam>&, int64_t>
SliceProposal::ToTuple() const {
  return std::make_tuple(std::ref(slice_shape), std::ref(slice_params),
                         slice_size);
}

bool SliceProposal::operator==(const SliceProposal& other) const {
  return ToTuple() == other.ToTuple();
}

std::string SliceParam::ToString() const {
  return absl::StrCat("[", start_inclusive, ",", end_exclusive, ")");
}

bool SliceParam::operator==(const SliceParam& other) const {
  return start_inclusive == other.start_inclusive &&
         end_exclusive == other.end_exclusive;
}

bool IsUniformSliceSizingEnabled(const SlicedPrefetchOptions& options) {
  return options.max_slices() > 0 && options.preferred_slice_size() > 0;
}

std::vector<int64_t> SlicedPrefetchStartTimePicker::Pick(
    int64_t num_slices, int64_t exclusive_prefetch_start_time,
    int64_t prefetch_end_time, absl::AnyInvocable<ElapsedTimeFn> elapsed_fn,
    absl::AnyInvocable<SameComputationParentFn> has_same_parent_fn) {
  CHECK_LE(exclusive_prefetch_start_time, prefetch_end_time);
  VLOG(5) << "Picking slice start times. num_slices = " << num_slices
          << "; exclusive_prefetch_start_time = "
          << exclusive_prefetch_start_time
          << "; prefetch_end_time = " << prefetch_end_time;

  // Prefetching starts after the selected start instruction and ends
  // before the selected end instruction. Thus, we have (end - (start + 1)) HLO
  // instructions worth of time to perform all of the sliced copies. So, the
  // only choices for start times that give us time to copy are <=
  // prefetch_end_time - 2.
  if (exclusive_prefetch_start_time >= prefetch_end_time - 2 ||
      num_slices == 1) {
    return std::vector<int64_t>(num_slices, exclusive_prefetch_start_time);
  }

  float total_elapsed =
      elapsed_fn(exclusive_prefetch_start_time, prefetch_end_time);
  if (total_elapsed <= 0.0) {
    return std::vector<int64_t>(num_slices, exclusive_prefetch_start_time);
  }

  std::vector<int64_t> start_times;
  start_times.reserve(num_slices);
  start_times.push_back(exclusive_prefetch_start_time);
  int64_t last_valid_candidate = exclusive_prefetch_start_time;
  int64_t candidate = exclusive_prefetch_start_time;
  while (candidate < prefetch_end_time - 1 && start_times.size() < num_slices) {
    float target_elapsed = total_elapsed *
                           static_cast<float>(num_slices - start_times.size()) /
                           static_cast<float>(num_slices);
    float elapsed = elapsed_fn(candidate, prefetch_end_time);
    if (elapsed < target_elapsed) {
      // We've gone past our target, so use the last valid candidate.
      start_times.push_back(last_valid_candidate);
      continue;
    }
    bool updating_candidate_impacts_elapsed =
        last_valid_candidate != candidate &&
        elapsed_fn(last_valid_candidate,
                   ExclusiveToInclusiveStartTime(candidate)) > 0.0;
    // has_same_parent_fn will look up the computation parent of the
    // instructions at prefetch_start_time and prefetch_end_time. If
    // prefetch_start_time is -1, no such instruction will exist. However, if we
    // want to insert an instruction after the -1 schedule position, we can
    // use the parent of the instruction at index 0 instead. Thus, we use
    // std::max below.
    if (has_same_parent_fn(std::max<int64_t>(0, exclusive_prefetch_start_time),
                           std::max<int64_t>(0, candidate)) &&
        updating_candidate_impacts_elapsed) {
      last_valid_candidate = candidate;
    }
    ++candidate;
  }
  while (start_times.size() < num_slices) {
    start_times.push_back(last_valid_candidate);
  }

  return start_times;
}

}  // namespace xla::memory_space_assignment
