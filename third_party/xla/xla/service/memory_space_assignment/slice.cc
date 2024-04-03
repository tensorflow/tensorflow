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

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/service/heap_simulator/heap_simulator.h"
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

}  // namespace xla::memory_space_assignment
