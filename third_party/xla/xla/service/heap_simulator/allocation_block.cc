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

#include "xla/service/heap_simulator/allocation_block.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tsl/platform/logging.h"

namespace xla {

std::string AllocatedSlice::ToString() const {
  return absl::StrCat("{ size: ", size, ", offset: ", offset,
                      ", inclusive_start_time: ", inclusive_start_time, " }");
}

std::tuple<int64_t, int64_t, int64_t> AllocatedSlice::ToTuple() const {
  return std::make_tuple(size, offset, inclusive_start_time);
}

bool AllocatedSlice::operator==(const AllocatedSlice& rhs) const {
  return ToTuple() == rhs.ToTuple();
}

std::vector<int64_t> SlicedAllocationData::SizesSortedByOffset() const {
  std::vector<int64_t> sizes_sorted_by_offset;
  sizes_sorted_by_offset.reserve(slices_sorted_by_offset.size());
  absl::c_for_each(slices_sorted_by_offset,
                   [&sizes_sorted_by_offset](const AllocatedSlice& slice) {
                     sizes_sorted_by_offset.push_back(slice.size);
                   });
  return sizes_sorted_by_offset;
}

std::vector<int64_t> SlicedAllocationData::SortedInclusiveStartTimes() const {
  std::vector<int64_t> sorted_inclusive_start_times;
  sorted_inclusive_start_times.reserve(slices_sorted_by_offset.size());
  absl::c_for_each(slices_sorted_by_offset, [&sorted_inclusive_start_times](
                                                const AllocatedSlice& slice) {
    sorted_inclusive_start_times.push_back(slice.inclusive_start_time);
  });
  absl::c_sort(sorted_inclusive_start_times);
  return sorted_inclusive_start_times;
}

std::string SlicedAllocationData::ToString() const {
  return absl::StrCat(
      "{ slices_sorted_by_offset: [ ",
      absl::StrJoin(slices_sorted_by_offset, ", ",
                    [](std::string* out, const AllocatedSlice& slice) {
                      absl::StrAppend(out, slice.ToString());
                    }),
      " ] }");
}

bool SlicedAllocationData::operator==(const SlicedAllocationData& rhs) const {
  return slices_sorted_by_offset == rhs.slices_sorted_by_offset;
}

std::string AllocationBlock::ToString() const {
  std::string original_slicing_str;
  if (original_slice_data.has_value()) {
    original_slicing_str = absl::StrCat("; original_slice_data: ",
                                        original_slice_data->ToString());
  }
  std::string repacked_slicing_str;
  if (repacked_slice_data.has_value()) {
    repacked_slicing_str = absl::StrCat("; repacked_slice_data: ",
                                        repacked_slice_data->ToString());
  }
  return absl::StrCat("[", inclusive_start_time, ", ", end_time,
                      "]; size: ", size, "; offset: ", offset,
                      "; initial offset: ", initial_offset,
                      "; # colocations: ", GetColocationsCount(),
                      original_slicing_str, repacked_slicing_str);
}

int AllocationBlock::GetColocationsCount() const {
  int count = 1;
  for (const AllocationBlock* colocated = next_colocated; colocated != this;
       colocated = colocated->next_colocated, ++count) {
    CHECK_NE(colocated, nullptr);
  }
  return count;
}

std::vector<AllocationBlock*> AllocationBlock::GetColocations() {
  std::vector<AllocationBlock*> colocations{this};
  for (AllocationBlock* colocated = next_colocated; colocated != this;
       colocated = colocated->next_colocated) {
    CHECK_NE(colocated, nullptr);
    colocations.push_back(colocated);
  }
  return colocations;
}

bool AllocationBlock::operator<(const AllocationBlock& other) const {
  return id < other.id;
}

}  // namespace xla
