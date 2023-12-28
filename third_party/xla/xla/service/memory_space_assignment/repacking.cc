/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/memory_space_assignment/repacking.h"

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"

namespace xla {
namespace memory_space_assignment {

std::string MemorySpaceAssignmentRepacker::AllocationBlock::ToString() const {
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

int MemorySpaceAssignmentRepacker::AllocationBlock::GetColocationsCount()
    const {
  int count = 1;
  for (const AllocationBlock* colocated = next_colocated; colocated != this;
       colocated = colocated->next_colocated, ++count) {
    CHECK_NE(colocated, nullptr);
  }
  return count;
}

std::vector<MemorySpaceAssignmentRepacker::AllocationBlock*>
MemorySpaceAssignmentRepacker::AllocationBlock::GetColocations() {
  std::vector<AllocationBlock*> colocations{this};
  for (AllocationBlock* colocated = next_colocated; colocated != this;
       colocated = colocated->next_colocated) {
    CHECK_NE(colocated, nullptr);
    colocations.push_back(colocated);
  }
  return colocations;
}

}  // namespace memory_space_assignment
}  // namespace xla
