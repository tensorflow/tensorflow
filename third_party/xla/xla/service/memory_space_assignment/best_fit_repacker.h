/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BEST_FIT_REPACKER_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BEST_FIT_REPACKER_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/memory_space_assignment/repacking.h"
#include "xla/statusor.h"

namespace xla {
namespace memory_space_assignment {

// This is a repacker algorithm that wraps around best fit heap algorithm in
// heap simulator.
class MemorySpaceAssignmentBestFitRepacker
    : public MemorySpaceAssignmentRepacker {
 public:
  using BufferInterval =
      GlobalDecreasingSizeBestFitHeap<AllocationBlock>::BufferInterval;
  using BufferIntervalCompare =
      GlobalDecreasingSizeBestFitHeap<AllocationBlock>::BufferIntervalCompare;

  struct BestFitRepackOptions {
    // Running the validator is potentially expensive.
    bool validate = false;

    // Specify the comparison function used for determining the order in which
    // buffers will be allocated, during repacking.
    BufferIntervalCompare buffer_interval_compare = nullptr;
  };

  MemorySpaceAssignmentBestFitRepacker(
      int64_t max_size, int64_t alignment,
      SliceTimePermutationIterator::Ty slice_time_permutation_iterator_type)
      : MemorySpaceAssignmentRepacker(max_size, alignment),
        options_(BestFitRepackOptions()),
        slice_time_permutation_iterator_type_(
            slice_time_permutation_iterator_type) {}
  MemorySpaceAssignmentBestFitRepacker(
      int64_t max_size, int64_t alignment,
      SliceTimePermutationIterator::Ty slice_time_permutation_iterator_type,
      BestFitRepackOptions options)
      : MemorySpaceAssignmentRepacker(max_size, alignment),
        options_(std::move(options)),
        slice_time_permutation_iterator_type_(
            slice_time_permutation_iterator_type) {}

  absl::StatusOr<bool> Repack(
      absl::Span<AllocationBlock*> allocations) override;

 private:
  BestFitRepackOptions options_;
  SliceTimePermutationIterator::Ty slice_time_permutation_iterator_type_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BEST_FIT_REPACKER_H_
