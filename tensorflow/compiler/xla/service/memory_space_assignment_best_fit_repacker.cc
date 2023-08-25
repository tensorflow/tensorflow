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

#include "tensorflow/compiler/xla/service/memory_space_assignment_best_fit_repacker.h"

#include <algorithm>
#include <functional>
#include <tuple>

#include "tensorflow/compiler/xla/service/heap_simulator.h"

namespace xla {

namespace {

using AllocationBlock = MemorySpaceAssignmentRepacker::AllocationBlock;
using Type = GlobalDecreasingSizeBestFitHeap<AllocationBlock>::Type;

// This class inherits GlobalDecreasingSizeBestFitHeap and converts
// AllocationBlock objects into BufferIntervals that the heap algorithm
// understands.
class BestFitRepacker
    : public GlobalDecreasingSizeBestFitHeap<AllocationBlock> {
 public:
  BestFitRepacker(int64_t max_size, int64_t alignment, Type type)
      : GlobalDecreasingSizeBestFitHeap<AllocationBlock>(alignment, type),
        max_size_(max_size) {}

  void ImportAllocationBlocks(absl::Span<AllocationBlock*> allocations) {
    allocation_blocks_ = allocations;
    for (AllocationBlock* allocation_block : allocations) {
      // Check if any of the colocations are already added to buffer_intervals_.
      bool need_allocation = true;
      auto aliased_it = absl::c_find_if(
          allocation_block->colocations, [&](AllocationBlock* search) {
            return buffer_intervals_.contains(search);
          });
      if (aliased_it != allocation_block->colocations.end()) {
        buffer_intervals_[*aliased_it].colocations.push_back(allocation_block);
        need_allocation = false;
      }
      buffer_intervals_[allocation_block] = {allocation_block,
                                             allocation_block->size,
                                             allocation_block->start_time,
                                             allocation_block->end_time,
                                             {},
                                             need_allocation};
    }
  }

  // Sorting by initial offset gives better buffer order stability between
  // related programs improving the success rate for cross-program prefetching.
  BufferIntervalCompare GetTemporalBufferIntervalCompare() const override {
    return LessThanByKey([this](const BufferInterval& x) {
      int64_t x_end = x.end;
      for (auto colocation : GetTransitiveColocations(x)) {
        x_end = std::max(x_end, buffer_intervals_.at(colocation).end);
      }
      // Sort by duration (descending), size (descending), initial offset
      // (ascending), buffer (ascending).
      return std::make_tuple(x.start - x_end, -x.size, x.buffer->initial_offset,
                             std::cref(*x.buffer));
    });
  }

  bool Repack() {
    Finish();
    bool success = result_.heap_size <= max_size_;
    if (success) {
      for (AllocationBlock* block : allocation_blocks_) {
        auto chunk_it = result_.chunk_map.find(block);
        if (chunk_it != result_.chunk_map.end()) {
          block->offset = chunk_it->second.offset;
        }
      }
    }
    return success;
  }

 private:
  int64_t max_size_;
  absl::Span<AllocationBlock*> allocation_blocks_;
};

}  // namespace

StatusOr<bool> MemorySpaceAssignmentBestFitRepacker::Repack(
    absl::Span<AllocationBlock*> allocations) {
  BestFitRepacker best_fit_repacker =
      BestFitRepacker(max_size_, alignment_, type_);
  best_fit_repacker.ImportAllocationBlocks(allocations);
  return best_fit_repacker.Repack();
}

}  // namespace xla
