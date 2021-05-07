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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// An interface to define allocation repacking algorithms.
class MemorySpaceAssignmentRepacker {
 public:
  MemorySpaceAssignmentRepacker(int64 max_size, int64 alignment)
      : max_size_(max_size), alignment_(alignment) {}
  virtual ~MemorySpaceAssignmentRepacker() = default;

  // A contiguous block of allocation consisting of start and end (logical)
  // times, size, and the initial offset. After repacking, if the repacking was
  // successful and the allocations were modified, the offset field holds the
  // new offset. To support aliased allocations, AllocationBlock also includes a
  // vector of AllocationBlock pointers, called colocations. All AllocationBlock
  // objects within the colocations must get the same offset. The id should be
  // unique and is used to ensure determinism for comparison tie-breaker.
  struct AllocationBlock {
    int64 start_time;
    int64 end_time;
    int64 size;
    int64 offset;
    int64 initial_offset;
    int64 id;
    std::vector<AllocationBlock*> colocations;

    std::string ToString() const {
      return absl::StrCat("[", start_time, ", ", end_time, "] : size = ", size,
                          ", offset = ", offset,
                          " initial offset = ", initial_offset);
    }

    // This is required by BufferIntervalCompare as a tie breaker. Use a unique
    // and deterministic id.
    bool operator<(const AllocationBlock& other) const { return id < other.id; }
  };

  // Repack the AllocationBlocks provided in the parameter. Returns true if
  // allocations have been modified and false if not. Returns a non-ok status if
  // there was an error.
  virtual StatusOr<bool> Repack(absl::Span<AllocationBlock*> allocations) = 0;

 protected:
  int64 max_size_;
  int64 alignment_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_
