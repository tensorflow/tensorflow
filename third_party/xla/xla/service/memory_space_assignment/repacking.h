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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/statusor.h"

namespace xla {
namespace memory_space_assignment {

// An interface to define allocation repacking algorithms.
class MemorySpaceAssignmentRepacker {
 public:
  MemorySpaceAssignmentRepacker(int64_t max_size, int64_t alignment)
      : max_size_(max_size), alignment_(alignment) {}
  virtual ~MemorySpaceAssignmentRepacker() = default;

  // Repack the AllocationBlocks provided in the parameter. Returns true if
  // allocations have been modified and false if not. Returns a non-ok status if
  // there was an error.
  virtual absl::StatusOr<bool> Repack(
      absl::Span<AllocationBlock*> allocations) = 0;

 protected:
  int64_t max_size_;
  int64_t alignment_;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_REPACKING_H_
