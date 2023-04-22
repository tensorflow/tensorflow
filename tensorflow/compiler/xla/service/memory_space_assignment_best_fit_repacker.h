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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BEST_FIT_REPACKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BEST_FIT_REPACKER_H_

#include "tensorflow/compiler/xla/service/heap_simulator.h"
#include "tensorflow/compiler/xla/service/memory_space_assignment_repacking.h"

namespace xla {

// This is a repacker algorithm that wraps around best fit heap algorithm in
// heap simulator.
class MemorySpaceAssignmentBestFitRepacker
    : public MemorySpaceAssignmentRepacker {
 public:
  using Type = GlobalDecreasingSizeBestFitHeap<AllocationBlock>::Type;

  explicit MemorySpaceAssignmentBestFitRepacker(
      int64 max_size, int64 alignment,
      Type type = GlobalDecreasingSizeBestFitHeap<AllocationBlock>::kTemporal)
      : MemorySpaceAssignmentRepacker(max_size, alignment), type_(type) {}

  StatusOr<bool> Repack(absl::Span<AllocationBlock*> allocations) override;

 private:
  Type type_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_BEST_FIT_REPACKER_H_
