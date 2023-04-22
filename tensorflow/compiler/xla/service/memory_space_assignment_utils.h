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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_

#include "tensorflow/compiler/xla/service/heap_simulator.h"

namespace xla {

// Encapsulates common utility methods for memory space assignment.
class MemorySpaceAssignmentUtils {
 public:
  // Returns true if this buffer is allowed to be placed in the alternate
  // memory.
  static bool IsIntervalAllowedInAlternateMemory(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval&
          interval);

  // Returns true if the HloValue is allowed to be placed in alternate memory.
  static bool IsValueAllowedInAlternateMemory(const HloValue* value);

  // Modifies the schedules in the given module to hoist (move earlier) constant
  // operations. This increases the opportunities to prefetch constant ops.
  static void HoistConstantOperations(HloModule& module);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_
