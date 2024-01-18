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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_

#include "xla/service/heap_simulator/heap_simulator.h"

namespace xla {
namespace memory_space_assignment {

// Encapsulates common utility methods for memory space assignment.
class MemorySpaceAssignmentUtils {
 public:
  // Returns true if this buffer is allowed to be placed in the alternate
  // memory.
  static bool IsIntervalAllowedInAlternateMemory(
      const GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval& interval,
      int64_t alternate_memory_space);

  // Returns true if the HloValue is allowed to be placed in alternate memory.
  static bool IsValueAllowedInAlternateMemory(const HloValue* value,
                                              int64_t alternate_memory_space);
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_UTILS_H_
