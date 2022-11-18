/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_TUNING_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_TUNING_UTILS_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/heap_simulator.h"

namespace xla {

namespace memory_space_assignment {

using BufferInterval =
    GlobalDecreasingSizeBestFitHeap<HloValue>::BufferInterval;

// This function converts sorted_buffer_intervals to order-customized buffer
// intervals respecting a given memory space assignment autotuning config.
void CustomizeSortedBufferInterval(
    std::optional<std::vector<uint64_t>> memory_space_assignment_config,
    std::vector<BufferInterval>& sorted_buffer_intervals);

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_TUNING_UTILS_H_
