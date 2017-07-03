/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULING_H_

#include <vector>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Returns the minimum memory required to compute the given module sequence,
// assuming no fragmentation.
StatusOr<int64> MinimumMemoryForSequence(
    const SequentialHloOrdering::HloModuleSequence& module_sequence,
    const LogicalBuffer::SizeFunction& size_function);

// Returns an HloModuleSequence which seeks to minimize the memory required for
// the computation. size_function is the function returning the number of bytes
// required for a LogicalBuffer.
StatusOr<SequentialHloOrdering::HloModuleSequence>
CreateMemoryMinimizingSequence(
    const HloModule& module, const LogicalBuffer::SizeFunction& size_function);

// Overload of above that computes the sequence for a single computation.
StatusOr<std::vector<const HloInstruction*>> CreateMemoryMinimizingSequence(
    const HloComputation& computation,
    const LogicalBuffer::SizeFunction& size_function);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SCHEDULING_H_
