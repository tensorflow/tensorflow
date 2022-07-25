/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GATHER_SCATTER_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GATHER_SCATTER_UTILS_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

StatusOr<HloInstruction*> MaybeTranspose(HloInstruction* operand,
                                         absl::Span<const int64_t> permutation);

StatusOr<std::vector<HloInstruction*>> MaybeTranspose(
    absl::Span<HloInstruction* const> operands,
    const std::vector<int64_t>& operand_permutation);

// Moves the given dimension to the last dimension.
// Example: MoveDimensionToEnd(tensor<1x2x3xi1>, 0): tensor<2x3x1xi1>.
StatusOr<HloInstruction*> MoveDimensionToEnd(HloInstruction* operand,
                                             size_t dimension, size_t rank);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GATHER_SCATTER_UTILS_H_
