/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GATHER_SCATTER_UTILS_H_
#define XLA_SERVICE_GATHER_SCATTER_UTILS_H_

#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Transforms the given index tensor to make it two-dimensional, with the index
// vector dimension being dimension 1.
// Example:
//   input: indices = tensor<4x2x3xi32>, index_vector_dim = 1
//   output: tensor<12x2xi32>
absl::StatusOr<HloInstruction*> TransformStartIndices(HloInstruction* indices,
                                                      int64_t index_vector_dim);

// Given a map from index vector positions to dimension numbers, returns a pair
// of permutations that when applied to the operand, let you replace the map
// with the identity permutation.
// In gather, the map is called `start_index_map`. In scatter, it's
// `scatter_dims_to_operand_dims`.
std::pair<std::vector<int64_t>, std::vector<int64_t>>
MakeOperandStartIndexPermutations(absl::Span<const int64_t>, int operand_rank);

absl::StatusOr<HloInstruction*> MaybeTranspose(
    HloInstruction* operand, absl::Span<const int64_t> permutation);

absl::StatusOr<std::vector<HloInstruction*>> MaybeTranspose(
    absl::Span<HloInstruction* const> operands,
    const std::vector<int64_t>& operand_permutation);

// Moves the given dimension to the last dimension.
// Example: MoveDimensionToEnd(tensor<1x2x3xi1>, 0): tensor<2x3x1xi1>.
absl::StatusOr<HloInstruction*> MoveDimensionToEnd(HloInstruction* operand,
                                                   size_t dimension,
                                                   size_t rank);

}  // namespace xla

#endif  // XLA_SERVICE_GATHER_SCATTER_UTILS_H_
