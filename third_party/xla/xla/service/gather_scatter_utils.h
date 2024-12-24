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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"

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

// Expands an index vector from the start_indices tensor into a vector that can
// be used to dynamic-slice out of the gather/scatter operand.
absl::StatusOr<HloInstruction*> ExpandIndexVectorIntoOperandSpace(
    const Shape& start_indices_shape, int64_t operand_rank,
    int64_t index_vector_dim, absl::Span<const int64_t> start_index_map,
    absl::Span<const int64_t> start_indices_batching_dims,
    absl::Span<const int64_t> operand_batching_dims,
    HloInstruction* index_vector, HloInstruction* induction_var);

// Returns true if the given dimension is a collapsed or batching dimension.
bool IsCollapsedOrBatchingDim(absl::Span<const int64_t> collapsed_dims,
                              absl::Span<const int64_t> batching_dims,
                              int64_t dim);

// Returns a map from start_indices explicit batching dims to their
// corresponding output dims.
absl::flat_hash_map<int64_t, int64_t>
GetStartIndicesDimToOutputDimForExplicitBatchingDims(
    absl::Span<const int64_t> start_indices_batching_dims,
    int64_t index_vector_dim, absl::Span<const int64_t> offset_dims,
    int64_t start_indices_rank, int64_t output_rank);

}  // namespace xla

#endif  // XLA_SERVICE_GATHER_SCATTER_UTILS_H_
