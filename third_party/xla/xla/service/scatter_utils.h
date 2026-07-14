/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SCATTER_UTILS_H_
#define XLA_SERVICE_SCATTER_UTILS_H_

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

// Transposes the given scatter_indices such that the index_vector_dim becomes
// the most-minor dimension.
absl::StatusOr<HloInstruction*> TransposeIndexVectorDimToLast(
    HloInstruction* scatter_indices, int64_t index_vector_dim);

// Permutes the `updates` tensor such that all the scatter dims appear in the
// major dimensions and all the window dimensions appear in the minor
// dimensions.
absl::StatusOr<HloInstruction*> PermuteScatterAndWindowDims(
    HloInstruction* updates, absl::Span<const int64_t> update_window_dims);

// Expands or contracts the scatter indices in the updates tensor.
absl::StatusOr<HloInstruction*> AdjustScatterDims(
    const Shape& scatter_indices_shape, HloInstruction* updates,
    int64_t index_vector_dim);

// Canonicalizes the scatter_indices tensor in order to keep them uniform while
// performing the scatter operation.
absl::StatusOr<HloInstruction*> CanonicalizeScatterIndices(
    HloInstruction* scatter_indices, int64_t index_vector_dim);

absl::StatusOr<HloComputation*> CallAndGetOutput(HloComputation* original,
                                                 int output_index);
absl::StatusOr<HloComputation*> CallComputationAndGetIthOutputWithBinaryParams(
    HloComputation* original, int output_index);

int64_t ScatterIndicesCount(const HloScatterInstruction* scatter);

// Checks if the combiner is associative.
bool IsScatterCombinerAssociative(const HloComputation* combiner);

// Checks if the scatter operation is deterministic.
bool IsScatterDeterministic(const HloScatterInstruction* scatter);

}  // namespace xla

#endif  // XLA_SERVICE_SCATTER_UTILS_H_
