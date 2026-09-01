/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_BITCAST_UTILS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_BITCAST_UTILS_H_

#include <cstdint>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"

namespace xla::gpu {

// Parameters to rewrite a bitcast(broadcast/transpose) as
// broadcast/transpose(bitcast) and vice versa.
struct BitcastParams {
  Shape new_shape;                      // The bitcast output shape.
  llvm::SmallVector<int64_t> new_dims;  // The dims of the broadcast/transpose.
};

// Returns parameters to rewrite a broadcast + bitcast as bitcast + broadcast.
absl::StatusOr<BitcastParams> CalculateBitcastOfBroadcast(
    const HloBroadcastInstruction* broadcast, const Shape& result_shape);

// Returns parameters to rewrite a bitcast + broadcast as broadcast + bitcast.
absl::StatusOr<BitcastParams> CalculateBroadcastOfBitcast(
    const HloBroadcastInstruction* broadcast, const Shape& operand_shape);

// Returns parameters to rewrite a transpose + bitcast as bitcast + transpose.
absl::StatusOr<BitcastParams> CalculateBitcastOfTranspose(
    const HloTransposeInstruction* transpose, const Shape& result_shape);

// Returns parameters to rewrite a bitcast + transpose as transpose + bitcast.
absl::StatusOr<BitcastParams> CalculateTransposeOfBitcast(
    const HloTransposeInstruction* transpose, const Shape& operand_shape);

// Copies the element type and size from `source` to `destination`.
void CopyElementType(const Shape& source, Shape* destination);

namespace detail {

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product (see `CommonFactors` from `util.h`) grouping
// ranges having product of 1 with neighbors.
//
// For example, if a=[2, 5, 1, 3] and b=[1, 10, 3, 1], the result will be
// {{0, 0}, {2, 2}, {4, 4}}, grouping [2,5] with [1,10] and [1,3] with [3,1].
absl::InlinedVector<std::pair<int64_t, int64_t>, 8>
CommonFactorsMergingTrivialRanges(absl::Span<const int64_t> a,
                                  absl::Span<const int64_t> b);

}  // namespace detail

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_BITCAST_UTILS_H_
