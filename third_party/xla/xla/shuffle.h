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

#ifndef XLA_SHUFFLE_H_
#define XLA_SHUFFLE_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace shuffle {

// The mode of a shuffle is the pattern in which it moves the elements of its
// shuffled dimensions, and the attributes that parameterize the pattern.

// Returns the mode of a shuffle that permutes the shuffled dimensions
// according to `indices`. See `ShuffleMode::Permute` in `xla_data.proto` for
// the layout that `indices` is expected to have.
ShuffleMode Permute(const LiteralSlice& indices);

// Returns the mode of a shuffle that rotates each shuffled dimension to the
// left by the corresponding entry of `shifts`.
ShuffleMode Rotate(absl::Span<const int64_t> shifts);

// Returns the mode of a shuffle that rotates each slice along the shuffled
// dimension to the left by its own shift. See `ShuffleMode::MultiRotate` in
// `xla_data.proto` for the layout that `multi_shifts` is expected to have.
ShuffleMode MultiRotate(const LiteralSlice& multi_shifts);

// Returns the indices of permute `mode`. This decodes and copies the indices;
// prefer `GetPermuteIndicesShape` when only their shape is needed.
absl::StatusOr<Literal> GetPermuteIndices(const ShuffleMode& mode);

// Returns the shape of the indices of permute `mode`.
absl::StatusOr<Shape> GetPermuteIndicesShape(const ShuffleMode& mode);

// Returns the shifts of multi-rotate `mode`. This decodes and copies the
// shifts; prefer `GetMultiShiftsShape` when only their shape is needed.
absl::StatusOr<Literal> GetMultiShifts(const ShuffleMode& mode);

// Returns the shape of the shifts of multi-rotate `mode`.
absl::StatusOr<Shape> GetMultiShiftsShape(const ShuffleMode& mode);

// Returns the equivalent permute indices for a multi-rotate for gather
// lowering. The indices have the shape of `multi_shifts`, except that the
// rotated dimension has size `dimension_size`. Assumes the Shuffle is verified.
absl::StatusOr<Literal> GetMultiRotatePermuteIndices(const ShuffleMode& mode,
                                                     int64_t dimension,
                                                     int64_t dimension_size);

// Returns the normalized shift in `[0, dim_size)`. Returns 0 if `dim_size` is
// 0, because every rotation of an empty dimension is a no-op.
int64_t NormalizeShift(int64_t shift, int64_t dim_size);

}  // namespace shuffle
}  // namespace xla

#endif  // XLA_SHUFFLE_H_
