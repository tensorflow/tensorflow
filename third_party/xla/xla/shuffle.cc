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

#include "xla/shuffle.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace shuffle {

ShuffleMode Permute(const LiteralSlice& indices) {
  ShuffleMode mode;
  *mode.mutable_permute()->mutable_indices() = indices.ToProto();
  return mode;
}

ShuffleMode Rotate(absl::Span<const int64_t> shifts) {
  ShuffleMode mode;
  mode.mutable_rotate()->mutable_shifts()->Assign(shifts.begin(), shifts.end());
  return mode;
}

ShuffleMode MultiRotate(const LiteralSlice& multi_shifts) {
  ShuffleMode mode;
  *mode.mutable_multi_rotate()->mutable_multi_shifts() = multi_shifts.ToProto();
  return mode;
}

absl::StatusOr<Literal> GetPermuteIndices(const ShuffleMode& mode) {
  CHECK(mode.has_permute()) << "the mode of the shuffle is not permute";
  return Literal::CreateFromProto(mode.permute().indices());
}

absl::StatusOr<Shape> GetPermuteIndicesShape(const ShuffleMode& mode) {
  CHECK(mode.has_permute()) << "the mode of the shuffle is not permute";
  return Shape::FromProto(mode.permute().indices().shape());
}

absl::StatusOr<Literal> GetMultiShifts(const ShuffleMode& mode) {
  CHECK(mode.has_multi_rotate())
      << "the mode of the shuffle is not multi-rotate";
  return Literal::CreateFromProto(mode.multi_rotate().multi_shifts());
}

absl::StatusOr<Shape> GetMultiShiftsShape(const ShuffleMode& mode) {
  CHECK(mode.has_multi_rotate())
      << "the mode of the shuffle is not multi-rotate";
  return Shape::FromProto(mode.multi_rotate().multi_shifts().shape());
}

absl::StatusOr<Literal> GetMultiRotatePermuteIndices(const ShuffleMode& mode,
                                                     int64_t dimension,
                                                     int64_t dimension_size) {
  ABSL_ASSIGN_OR_RETURN(const Literal multi_shifts, GetMultiShifts(mode));
  // Expand the rotated dimension to the size of the dimension in the operand.
  std::vector<int64_t> indices_dimensions(
      multi_shifts.shape().dimensions().begin(),
      multi_shifts.shape().dimensions().end());
  indices_dimensions[dimension] = dimension_size;

  Literal indices =
      Literal::CreateFromShape(ShapeUtil::MakeShape(S32, indices_dimensions));
  // For each element in the result, find the corresponding shift and add it to
  // the index.
  ABSL_RETURN_IF_ERROR(
      indices.Populate<int32_t>([&](absl::Span<const int64_t> index) {
        DimensionVector shift_index(index.begin(), index.end());
        shift_index[dimension] = 0;
        const int64_t shift = *multi_shifts.GetIntegralAsS64(shift_index);
        return NormalizeShift(index[dimension] + shift, dimension_size);
      }));
  return indices;
}

int64_t NormalizeShift(int64_t shift, int64_t dim_size) {
  CHECK_GE(dim_size, 0) << "a dimension cannot have a negative size";
  return dim_size == 0 ? 0 : ((shift % dim_size) + dim_size) % dim_size;
}

}  // namespace shuffle
}  // namespace xla
