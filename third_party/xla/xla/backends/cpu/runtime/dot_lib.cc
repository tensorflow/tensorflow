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

#include "xla/backends/cpu/runtime/dot_lib.h"

#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla::cpu {

absl::InlinedVector<BufferUse, 4> DotBufferUses(const DotSlices& slices) {
  return {BufferUse::Read(slices.lhs_buffer),
          BufferUse::Read(slices.rhs_buffer),
          BufferUse::Write(slices.out_buffer)};
}

absl::StatusOr<DotShape> GetDotShape(const DotDimensionNumbers& dot_dimensions,
                                     const Shape& lhs_shape,
                                     const Shape& rhs_shape,
                                     const Shape& out_shape) {
  // All shapes must be in dim0-major layout.
  if (!LayoutUtil::IsMonotonicWithDim0Major(lhs_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(rhs_shape.layout()) ||
      !LayoutUtil::IsMonotonicWithDim0Major(out_shape.layout())) {
    return InvalidArgument(
        "DotThunk requires all operands and outputs to be in "
        "dim0-major layout: lhs_shape=[%s], rhs_shape=[%s], out_shape=[%s]",
        lhs_shape.ToString(true), rhs_shape.ToString(true),
        out_shape.ToString(true));
  }

  // Batch dimensions must be contiguous and start at 0.
  std::vector<int64_t> batch_dims(dot_dimensions.lhs_batch_dimensions().size());
  absl::c_iota(batch_dims, 0);

  if (!absl::c_equal(dot_dimensions.lhs_batch_dimensions(), batch_dims) ||
      !absl::c_equal(dot_dimensions.rhs_batch_dimensions(), batch_dims)) {
    return InvalidArgument(
        "Batch dimensions must be contiguous and start at 0: "
        "lhs_batch_dims=[%s], rhs_batch_dims=[%s]",
        absl::StrJoin(dot_dimensions.lhs_batch_dimensions(), ","),
        absl::StrJoin(dot_dimensions.rhs_batch_dimensions(), ","));
  }

  int64_t num_batch_dims = batch_dims.size();
  int64_t batch_size =
      std::accumulate(out_shape.dimensions().begin(),
                      out_shape.dimensions().begin() + num_batch_dims, 1LL,
                      std::multiplies<int64_t>());

  Shape lhs_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, lhs_shape);
  Shape rhs_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, rhs_shape);
  Shape out_matmul_shape = ShapeUtil::DeleteDimensions(batch_dims, out_shape);

  // Check that matmul shapes are rank 2 or less and can be represented as
  // Eigen 2D contraction.
  if (lhs_matmul_shape.rank() > 2 || rhs_matmul_shape.rank() > 2 ||
      out_matmul_shape.rank() > 2) {
    return InvalidArgument(
        "MatMul shape must be rank 2 or less: lhs=%s, rhs=%s, out=%s",
        lhs_matmul_shape.ToString(true), rhs_matmul_shape.ToString(true),
        out_matmul_shape.ToString(true));
  }

  return DotShape{
      batch_size,
      std::move(lhs_matmul_shape),
      std::move(rhs_matmul_shape),
      std::move(out_matmul_shape),
  };
}

absl::StatusOr<DotCanonicalDims> GetDotCanonicalDims(
    const DotDimensionNumbers& dot_dimensions, const DotShape& dot_shape) {
  // Copy from the original dot dimension numbers.
  absl::InlinedVector<int64_t, 2> lhs_contracting_dims;
  absl::InlinedVector<int64_t, 2> rhs_contracting_dims;

  lhs_contracting_dims.assign(
      dot_dimensions.lhs_contracting_dimensions().begin(),
      dot_dimensions.lhs_contracting_dimensions().end());
  rhs_contracting_dims.assign(
      dot_dimensions.rhs_contracting_dimensions().begin(),
      dot_dimensions.rhs_contracting_dimensions().end());

  // Adjust contracting dimensions for leading batch dimensions.
  for (int64_t& dim : lhs_contracting_dims)
    dim -= dot_dimensions.lhs_batch_dimensions_size();
  for (int64_t& dim : rhs_contracting_dims)
    dim -= dot_dimensions.rhs_batch_dimensions_size();

  // Non-contracting dots should never make it here.
  TF_RET_CHECK(lhs_contracting_dims.size() == 1);
  TF_RET_CHECK(rhs_contracting_dims.size() == 1);
  TF_RET_CHECK(lhs_contracting_dims[0] < 2);
  TF_RET_CHECK(rhs_contracting_dims[0] < 2);

  auto is_column_major = [](const Shape& shape) {
    return shape.rank() > 1 && LayoutUtil::Minor(shape.layout(), 0) == 0;
  };

  return DotCanonicalDims{
      /*m=*/dot_shape.lhs_matmul_shape.rank() <= 1
          ? int64_t{1}
          : dot_shape.lhs_matmul_shape.dimensions(1 - lhs_contracting_dims[0]),
      /*k=*/dot_shape.lhs_matmul_shape.dimensions(lhs_contracting_dims[0]),
      /*n=*/dot_shape.rhs_matmul_shape.rank() <= 1
          ? int64_t{1}
          : dot_shape.rhs_matmul_shape.dimensions(1 - rhs_contracting_dims[0]),
      /*lhs_column_major=*/is_column_major(dot_shape.lhs_matmul_shape),
      /*lhs_canonical=*/dot_shape.lhs_matmul_shape.rank() <= 1 ||
          lhs_contracting_dims[0] == 1,
      /*rhs_column_major=*/is_column_major(dot_shape.rhs_matmul_shape),
      /*rhs_canonical=*/rhs_contracting_dims[0] == 0};
}

}  // namespace xla::cpu
