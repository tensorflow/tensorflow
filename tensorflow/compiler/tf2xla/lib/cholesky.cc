/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/cholesky.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/batch_dot.h"
#include "tensorflow/compiler/tf2xla/lib/triangular_solve.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

// def cholesky_unblocked(a):
//   assert len(a.shape) == 2 and a.shape[-2] == a.shape[-1]
//   n = a.shape[-2]
//   l = np.zeros_like(a)
//   for j in xrange(n):
//     r = l[..., j, :j]
//     l[..., j, j] = np.sqrt(a[..., j, j] - np.dot(r, r))
//     l[..., j+1:, j] = (a[..., j+1:, j] - np.dot(l[..., j+1:, :j],
//         np.transpose(r))) / l[..., j, j]
//   return l
xla::StatusOr<xla::ComputationDataHandle> CholeskyUnblocked(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& a) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Shape> shape, builder->GetShape(a));
  xla::ComputationDataHandle l = Zeros(builder, *shape);
  const int64 n = xla::ShapeUtil::GetDimension(*shape, -2);
  for (int j = 0; j < n; ++j) {
    // Picture of block structure:
    // ...   \
    //        \
    // -- r -- d
    //         |\
    //    B    c \
    //         |  \
    //         |  ...
    //
    //         ^
    //      column j
    TF_ASSIGN_OR_RETURN(auto d,
                        SliceInMinorDims(builder, a, {j, j}, {j + 1, j + 1}));
    TF_ASSIGN_OR_RETURN(auto c,
                        SliceInMinorDims(builder, a, {j + 1, j}, {n, j + 1}));
    xla::ComputationDataHandle new_d_squared = d;
    xla::ComputationDataHandle br;
    if (j > 0) {
      TF_ASSIGN_OR_RETURN(auto r,
                          SliceInMinorDims(builder, l, {j, 0}, {j + 1, j}));
      TF_ASSIGN_OR_RETURN(auto b,
                          SliceInMinorDims(builder, l, {j + 1, 0}, {n, j}));
      TF_ASSIGN_OR_RETURN(auto r_squared,
                          BatchDot(builder, r, r, /*transpose_x=*/false,
                                   /*transpose_y=*/true, /*conjugate_x=*/false,
                                   /*conjugate_y=*/false));
      new_d_squared = builder->Sub(new_d_squared, r_squared);

      TF_ASSIGN_OR_RETURN(br, BatchDot(builder, b, r, /*transpose_x=*/false,
                                       /*transpose_y=*/true,
                                       /*conjugate_x=*/false,
                                       /*conjugate_y=*/false));
    }
    auto new_d_inv = builder->Pow(
        new_d_squared, FloatLiteral(builder, shape->element_type(), -0.5));
    auto new_d = builder->Mul(new_d_inv, new_d_squared);
    TF_ASSIGN_OR_RETURN(l, UpdateSliceInMinorDims(builder, l, new_d, {j, j}));

    if (j > 0) {
      c = builder->Sub(c, br);
    }
    auto new_c = builder->Mul(c, new_d_inv);
    TF_ASSIGN_OR_RETURN(l,
                        UpdateSliceInMinorDims(builder, l, new_c, {j + 1, j}));
  }
  return l;
}

}  // namespace

xla::StatusOr<xla::ComputationDataHandle> Cholesky(
    xla::ComputationBuilder* builder, xla::ComputationDataHandle a,
    int64 block_size) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Shape> a_shape,
                      builder->GetShape(a));
  const int ndims = xla::ShapeUtil::Rank(*a_shape);
  if (ndims < 2) {
    return errors::InvalidArgument(
        "Arguments to Cholesky must have rank >= 2: ", ndims);
  }

  const int64 n = xla::ShapeUtil::GetDimension(*a_shape, -1);
  if (n != xla::ShapeUtil::GetDimension(*a_shape, -2)) {
    return errors::InvalidArgument(
        "Arguments to Cholesky must be square matrices: ",
        xla::ShapeUtil::HumanString(*a_shape));
  }

  if (block_size < 1) {
    return errors::InvalidArgument(
        "block_size argument to Cholesky must be >= 1; got ", block_size);
  }

  // Blocked left-looking Cholesky factorization.
  // Algorithm 1 from
  // Haidar, Azzam, et al. "High-performance Cholesky factorization for GPU-only
  // execution." Proceedings of General Purpose GPUs. ACM, 2017.
  xla::ComputationDataHandle l = Zeros(builder, *a_shape);
  for (int64 i = 0; i < n; i += block_size) {
    int64 k = std::min(block_size, n - i);
    if (i > 0) {
      // TODO(phawkins): consider implementing SYRK for the diagonal part of
      // the panel.
      // a[i:, i:i+k] -= np.dot(l[i:, :i], np.transpose(l[i:i+k, :i]))
      TF_ASSIGN_OR_RETURN(auto lhs,
                          SliceInMinorDims(builder, l, {i, 0}, {n, i}));
      TF_ASSIGN_OR_RETURN(auto rhs,
                          SliceInMinorDims(builder, l, {i, 0}, {i + k, i}));
      TF_ASSIGN_OR_RETURN(auto delta,
                          BatchDot(builder, lhs, rhs, /*transpose_x=*/false,
                                   /*transpose_y=*/true, /*conjugate_x=*/false,
                                   /*conjugate_y=*/false));
      TF_ASSIGN_OR_RETURN(auto before,
                          SliceInMinorDims(builder, a, {i, i}, {n, i + k}));
      TF_ASSIGN_OR_RETURN(
          a, UpdateSliceInMinorDims(builder, a, builder->Sub(before, delta),
                                    {i, i}));
    }

    // l[i:i+k, i:i+k] = cholesky_unblocked(a[i:i+k, i:i+k])
    TF_ASSIGN_OR_RETURN(auto x,
                        SliceInMinorDims(builder, a, {i, i}, {i + k, i + k}));
    TF_ASSIGN_OR_RETURN(auto factorized, CholeskyUnblocked(builder, x));
    TF_ASSIGN_OR_RETURN(l,
                        UpdateSliceInMinorDims(builder, l, factorized, {i, i}));

    if (i + k < n) {
      // l[i+k:, i:i+k] = trsm_right_transpose(l[i:i+k, i:i+k], a[i+k:, i:i+k])
      TF_ASSIGN_OR_RETURN(auto panel,
                          SliceInMinorDims(builder, a, {i + k, i}, {n, i + k}));
      TF_ASSIGN_OR_RETURN(auto update,
                          TriangularSolve(builder, factorized, panel,
                                          /*left_side=*/false,
                                          /*lower=*/true,
                                          /*transpose_a=*/true,
                                          /*conjugate_a=*/false,
                                          /*block_size=*/8));
      TF_ASSIGN_OR_RETURN(
          l, UpdateSliceInMinorDims(builder, l, update, {i + k, i}));
    }
  }
  return l;
}

}  // namespace tensorflow
