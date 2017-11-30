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

#include "tensorflow/compiler/tf2xla/lib/triangular_solve.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/tf2xla/lib/batch_dot.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

xla::StatusOr<xla::ComputationDataHandle> TriangularSolve(
    xla::ComputationBuilder* builder, const xla::ComputationDataHandle& a,
    xla::ComputationDataHandle b, int64 block_size) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Shape> a_shape,
                      builder->GetShape(a));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Shape> b_shape,
                      builder->GetShape(b));
  if (xla::ShapeUtil::Rank(*a_shape) != xla::ShapeUtil::Rank(*b_shape)) {
    return errors::InvalidArgument(
        "Arguments to TriangularSolve have different ranks: ",
        xla::ShapeUtil::HumanString(*a_shape), " vs. ",
        xla::ShapeUtil::HumanString(*b_shape));
  }
  const int ndims = xla::ShapeUtil::Rank(*a_shape);
  if (ndims < 2) {
    return errors::InvalidArgument(
        "Arguments to TriangularSolve must have rank >= 2: ", ndims);
  }
  // The batch dimensions must be equal.
  std::vector<int64> batch_dimensions;
  for (int i = 0; i < ndims - 2; ++i) {
    int64 a_size = a_shape->dimensions(i);
    int64 b_size = b_shape->dimensions(i);
    if (a_size != b_size) {
      return errors::InvalidArgument(
          "Batch dimensions of arguments to TriangularSolve must be equal: ",
          xla::ShapeUtil::HumanString(*a_shape), " vs ",
          xla::ShapeUtil::HumanString(*b_shape));
    }
    batch_dimensions.push_back(a_size);
  }

  const int64 n = xla::ShapeUtil::GetDimension(*a_shape, -1);
  const int64 m = xla::ShapeUtil::GetDimension(*b_shape, -2);
  if (n != xla::ShapeUtil::GetDimension(*a_shape, -2)) {
    return errors::InvalidArgument(
        "The 'a' arguments to TriangularSolve must be square matrices: ",
        xla::ShapeUtil::HumanString(*a_shape));
  }
  if (n != xla::ShapeUtil::GetDimension(*b_shape, -1)) {
    return errors::InvalidArgument(
        "Arguments to TriangularSolve have incompatible matrix shapes: ",
        xla::ShapeUtil::HumanString(*a_shape), " vs ",
        xla::ShapeUtil::HumanString(*b_shape));
  }

  if (block_size < 1) {
    return errors::InvalidArgument(
        "block_size argument to TriangularSolve must be >= 1; got ",
        block_size);
  }

  // Returns [b1, b2, ... , bn, indices[0], indices[1]].
  auto prepend_batch_dims = [&](std::array<int64, 2> indices) {
    std::vector<int64> output(ndims);
    std::copy(batch_dimensions.begin(), batch_dimensions.end(), output.begin());
    std::copy(indices.begin(), indices.end(),
              output.begin() + batch_dimensions.size());
    return output;
  };

  std::map<int, xla::Computation> base_computations;
  auto get_base_triangular_solve =
      [&](int k) -> xla::StatusOr<xla::Computation*> {
    xla::Computation& computation = base_computations[k];
    if (computation.IsNull()) {
      std::unique_ptr<xla::ComputationBuilder> sub = builder->CreateSubBuilder(
          tensorflow::strings::StrCat("trsm_base_", k));

      auto a_param =
          sub->Parameter(0,
                         xla::ShapeUtil::MakeShape(b_shape->element_type(),
                                                   prepend_batch_dims({k, k})),
                         "a");

      auto b_param =
          sub->Parameter(1,
                         xla::ShapeUtil::MakeShape(b_shape->element_type(),
                                                   prepend_batch_dims({m, k})),
                         "b");

      // TODO(phawkins): it might make sense to use a while loop here, rather
      // than unrolling.
      // TODO(phawkins): the left-looking variant of the algorithm might be more
      // efficient at block size 1.
      TF_RETURN_IF_ERROR(TriangularSolve(sub.get(), a_param, b_param,
                                         /*block_size=*/1)
                             .status());

      TF_ASSIGN_OR_RETURN(computation, sub->Build());
    }
    return &computation;
  };

  xla::ComputationDataHandle output = Zeros(builder, *b_shape);

  // Right-looking blocked triangular solve.
  // For an explanation of the algorithm, see the TRSM discussion in:
  // Goto, Kazushige, and Robert Van De Geijn. "High-performance implementation
  // of the level-3 BLAS." ACM Transactions on Mathematical Software (TOMS) 35.1
  // (2008): 4.
  for (int64 i = 0; i < n; i += block_size) {
    int64 k = std::min(block_size, n - i);

    // if k > 1:
    //   output[..., :, i:i+k] = triangular_solve(
    //       a[..., i:i+k, ..., i:i+k], b[..., :, i:i+k], side='Right',
    //       kind='Lower', transpose=True, block_size=1)
    // else:
    //   output[..., :, i] = b[..., :, i] / a[..., i, i]
    TF_ASSIGN_OR_RETURN(auto a_slice,
                        SliceInMinorDims(builder, a, {i, i}, {i + k, i + k}));
    TF_ASSIGN_OR_RETURN(auto b_slice,
                        SliceInMinorDims(builder, b, {0, i}, {m, i + k}));
    xla::ComputationDataHandle update;
    if (k > 1) {
      TF_ASSIGN_OR_RETURN(xla::Computation * solve,
                          get_base_triangular_solve(k));
      update = builder->Call(*solve, {a_slice, b_slice});
    } else {
      update = builder->Div(b_slice, a_slice);
    }

    TF_ASSIGN_OR_RETURN(
        output, UpdateSliceInMinorDims(builder, output, update, {0, i}));
    // b[..., :, i+k:] -= np.dot(output[..., :, i:i+k],
    //                           np.transpose(..., a[i+k:, i:i+k]))
    if (i + k < n) {
      TF_ASSIGN_OR_RETURN(auto a_slice_2,
                          SliceInMinorDims(builder, a, {i + k, i}, {n, i + k}));
      TF_ASSIGN_OR_RETURN(auto b_update, BatchDot(builder, update, a_slice_2,
                                                  /*transpose_x=*/false,
                                                  /*transpose_y=*/true));

      TF_ASSIGN_OR_RETURN(auto b_slice_2,
                          SliceInMinorDims(builder, b, {0, i + k}, {m, n}));
      b_update = builder->Sub(b_slice_2, b_update);
      TF_ASSIGN_OR_RETURN(
          b, UpdateSliceInMinorDims(builder, b, b_update, {0, i + k}));
    }
  }
  return output;
}

}  // namespace tensorflow
