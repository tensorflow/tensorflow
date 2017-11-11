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

#include "tensorflow/compiler/tf2xla/lib/batch_dot.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// The current implementation simply unrolls the computation along the batch
// dimension.
// TODO(andydavis): add batching support to XLA's Dot operator.
xla::StatusOr<xla::ComputationDataHandle> BatchDot(
    xla::ComputationBuilder* builder, xla::ComputationDataHandle x,
    xla::ComputationDataHandle y, bool transpose_x, bool transpose_y) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Shape> x_shape,
                      builder->GetShape(x));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::Shape> y_shape,
                      builder->GetShape(y));

  // Check that both tensors have the same number of dimensions. There must be
  // at least two (the batch dimensions can be empty).
  if (xla::ShapeUtil::Rank(*x_shape) != xla::ShapeUtil::Rank(*y_shape)) {
    return errors::InvalidArgument(
        "Arguments to BatchedDot have different ranks: ",
        xla::ShapeUtil::HumanString(*x_shape), " vs. ",
        xla::ShapeUtil::HumanString(*y_shape));
  }
  const int ndims = xla::ShapeUtil::Rank(*x_shape);
  if (ndims < 2) {
    return errors::InvalidArgument(
        "Arguments to BatchedDot must have rank >= 2: ", ndims);
  }

  // The batch dimensions must be equal and the matrix dimensions must be
  // valid.
  std::vector<int64> dimensions;
  int64 batch_count = 1;
  for (int i = 0; i < ndims - 2; ++i) {
    int64 x_size = x_shape->dimensions(i);
    int64 y_size = y_shape->dimensions(i);
    if (x_size != y_size) {
      return errors::InvalidArgument(
          "Dimension ", i, " of inputs to BatchedDot must be equal: ",
          xla::ShapeUtil::HumanString(*x_shape), " vs ",
          xla::ShapeUtil::HumanString(*y_shape));
    }
    dimensions.push_back(x_size);
    batch_count *= x_size;
  }

  int x_inner_dim = transpose_x ? (ndims - 2) : (ndims - 1);
  int y_inner_dim = transpose_y ? (ndims - 1) : (ndims - 2);
  int64 x_inner_dim_size = x_shape->dimensions(x_inner_dim);
  int64 y_inner_dim_size = y_shape->dimensions(y_inner_dim);
  if (x_inner_dim_size != y_inner_dim_size) {
    return errors::InvalidArgument(
        "Dimensions ", x_inner_dim, " and ", y_inner_dim,
        " of arguments to BatchedDot must be equal: ",
        xla::ShapeUtil::HumanString(*x_shape), " transpose: ", transpose_x,
        " vs. ", xla::ShapeUtil::HumanString(*y_shape),
        " transpose: ", transpose_y);
  }

  // If there are no batch dimensions, use a regular Dot. This case exists
  // to improve the readability of the emitted graphs.
  if (dimensions.empty()) {
    auto lhs = transpose_x ? builder->Transpose(x, {1, 0}) : x;
    auto rhs = transpose_y ? builder->Transpose(y, {1, 0}) : y;
    return builder->Dot(lhs, rhs);
  }

  int x_outer_dim = transpose_x ? (ndims - 1) : (ndims - 2);
  int y_outer_dim = transpose_y ? (ndims - 2) : (ndims - 1);
  dimensions.push_back(x_shape->dimensions(x_outer_dim));
  dimensions.push_back(y_shape->dimensions(y_outer_dim));

  if (x_shape->element_type() == xla::C64 && transpose_x) {
    x = builder->Conj(x);
  }
  if (y_shape->element_type() == xla::C64 && transpose_y) {
    y = builder->Conj(y);
  }

  // Reshape input tensors into 3D tensors by flattening the batch
  // dimensions. This makes it easier to unroll the batch dimension.
  auto x_flat =
      builder->Reshape(x, {batch_count, x_shape->dimensions(ndims - 2),
                           x_shape->dimensions(ndims - 1)});
  auto y_flat =
      builder->Reshape(y, {batch_count, y_shape->dimensions(ndims - 2),
                           y_shape->dimensions(ndims - 1)});

  // Slice batches into individual matrices and multiply them.
  std::vector<xla::ComputationDataHandle> out_slices;
  for (int64 i = 0; i < batch_count; ++i) {
    // Slice off individual matrices and reshape to 2D tensors.
    auto x_slice = builder->Slice(
        x_flat, {i, 0, 0},
        {i + 1, x_shape->dimensions(ndims - 2), x_shape->dimensions(ndims - 1)},
        {1, 1, 1});
    x_slice = builder->Reshape(x_slice, {x_shape->dimensions(ndims - 2),
                                         x_shape->dimensions(ndims - 1)});
    auto y_slice = builder->Slice(
        y_flat, {i, 0, 0},
        {i + 1, y_shape->dimensions(ndims - 2), y_shape->dimensions(ndims - 1)},
        {1, 1, 1});
    y_slice = builder->Reshape(y_slice, {y_shape->dimensions(ndims - 2),
                                         y_shape->dimensions(ndims - 1)});

    // Transpose if needed.
    auto lhs = transpose_x ? builder->Transpose(x_slice, {1, 0}) : x_slice;
    auto rhs = transpose_y ? builder->Transpose(y_slice, {1, 0}) : y_slice;

    // Multiply matrices and add an outer singleton dimension to the output
    // so we can concatenate along the flattened batch dimension later.
    auto out = builder->Dot(lhs, rhs);
    out = builder->Reshape(out,
                           {1, dimensions[ndims - 2], dimensions[ndims - 1]});
    out_slices.push_back(out);
  }

  // Concatenate output slices and reshape to original number of dimensions.
  xla::ComputationDataHandle data;
  if (out_slices.empty()) {
    // It is illegal to pass an empty list to ConcatInDim.
    // The batch count is empty, so both inputs must have zero elements.
    // Arbitrarily use the left input as the argument to Reshape().
    data = x;
  } else {
    data = builder->ConcatInDim(out_slices, 0);
  }
  return builder->Reshape(data, dimensions);
}

}  // namespace tensorflow
