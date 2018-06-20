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

xla::StatusOr<xla::XlaOp> BatchDot(xla::XlaBuilder* builder, xla::XlaOp x,
                                   xla::XlaOp y, bool transpose_x,
                                   bool transpose_y, bool conjugate_x,
                                   bool conjugate_y) {
  TF_ASSIGN_OR_RETURN(xla::Shape x_shape, builder->GetShape(x));
  TF_ASSIGN_OR_RETURN(xla::Shape y_shape, builder->GetShape(y));

  // Check that both tensors have the same number of dimensions. There must be
  // at least two (the batch dimensions can be empty).
  if (xla::ShapeUtil::Rank(x_shape) != xla::ShapeUtil::Rank(y_shape)) {
    return errors::InvalidArgument(
        "Arguments to BatchedDot have different ranks: ",
        xla::ShapeUtil::HumanString(x_shape), " vs. ",
        xla::ShapeUtil::HumanString(y_shape));
  }
  const int ndims = xla::ShapeUtil::Rank(x_shape);
  if (ndims < 2) {
    return errors::InvalidArgument(
        "Arguments to BatchedDot must have rank >= 2: ", ndims);
  }

  // The batch dimensions must be equal and the matrix dimensions must be
  // valid.
  std::vector<int64> batch_dimension_numbers;
  for (int i = 0; i < ndims - 2; ++i) {
    if (x_shape.dimensions(i) != y_shape.dimensions(i)) {
      return errors::InvalidArgument(
          "Dimension ", i, " of inputs to BatchedDot must be equal: ",
          xla::ShapeUtil::HumanString(x_shape), " vs ",
          xla::ShapeUtil::HumanString(y_shape));
    }
    batch_dimension_numbers.push_back(i);
  }

  int x_inner_dim = transpose_x ? (ndims - 2) : (ndims - 1);
  int y_inner_dim = transpose_y ? (ndims - 1) : (ndims - 2);
  if (x_shape.dimensions(x_inner_dim) != y_shape.dimensions(y_inner_dim)) {
    return errors::InvalidArgument(
        "Dimensions ", x_inner_dim, " and ", y_inner_dim,
        " of arguments to BatchedDot must be equal: ",
        xla::ShapeUtil::HumanString(x_shape), " transpose: ", transpose_x,
        " vs. ", xla::ShapeUtil::HumanString(y_shape),
        " transpose: ", transpose_y);
  }

  // Check for zero lhs/rhs dim size.
  if (xla::ShapeUtil::IsZeroElementArray(x_shape) ||
      xla::ShapeUtil::IsZeroElementArray(y_shape)) {
    std::vector<int64> dimensions(batch_dimension_numbers.size());
    for (int i = 0; i < batch_dimension_numbers.size(); ++i) {
      dimensions[i] = x_shape.dimensions(batch_dimension_numbers[i]);
    }
    int x_outer_dim = transpose_x ? (ndims - 1) : (ndims - 2);
    int y_outer_dim = transpose_y ? (ndims - 2) : (ndims - 1);
    dimensions.push_back(x_shape.dimensions(x_outer_dim));
    dimensions.push_back(y_shape.dimensions(y_outer_dim));
    return builder->Broadcast(
        builder->ConstantLiteral(xla::Literal::Zero(x_shape.element_type())),
        dimensions);
  }

  if (x_shape.element_type() == xla::C64 && conjugate_x) {
    x = builder->Conj(x);
  }
  if (y_shape.element_type() == xla::C64 && conjugate_y) {
    y = builder->Conj(y);
  }

  // If there are no batch dimensions, use a regular Dot.
  // TODO(b/69062148) Remove this code when Dot emitters can be passed
  // dimensions to transpose directly (i.e. without requiring a Transpose HLO).
  if (batch_dimension_numbers.empty()) {
    auto lhs = transpose_x ? builder->Transpose(x, {1, 0}) : x;
    auto rhs = transpose_y ? builder->Transpose(y, {1, 0}) : y;
    return builder->Dot(lhs, rhs);
  }

  xla::DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(x_inner_dim);
  dot_dnums.add_rhs_contracting_dimensions(y_inner_dim);
  for (auto batch_dimension_number : batch_dimension_numbers) {
    dot_dnums.add_lhs_batch_dimensions(batch_dimension_number);
    dot_dnums.add_rhs_batch_dimensions(batch_dimension_number);
  }
  return builder->DotGeneral(x, y, dot_dnums);
}

}  // namespace tensorflow
