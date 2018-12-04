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

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

xla::XlaOp BatchDot(xla::XlaOp x, xla::XlaOp y,
                    xla::PrecisionConfig::Precision precision) {
  xla::XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> xla::StatusOr<xla::XlaOp> {
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

    int x_inner_dim = ndims - 1;
    int y_inner_dim = ndims - 2;
    if (x_shape.dimensions(x_inner_dim) != y_shape.dimensions(y_inner_dim)) {
      return errors::InvalidArgument(
          "Dimensions ", x_inner_dim, " and ", y_inner_dim,
          " of arguments to BatchedDot must be equal: ",
          xla::ShapeUtil::HumanString(x_shape), " vs. ",
          xla::ShapeUtil::HumanString(y_shape));
    }

    // Check for zero lhs/rhs dim size.
    if (xla::ShapeUtil::IsZeroElementArray(x_shape) ||
        xla::ShapeUtil::IsZeroElementArray(y_shape)) {
      std::vector<int64> dimensions(batch_dimension_numbers.size());
      for (int i = 0; i < batch_dimension_numbers.size(); ++i) {
        dimensions[i] = x_shape.dimensions(batch_dimension_numbers[i]);
      }
      int x_outer_dim = ndims - 2;
      int y_outer_dim = ndims - 1;
      dimensions.push_back(x_shape.dimensions(x_outer_dim));
      dimensions.push_back(y_shape.dimensions(y_outer_dim));
      return xla::Broadcast(
          xla::ConstantLiteral(builder,
                               xla::LiteralUtil::Zero(x_shape.element_type())),
          dimensions);
    }

    xla::PrecisionConfig precision_proto;
    precision_proto.add_operand_precision(precision);
    precision_proto.add_operand_precision(precision);

    xla::DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(x_inner_dim);
    dot_dnums.add_rhs_contracting_dimensions(y_inner_dim);
    for (auto batch_dimension_number : batch_dimension_numbers) {
      dot_dnums.add_lhs_batch_dimensions(batch_dimension_number);
      dot_dnums.add_rhs_batch_dimensions(batch_dimension_number);
    }

    return xla::DotGeneral(x, y, dot_dnums, &precision_proto);
  });
}

}  // namespace tensorflow
