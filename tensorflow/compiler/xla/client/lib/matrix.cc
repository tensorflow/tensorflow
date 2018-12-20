/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/matrix.h"

#include <numeric>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64 m,
                     int64 n) {
  auto a = Iota(builder, type, m);
  auto b = Iota(builder, type, n);
  auto indicator = Eq(a, Broadcast(b, {m}), /*broadcast_dimensions=*/{0});
  return ConvertElementType(indicator, type);
}

XlaOp GetMatrixDiagonal(XlaOp x) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_dims >= 2);
    const int64 m = shape.dimensions(n_dims - 2);
    const int64 n = shape.dimensions(n_dims - 1);
    absl::Span<const int64> major_dims =
        AsInt64Slice(shape.dimensions()).subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, U32, n);
    auto b = Iota(builder, U32, m);
    auto indicator = Eq(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    auto mask = Broadcast(indicator, major_dims);

    // TPUs don't support S64 add reduction at the moment. But fortunately
    // OR-reductions work just as well for integers.
    XlaComputation reducer =
        primitive_util::IsIntegralType(shape.element_type())
            ? CreateScalarOrComputation(shape.element_type(), builder)
            : CreateScalarAddComputation(shape.element_type(), builder);

    return Reduce(Select(mask, x, Zeros(builder, shape)), ScalarLike(x, 0),
                  reducer, {m >= n ? n_dims - 2 : n_dims - 1});
  });
}

XlaOp TriangleMask(XlaOp x, int diagonal) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_dims >= 2);
    const int64 m = shape.dimensions(n_dims - 2);
    const int64 n = shape.dimensions(n_dims - 1);
    absl::Span<const int64> major_dims =
        AsInt64Slice(shape.dimensions()).subspan(/*pos=*/0, /*len=*/n_dims - 2);
    auto a = Iota(builder, S32, n);
    auto b = Iota(builder, S32, m) + ConstantR0<int32>(builder, diagonal);
    XlaOp indicator;
    indicator = Ge(b, Broadcast(a, {m}), /*broadcast_dimensions=*/{0});
    return Broadcast(indicator, major_dims);
  });
}

XlaOp Triangle(XlaOp x, bool lower) {
  return lower ? Select(TriangleMask(x, 0), x, ZerosLike(x))
               : Select(TriangleMask(x, -1), ZerosLike(x), x);
}

XlaOp UpperTriangle(XlaOp x) { return Triangle(x, false); }

XlaOp LowerTriangle(XlaOp x) { return Triangle(x, true); }

XlaOp BatchDot(XlaOp x, XlaOp y, PrecisionConfig::Precision precision) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape x_shape, builder->GetShape(x));
    TF_ASSIGN_OR_RETURN(Shape y_shape, builder->GetShape(y));

    // Check that both tensors have the same number of dimensions. There must be
    // at least two (the batch dimensions can be empty).
    if (ShapeUtil::Rank(x_shape) != ShapeUtil::Rank(y_shape)) {
      return InvalidArgument(
          "Arguments to BatchDot have different ranks: %s vs. %s",
          ShapeUtil::HumanString(x_shape), ShapeUtil::HumanString(y_shape));
    }
    const int ndims = ShapeUtil::Rank(x_shape);
    if (ndims < 2) {
      return InvalidArgument(
          "Arguments to BatchDot must have rank >= 2: got %d", ndims);
    }

    // The batch dimensions must be equal and the matrix dimensions must be
    // valid.
    std::vector<int64> batch_dimension_numbers;
    for (int i = 0; i < ndims - 2; ++i) {
      if (x_shape.dimensions(i) != y_shape.dimensions(i)) {
        return InvalidArgument(
            "Dimension %d of inputs to BatchDot must be equal: shapes %s vs %s",
            i, ShapeUtil::HumanString(x_shape),
            ShapeUtil::HumanString(y_shape));
      }
      batch_dimension_numbers.push_back(i);
    }

    int x_inner_dim = ndims - 1;
    int y_inner_dim = ndims - 2;
    if (x_shape.dimensions(x_inner_dim) != y_shape.dimensions(y_inner_dim)) {
      return InvalidArgument(
          "Dimensions %d and %d of arguments to BatchDot must be equal: "
          "shapes %s vs %s",
          x_inner_dim, y_inner_dim, ShapeUtil::HumanString(x_shape),
          ShapeUtil::HumanString(y_shape));
    }

    // Check for zero lhs/rhs dim size.
    if (ShapeUtil::IsZeroElementArray(x_shape) ||
        ShapeUtil::IsZeroElementArray(y_shape)) {
      std::vector<int64> dimensions(batch_dimension_numbers.size());
      for (int i = 0; i < batch_dimension_numbers.size(); ++i) {
        dimensions[i] = x_shape.dimensions(batch_dimension_numbers[i]);
      }
      int x_outer_dim = ndims - 2;
      int y_outer_dim = ndims - 1;
      dimensions.push_back(x_shape.dimensions(x_outer_dim));
      dimensions.push_back(y_shape.dimensions(y_outer_dim));
      return Broadcast(
          ConstantLiteral(builder, LiteralUtil::Zero(x_shape.element_type())),
          dimensions);
    }

    PrecisionConfig precision_proto;
    precision_proto.add_operand_precision(precision);
    precision_proto.add_operand_precision(precision);

    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(x_inner_dim);
    dot_dnums.add_rhs_contracting_dimensions(y_inner_dim);
    for (auto batch_dimension_number : batch_dimension_numbers) {
      dot_dnums.add_lhs_batch_dimensions(batch_dimension_number);
      dot_dnums.add_rhs_batch_dimensions(batch_dimension_number);
    }

    return DotGeneral(x, y, dot_dnums, &precision_proto);
  });
}

XlaOp TransposeInMinorDims(XlaOp x) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    const int64 n_dims = ShapeUtil::Rank(shape);
    TF_RET_CHECK(n_dims >= 2);
    std::vector<int64> permutation(n_dims);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::swap(permutation[n_dims - 1], permutation[n_dims - 2]);
    return Transpose(x, permutation);
  });
}

XlaOp MaybeTransposeInMinorDims(XlaOp x, bool transpose) {
  return transpose ? TransposeInMinorDims(x) : x;
}
}  // namespace xla
