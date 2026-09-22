/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/builder/lib/arithmetic.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

XlaComputation CreateScalarComputation(const std::string& name,
                                       PrimitiveType type, XlaBuilder* builder,
                                       XlaOpGenerator generator) {
  std::unique_ptr<XlaBuilder> b;
  if (type == PRED) {
    b = builder->CreateSubBuilder(name);
  } else {
    b = builder->CreateSubBuilder(
        absl::StrCat(name, "_", PrimitiveType_Name(type)));
  }

  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto lhs = Parameter(b.get(), 0, scalar, "lhs");
  auto rhs = Parameter(b.get(), 1, scalar, "rhs");
  generator(lhs, rhs);
  return b->BuildAndNoteError();
}

XlaComputation CreateScalarAddComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "add", type, builder, [](XlaOp lhs, XlaOp rhs) { return Add(lhs, rhs); });
}

XlaComputation CreateScalarMultiplyComputation(PrimitiveType type,
                                               XlaBuilder* builder) {
  return CreateScalarComputation(
      "mul", type, builder, [](XlaOp lhs, XlaOp rhs) { return Mul(lhs, rhs); });
}

XlaComputation CreateScalarGeComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
  return CreateScalarComputation(
      "ge", type, builder, [](XlaOp lhs, XlaOp rhs) { return Ge(lhs, rhs); });
}

XlaComputation CreateScalarMaxComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "max", type, builder, [](XlaOp lhs, XlaOp rhs) { return Max(lhs, rhs); });
}

XlaComputation CreateScalarMinComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "min", type, builder, [](XlaOp lhs, XlaOp rhs) { return Min(lhs, rhs); });
}

XlaComputation CreateScalarAndComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "and", type, builder, [](XlaOp lhs, XlaOp rhs) { return And(lhs, rhs); });
}

XlaComputation CreateScalarOrComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
  return CreateScalarComputation(
      "or", type, builder, [](XlaOp lhs, XlaOp rhs) { return Or(lhs, rhs); });
}

XlaComputation CreateScalarIdentityWithZeroComputation(PrimitiveType type,
                                                       XlaBuilder* builder) {
  XlaComputation reducer =
      (primitive_util::IsIntegralType(type) || type == PRED)
          ? CreateScalarOrComputation(type, builder)
          : CreateScalarAddComputation(type, builder);
  return reducer;
}

XlaOp Any(XlaOp predicates) {
  XlaBuilder* builder = predicates.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    auto f = ConstantR0<bool>(builder, false);
    XlaComputation logical_or = CreateScalarOrComputation(PRED, builder);
    ABSL_ASSIGN_OR_RETURN(const Shape& predicates_shape,
                     builder->GetShape(predicates));
    std::vector<int64_t> all_dimensions(predicates_shape.dimensions().size());
    absl::c_iota(all_dimensions, 0);
    return Reduce(predicates, f, logical_or, all_dimensions);
  });
}

static XlaComputation CreateMinMaxComputation(XlaBuilder* outer_builder,
                                              PrimitiveType value_type,
                                              PrimitiveType index_type,
                                              bool is_min) {
  auto sub_builder = outer_builder->CreateSubBuilder("minmax_func");
  XlaBuilder* b = sub_builder.get();
  XlaOp lhs_value =
      Parameter(b, 0, ShapeUtil::MakeShape(value_type, {}), "lhs_value");
  XlaOp lhs_index =
      Parameter(b, 1, ShapeUtil::MakeShape(index_type, {}), "lhs_index");
  XlaOp rhs_value =
      Parameter(b, 2, ShapeUtil::MakeShape(value_type, {}), "rhs_value");
  XlaOp rhs_index =
      Parameter(b, 3, ShapeUtil::MakeShape(index_type, {}), "rhs_index");

  XlaOp cmp = is_min ? Le(lhs_value, rhs_value) : Ge(lhs_value, rhs_value);
  XlaOp eq = Eq(lhs_value, rhs_value);
  if (primitive_util::HasNaN(value_type)) {
    // IEEE-754 comparisons involving a NaN are always false, so `cmp` on its
    // own makes the selections below fall through to the right-hand operand
    // whenever either side is NaN. That lets a NaN win the reduction and be
    // reported as the extremum's index. Order NaNs strictly after every
    // non-NaN value instead, so a NaN can only win when every reduced value is
    // NaN. This matches the Eigen reducers that back eager execution.
    //
    // `Ne(x, x)` is used instead of `IsNan()` from math.h because that library
    // depends on this one, so the reverse dependency would be circular.
    XlaOp lhs_is_nan = Ne(lhs_value, lhs_value);
    XlaOp rhs_is_nan = Ne(rhs_value, rhs_value);
    // When `rhs` is NaN, keep `lhs` unless it is NaN as well. Otherwise `cmp`
    // is already correct: it is false when only `lhs` is NaN, which discards
    // it in favor of `rhs`.
    cmp = Select(rhs_is_nan, Not(lhs_is_nan), cmp);
    // Treat two NaNs as equal so the tie-break below returns the lowest index.
    // This is required for correctness, not just determinism: `Reduce` applies
    // this computation in an unspecified order, so it must be commutative, and
    // without this the result would depend on which operand happened to be on
    // the right-hand side.
    eq = Or(eq, And(lhs_is_nan, rhs_is_nan));
  }
  XlaOp max = Select(cmp, lhs_value, rhs_value);
  XlaOp arg_max = Select(cmp, lhs_index, rhs_index);
  XlaOp tie_id = Min(lhs_index, rhs_index);
  arg_max = Select(eq, tie_id, arg_max);
  Tuple(b, {max, arg_max});
  return b->BuildAndNoteError();
}

XlaOp ArgMinMax(XlaOp input, PrimitiveType output_type, int axis, bool is_min) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> absl::StatusOr<XlaOp> {
    ABSL_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    XlaOp value_init_value;
    if (is_min) {
      value_init_value = MaxValue(builder, input_shape.element_type());
    } else {
      value_init_value = MinValue(builder, input_shape.element_type());
    }
    int64_t dimension_size = input_shape.dimensions(axis);
    auto index_type = dimension_size <= INT32_MAX ? S32 : output_type;
    XlaOp index_init_value = Zero(builder, index_type);
    auto iota_shape =
        ShapeUtil::MakeShape(index_type, input_shape.dimensions());
    XlaOp iota = Iota(builder, iota_shape, axis);

    XlaComputation reducer = CreateMinMaxComputation(
        builder, input_shape.element_type(), index_type, is_min);
    XlaOp max_argmax = Reduce(builder, {input, iota},
                              {value_init_value, index_init_value}, reducer,
                              /*dimensions_to_reduce=*/{axis});
    XlaOp argmax = GetTupleElement(max_argmax, 1);
    if (index_type != output_type) {
      argmax = ConvertElementType(argmax, output_type);
    }
    return argmax;
  });
}

XlaOp ArgMax(XlaOp input, PrimitiveType output_type, int axis) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/false);
}

}  // namespace xla
