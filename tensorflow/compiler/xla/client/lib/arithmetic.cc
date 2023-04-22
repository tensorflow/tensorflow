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

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

XlaComputation CreateScalarComputation(const string& name, PrimitiveType type,
                                       XlaBuilder* builder,
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
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    auto f = ConstantR0<bool>(builder, false);
    XlaComputation logical_or = CreateScalarOrComputation(PRED, builder);
    TF_ASSIGN_OR_RETURN(const Shape& predicates_shape,
                        builder->GetShape(predicates));
    std::vector<int64> all_dimensions(predicates_shape.rank());
    std::iota(all_dimensions.begin(), all_dimensions.end(), 0);
    return Reduce(predicates, f, logical_or, all_dimensions);
  });
}

namespace {

XlaComputation CreateMinMaxComputation(XlaBuilder* outer_builder,
                                       PrimitiveType value_type,
                                       PrimitiveType index_type, bool is_min,
                                       bool stable, bool tie_low) {
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
  XlaOp max = Select(cmp, lhs_value, rhs_value);
  XlaOp arg_max = Select(cmp, lhs_index, rhs_index);
  if (stable) {
    XlaOp eq = Eq(lhs_value, rhs_value);
    XlaOp tie_id =
        tie_low ? Min(lhs_index, rhs_index) : Max(lhs_index, rhs_index);
    arg_max = Select(eq, tie_id, arg_max);
  }
  Tuple(b, {max, arg_max});
  return b->BuildAndNoteError();
}

XlaOp ArgMinMax(XlaOp input, PrimitiveType output_type, int axis, bool is_min,
                bool stable, bool tie_low) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    XlaOp value_init_value;
    if (is_min) {
      value_init_value = MaxValue(builder, input_shape.element_type());
    } else {
      value_init_value = MinValue(builder, input_shape.element_type());
    }
    int64 dimension_size = input_shape.dimensions(axis);
    auto index_type = dimension_size <= INT32_MAX ? S32 : output_type;
    XlaOp index_init_value = Zero(builder, index_type);
    auto iota_shape = input_shape;
    iota_shape.set_element_type(index_type);
    XlaOp iota = Iota(builder, iota_shape, axis);

    XlaComputation reducer =
        CreateMinMaxComputation(builder, input_shape.element_type(), index_type,
                                is_min, stable, tie_low);
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

XlaOp ArgMinMaxTwoPass(XlaOp input, PrimitiveType output_type, int axis,
                       bool is_min, bool tie_low) {
  XlaBuilder* builder = input.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape input_shape, builder->GetShape(input));
    XlaOp init_value;
    XlaComputation reducer;
    if (is_min) {
      init_value = MaxValue(builder, input_shape.element_type());
      reducer = CreateScalarMinComputation(input_shape.element_type(), builder);
    } else {
      init_value = MinValue(builder, input_shape.element_type());
      reducer = CreateScalarMaxComputation(input_shape.element_type(), builder);
    }

    XlaOp iota = Iota(
        builder, ShapeUtil::ChangeElementType(input_shape, output_type), axis);
    XlaOp reduced_input = Reduce(input, init_value, reducer,
                                 /*dimensions_to_reduce=*/{axis});
    std::vector<int64> broadcast_dims(input_shape.rank() - 1);
    std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
    std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
    if (tie_low) {
      XlaOp max_idx = MaxValue(builder, output_type);
      XlaOp select_mask = Select(Eq(input, reduced_input, broadcast_dims),
                                 /*on_true=*/iota,
                                 /*on_false=*/
                                 max_idx);
      return Reduce(select_mask, max_idx,
                    CreateScalarMinComputation(output_type, builder),
                    /*dimensions_to_reduce=*/{axis});
    } else {
      XlaOp min_idx = MinValue(builder, output_type);
      XlaOp select_mask = Select(Eq(input, reduced_input, broadcast_dims),
                                 /*on_true=*/iota,
                                 /*on_false=*/
                                 min_idx);
      return Reduce(select_mask, min_idx,
                    CreateScalarMaxComputation(output_type, builder),
                    /*dimensions_to_reduce=*/{axis});
    }
  });
}
}  // namespace

XlaOp ArgMax(XlaOp input, PrimitiveType output_type, int axis, bool stable,
             bool tie_low) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/false, stable, tie_low);
}

XlaOp ArgMin(XlaOp input, PrimitiveType output_type, int axis, bool stable,
             bool tie_low) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/true, stable, tie_low);
}

XlaOp ArgMaxTwoPass(XlaOp input, PrimitiveType output_type, int axis,
                    bool tie_low) {
  return ArgMinMaxTwoPass(input, output_type, axis, /*is_min=*/false, tie_low);
}

XlaOp ArgMinTwoPass(XlaOp input, PrimitiveType output_type, int axis,
                    bool tie_low) {
  return ArgMinMaxTwoPass(input, output_type, axis, /*is_min=*/true, tie_low);
}
}  // namespace xla
