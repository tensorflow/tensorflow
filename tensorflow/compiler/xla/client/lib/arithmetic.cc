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
namespace {

using XlaOpGenerator = XlaOp (*)(XlaBuilder*, const XlaOp&, const XlaOp&);

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
  generator(b.get(), lhs, rhs);
  return b->BuildAndNoteError();
}

}  // namespace

XlaComputation CreateScalarAddComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "add", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Add(lhs, rhs);
      });
}

XlaComputation CreateScalarMultiplyComputation(PrimitiveType type,
                                               XlaBuilder* builder) {
  return CreateScalarComputation(
      "mul", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Mul(lhs, rhs);
      });
}

XlaComputation CreateScalarGeComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
  return CreateScalarComputation("ge", type, builder,
                                 [](XlaBuilder* b, const XlaOp& lhs,
                                    const XlaOp& rhs) { return Ge(lhs, rhs); });
}

XlaComputation CreateScalarMaxComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "max", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Max(lhs, rhs);
      });
}

XlaComputation CreateScalarMinComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "min", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return Min(lhs, rhs);
      });
}

XlaComputation CreateScalarAndComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "and", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return And(lhs, rhs);
      });
}

XlaComputation CreateScalarOrComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
  return CreateScalarComputation("or", type, builder,
                                 [](XlaBuilder* b, const XlaOp& lhs,
                                    const XlaOp& rhs) { return Or(lhs, rhs); });
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

XlaOp ArgMinMax(XlaOp input, PrimitiveType output_type, int axis, bool is_min) {
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

    XlaOp input_max = Reduce(input, init_value, reducer,
                             /*dimensions_to_reduce=*/{axis});
    std::vector<int64> broadcast_dims(input_shape.rank() - 1);
    std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
    std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
    // Compute a mask that has 1s for elements equal to the maximum.
    XlaOp partial_mask =
        ConvertElementType(Eq(input, input_max, broadcast_dims), output_type);

    // In order to make identity elements for a bitwise And, we:
    //   Left shift the 1 to the leftmost bit, yielding 0x10...0
    //   Arithmetic right shift the 1 back to the rightmost bit, yielding
    //   0xFF...F
    int32 bits_in_type =
        ShapeUtil::ByteSizeOfPrimitiveType(output_type) * 8 - 1;
    XlaOp shift_amount = ConstantR0WithType(builder, output_type, bits_in_type);
    XlaOp full_mask = ShiftRightArithmetic(
        ShiftLeft(partial_mask, shift_amount), shift_amount);

    // And with the vector [0, 1, 2, ...] to convert each 0xFF...F into its
    // index.

    const int64 axis_size = ShapeUtil::GetDimension(input_shape, axis);
    XlaOp iota = Iota(builder, output_type, axis_size);
    XlaOp product = And(full_mask, iota, /*broadcast_dimensions=*/{axis});

    // If there are multiple maximum elements, choose the one with the highest
    // index.
    return Reduce(product, MinValue(builder, output_type),
                  CreateScalarMaxComputation(output_type, builder),
                  /*dimensions_to_reduce=*/{axis});
  });
}

}  // namespace

XlaOp ArgMax(XlaOp input, PrimitiveType output_type, int axis) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/false);
}

XlaOp ArgMin(XlaOp input, PrimitiveType output_type, int axis) {
  return ArgMinMax(input, output_type, axis, /*is_min=*/true);
}

}  // namespace xla
