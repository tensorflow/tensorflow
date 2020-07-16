/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/comparators.h"

#include <limits>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

using XlaOpGenerator = XlaOp (*)(XlaOp, XlaOp, absl::Span<const int64>);

XlaOp BitcastConvertFloatingPointToIntegral(const XlaOp& value,
                                            int64 bit_width) {
  PrimitiveType signed_type;
  PrimitiveType unsigned_type;
  XlaOp max_value;
  switch (bit_width) {
    case 16:
      max_value =
          ConstantR0(value.builder(),
                     static_cast<uint16>(std::numeric_limits<int16>::max()));
      signed_type = S16;
      unsigned_type = U16;
      break;
    case 32:
      max_value =
          ConstantR0(value.builder(),
                     static_cast<uint32>(std::numeric_limits<int32>::max()));
      signed_type = S32;
      unsigned_type = U32;
      break;
    case 64:
      max_value =
          ConstantR0(value.builder(),
                     static_cast<uint64>(std::numeric_limits<int64>::max()));
      signed_type = S64;
      unsigned_type = U64;
      break;
    default:
      return value.builder()->ReportError(
          InvalidArgument("Invalid bit width %lld for Comparator floating "
                          "point parameter.",
                          bit_width));
  }
  // Switch from a floating point value to a integer value in such a way that
  // when using the integer value to compare, we get the same result for normal
  // values, and -Nan is treated as the smallest value, and Nan is treated as
  // the largest value.
  // If f is a float, and
  // x = bit_cast<int32>(f);
  // y = x < 0 ? numeric_limits<int32>::max() - x : x;
  // then y is ordered as an int32 such that finite values have the obvious
  // order, -0 is ordered before 0, and -NaN and NaN appear at the beginning
  // and end of the ordering.
  // Note that in order to avoid -x to overflow, we calculate
  // numeric_limits<int32>::max() - x as unsigned, and then convert back to
  // signed.
  auto signed_value = BitcastConvertType(value, signed_type);
  auto unsigned_value = BitcastConvertType(value, unsigned_type);
  auto flipped_value =
      BitcastConvertType(Sub(max_value, unsigned_value), signed_type);
  auto is_negative = Lt(signed_value, Zero(value.builder(), signed_type));
  return Select(is_negative, flipped_value, signed_value);
}

void ConvertFloatingPoint(const PrimitiveType& operand_type, XlaOp* lhs_param,
                          XlaOp* rhs_param) {
  if (primitive_util::IsFloatingPointType(operand_type)) {
    PrimitiveType compare_type = operand_type;
    // Special-case handling for BF16. We currently do not support direct
    // comparisons with BF16, so we convert to F32 and then use the F32
    // comparison logic.
    if (compare_type == BF16) {
      compare_type = F32;
      *lhs_param = ConvertElementType(*lhs_param, F32);
      *rhs_param = ConvertElementType(*rhs_param, F32);
    }
    int64 bit_width = primitive_util::BitWidth(compare_type);
    *lhs_param = BitcastConvertFloatingPointToIntegral(*lhs_param, bit_width);
    *rhs_param = BitcastConvertFloatingPointToIntegral(*rhs_param, bit_width);
  }
}

XlaComputation CreateScalarComparisonComputation(
    const string& name, const std::vector<PrimitiveType>& operand_types,
    XlaBuilder* builder, XlaOpGenerator generator) {
  CHECK_NE(operand_types.size(), 0);
  std::vector<absl::optional<XlaOpGenerator>> generators(operand_types.size());
  generators[0] = generator;
  return CreateScalarComparisonComputation(name, operand_types, generators,
                                           builder);
}
}  // namespace

XlaComputation CreateScalarComparisonComputation(
    const string& name, const std::vector<PrimitiveType>& operand_types,
    const std::vector<absl::optional<XlaOpGenerator>>& generators,
    XlaBuilder* builder) {
  // Create a default computation where we compare only the first two
  // parameters of type 'operand_types[0]'.
  auto b = builder->CreateSubBuilder(name);
  if (operand_types.empty()) {
    b->ReportError(InvalidArgument("operand_types should not be empty"));
    return b->BuildAndNoteError();
  }

  CHECK_EQ(operand_types.size(), generators.size());
  int64 parameter_count = 0;
  int64 last_generator_index = 0;
  std::vector<XlaOp> lhs_params;
  std::vector<XlaOp> rhs_params;

  // For each type in 'operand_types' we create two parameters of this type. The
  // idea is that this computation can be used by n-ary Sort, and potentially
  // should support comparing also the other operands of sort. In this default
  // computation, however, we will not actually use any parameters except the
  // first two.
  for (auto operand_type : operand_types) {
    auto scalar_shape = ShapeUtil::MakeShape(operand_type, {});
    auto lhs_param = Parameter(b.get(), parameter_count * 2, scalar_shape,
                               absl::StrCat("p.", parameter_count, ".lhs"));
    auto rhs_param = Parameter(b.get(), parameter_count * 2 + 1, scalar_shape,
                               absl::StrCat("p.", parameter_count, ".rhs"));
    ConvertFloatingPoint(operand_type, &lhs_param, &rhs_param);
    lhs_params.emplace_back(lhs_param);
    rhs_params.emplace_back(rhs_param);
    if (generators[parameter_count].has_value()) {
      last_generator_index = parameter_count;
    }
    parameter_count++;
  }

  CHECK_NE(parameter_count, 0);

  Shape shape = b->GetShape(lhs_params[0]).ValueOrDie();
  shape.set_element_type(PRED);
  XlaOp param_equal = Broadcast(One(b.get(), shape.element_type()),
                                AsInt64Slice(shape.dimensions()));
  XlaOp result = param_equal;

  for (int64 i = 0; i < parameter_count; i++) {
    if (generators[i].has_value()) {
      result = Select(param_equal,
                      generators[i].value()(lhs_params[i], rhs_params[i], {}),
                      result);
      if (i != last_generator_index) {
        param_equal = And(param_equal, Eq(lhs_params[i], rhs_params[i]));
      }
    }
  }

  return b->BuildAndNoteError();
}

// Creates a scalar less-than computation and returns it.
XlaComputation CreateScalarLtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
  return CreateScalarComparisonComputation("compare-less-than", operand_types,
                                           builder, Lt);
}

// Creates a scalar greater-than computation and returns it.
XlaComputation CreateScalarGtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
  return CreateScalarComparisonComputation("compare-greater-than",
                                           operand_types, builder, Gt);
}

}  // namespace xla
