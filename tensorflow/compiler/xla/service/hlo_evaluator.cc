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
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

template <typename NativeT>
std::unique_ptr<Literal> ElementWiseUnaryOp(
    const Shape& shape, std::function<NativeT(NativeT)>&& unary_op,
    const Literal& operand) {
  DCHECK(ShapeUtil::SameDimensions(shape, operand.shape()));

  auto result = MakeUnique<Literal>();
  *result->mutable_shape() = shape;
  LiteralUtil::Reserve(ShapeUtil::ElementsIn(shape), result.get());

  std::vector<int64> multi_index(ShapeUtil::Rank(result->shape()), 0);
  do {
    LiteralUtil::Set<NativeT>(
        result.get(), multi_index,
        unary_op(LiteralUtil::Get<NativeT>(operand, multi_index)));
  } while (IndexUtil::BumpIndices(result->shape(), &multi_index));

  return result;
}

template <typename NativeT>
std::unique_ptr<Literal> ElementWiseBinaryOp(
    const Shape& shape, std::function<NativeT(NativeT, NativeT)>&& binary_op,
    const Literal& lhs, const Literal& rhs) {
  DCHECK(ShapeUtil::SameDimensions(shape, rhs.shape()));
  DCHECK(ShapeUtil::SameDimensions(lhs.shape(), rhs.shape()));

  auto result = MakeUnique<Literal>();
  *result->mutable_shape() = shape;
  LiteralUtil::Reserve(ShapeUtil::ElementsIn(shape), result.get());

  std::vector<int64> multi_index(ShapeUtil::Rank(result->shape()), 0);
  do {
    LiteralUtil::Set<NativeT>(
        result.get(), multi_index,
        binary_op(LiteralUtil::Get<NativeT>(lhs, multi_index),
                  LiteralUtil::Get<NativeT>(rhs, multi_index)));
  } while (IndexUtil::BumpIndices(result->shape(), &multi_index));

  return result;
}

template <typename NativeT, typename LhsType, typename RhsType,
          typename EhsType>
std::unique_ptr<Literal> ElementWiseTernaryOp(
    const Shape& shape,
    std::function<NativeT(LhsType lhs, RhsType rhs, EhsType ehs)>&& ternary_op,
    const Literal& lhs, const Literal& rhs, const Literal& ehs) {
  DCHECK(ShapeUtil::SameDimensions(shape, lhs.shape()));
  DCHECK(ShapeUtil::SameDimensions(lhs.shape(), rhs.shape()));
  DCHECK(ShapeUtil::SameDimensions(rhs.shape(), ehs.shape()));

  auto result = MakeUnique<Literal>();
  *result->mutable_shape() = shape;
  LiteralUtil::Reserve(ShapeUtil::ElementsIn(shape), result.get());

  std::vector<int64> multi_index(ShapeUtil::Rank(result->shape()), 0);
  do {
    LiteralUtil::Set<NativeT>(
        result.get(), multi_index,
        ternary_op(LiteralUtil::Get<LhsType>(lhs, multi_index),
                   LiteralUtil::Get<RhsType>(rhs, multi_index),
                   LiteralUtil::Get<EhsType>(ehs, multi_index)));
  } while (IndexUtil::BumpIndices(result->shape(), &multi_index));

  return result;
}

// Templated abs so that unsigned types can be passed in without warning.
template <
    typename NativeT,
    typename std::enable_if<std::is_unsigned<NativeT>::value>::type* = nullptr>
NativeT AbsoluteVal(NativeT value) {
  return value;
}

template <
    typename NativeT,
    typename std::enable_if<std::is_signed<NativeT>::value>::type* = nullptr>
NativeT AbsoluteVal(NativeT value) {
  return std::abs(value);
}

template <typename NativeT>
StatusOr<std::unique_ptr<Literal>> EvaluateOpForLiteralInternal(
    HloInstruction* instruction) {
  DCHECK(hlo_query::AllOperandsAreConstants(*instruction));

  const std::vector<HloInstruction*>& operands = instruction->operands();
  HloOpcode opcode = instruction->opcode();
  const Shape& shape = instruction->shape();

  switch (opcode) {
    // TODO(b/35950897): many of the stl function used here are not overloaded
    // for all XLA primitive types.
    // Unary element-wise ops.
    case HloOpcode::kAbs:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return AbsoluteVal(operand); },
          operands[0]->literal());
    case HloOpcode::kCeil:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return std::ceil(operand); },
          operands[0]->literal());
    case HloOpcode::kConvert:
      CHECK_EQ(operands.size(), 1);
      // TODO(b/35950897): implement Convert.
      return Unimplemented("unhandled HLO ops for HloEvaluator: %s.",
                           HloOpcodeString(opcode).c_str());
    case HloOpcode::kCopy:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return operand; },
          operands[0]->literal());
    case HloOpcode::kExp:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return std::exp(operand); },
          operands[0]->literal());
    case HloOpcode::kFloor:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return std::floor(operand); },
          operands[0]->literal());
    case HloOpcode::kIsFinite:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return std::isfinite(operand); },
          operands[0]->literal());
    case HloOpcode::kLog:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return std::log(operand); },
          operands[0]->literal());
    case HloOpcode::kLogicalNot:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return !operand; },
          operands[0]->literal());
    case HloOpcode::kNegate:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return -operand; },
          operands[0]->literal());
    case HloOpcode::kSign:
      CHECK_EQ(operands.size(), 1);
      CHECK(primitive_util::IsIntegralType(shape.element_type()));
      return ElementWiseUnaryOp<int>(shape,
                                     [](NativeT operand) {
                                       return (NativeT(0) < operand) -
                                              (operand < NativeT(0));
                                     },
                                     operands[0]->literal());
    case HloOpcode::kTanh:
      CHECK_EQ(operands.size(), 1);
      return ElementWiseUnaryOp<NativeT>(
          shape, [](NativeT operand) { return std::tanh(operand); },
          operands[0]->literal());
    // Binary element-wise ops.
    case HloOpcode::kAdd:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs + rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kDivide:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs / rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kMultiply:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs * rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kSubtract:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs - rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kEq:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<bool>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs == rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kGe:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<bool>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs >= rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kGt:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<bool>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs > rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kLe:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<bool>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs <= rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kLt:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<bool>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs < rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kNe:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<bool>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs != rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kMaximum:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return std::max(lhs, rhs); },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kMinimum:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return std::min(lhs, rhs); },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kPower:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return std::pow(lhs, rhs); },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kRemainder:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape,
          [](NativeT lhs, NativeT rhs) { return std::remainder(lhs, rhs); },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kLogicalAnd:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs && rhs; },
          operands[0]->literal(), operands[1]->literal());
    case HloOpcode::kLogicalOr:
      CHECK_EQ(operands.size(), 2);
      return ElementWiseBinaryOp<NativeT>(
          shape, [](NativeT lhs, NativeT rhs) { return lhs || rhs; },
          operands[0]->literal(), operands[1]->literal());
    // Ternary element-wise ops.
    case HloOpcode::kClamp: {
      CHECK_EQ(operands.size(), 3);
      std::function<NativeT(NativeT, NativeT, NativeT)> clamp_op =
          [](NativeT low, NativeT high, NativeT value) {
            return std::max(low, std::min(value, high));
          };
      return ElementWiseTernaryOp<NativeT, NativeT, NativeT, NativeT>(
          shape, std::move(clamp_op), operands[0]->literal(),
          operands[1]->literal(), operands[2]->literal());
    } break;
    case HloOpcode::kSelect: {
      CHECK_EQ(operands.size(), 3);
      CHECK(!ShapeUtil::IsTuple(shape));
      std::function<NativeT(bool, NativeT, NativeT)> select_op =
          [](bool pred, NativeT on_true, NativeT on_false) {
            if (pred) {
              return on_true;
            }
            return on_false;
          };
      return ElementWiseTernaryOp<NativeT, bool, NativeT, NativeT>(
          shape, std::move(select_op), operands[0]->literal(),
          operands[1]->literal(), operands[2]->literal());
    } break;
    default:
      return Unimplemented("unhandled HLO ops for HloEvaluator: %s.",
                           HloOpcodeString(opcode).c_str());
  }
}

}  // namespace

/* static */ StatusOr<std::unique_ptr<Literal>>
HloEvaluator::EvaluateOpForLiteral(HloInstruction* instruction) {
  DCHECK(hlo_query::AllOperandsAreConstants(*instruction));

  Shape shape = instruction->shape();
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape));

  // REVIEW QUESTION: other than a few operations, do we need to handle the
  // general case of operands being of different types in the context of the
  // evaluator?

  switch (shape.element_type()) {
    case PRED:
      return EvaluateOpForLiteralInternal<bool>(instruction);
    case U8:
      return EvaluateOpForLiteralInternal<uint8>(instruction);
    case U16:
      LOG(FATAL) << "U16/uint16 is unimplemented.";
    case U32:
      return EvaluateOpForLiteralInternal<uint32>(instruction);
    case U64:
      return EvaluateOpForLiteralInternal<uint64>(instruction);
    case S8:
      return EvaluateOpForLiteralInternal<int8>(instruction);
    case S16:
      LOG(FATAL) << "S16/int16 is unimplemented.";
    case S32:
      return EvaluateOpForLiteralInternal<int32>(instruction);
    case S64:
      return EvaluateOpForLiteralInternal<int64>(instruction);
    case F16:
      LOG(FATAL) << "F16 is unimplemented.";
    case F32:
      return EvaluateOpForLiteralInternal<float>(instruction);
    case F64:
      return EvaluateOpForLiteralInternal<double>(instruction);
    default:
      return Unimplemented("unhandled primitive type: %s.",
                           PrimitiveType_Name(shape.element_type()).c_str());
  }
}

}  // namespace xla
