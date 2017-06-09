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
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

template <typename OperandT>
StatusOr<std::unique_ptr<Literal>> Compare(const Shape& shape, HloOpcode opcode,
                                           const Literal& lhs_literal,
                                           const Literal& rhs_literal) {
  std::function<bool(OperandT, OperandT)> compare_op;
  switch (opcode) {
    case HloOpcode::kEq:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case HloOpcode::kNe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    case HloOpcode::kGe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el >= rhs_el;
      };
      break;
    case HloOpcode::kGt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el > rhs_el;
      };
      break;
    case HloOpcode::kLe:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el <= rhs_el;
      };
      break;
    case HloOpcode::kLt:
      compare_op = [](OperandT lhs_el, OperandT rhs_el) {
        return lhs_el < rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled HLO opcode for conversion to Comparison: "
                 << HloOpcodeString(opcode);
  }

  auto result = LiteralUtil::CreateFromShape(shape);
  TF_RETURN_IF_ERROR(LiteralUtil::Populate<bool>(
      result.get(), [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return compare_op(LiteralUtil::Get<OperandT>(lhs_literal, multi_index),
                          LiteralUtil::Get<OperandT>(rhs_literal, multi_index));
      }));

  return std::move(result);
}

template <typename ReturnT, typename NativeT>
StatusOr<std::unique_ptr<Literal>> ElementWiseUnaryOpImpl(
    HloInstruction* instruction,
    const std::function<ReturnT(NativeT)>& unary_op,
    const Literal& operand_literal) {
  const auto shape = instruction->shape();
  const auto* operand = instruction->operand(0);

  // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
  // removed.
  if (!ShapeUtil::SameDimensions(shape, operand->shape())) {
    return Unimplemented(
        "Implicit broadcasting is currently unsupported in HLO evaluator "
        "Shape Mismatch: %s vs %s",
        ShapeUtil::HumanString(shape).c_str(),
        ShapeUtil::HumanString(operand->shape()).c_str());
  }

  auto result = LiteralUtil::CreateFromShape(shape);

  TF_RETURN_IF_ERROR(LiteralUtil::Populate<ReturnT>(
      result.get(), [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return unary_op(
            LiteralUtil::Get<NativeT>(operand_literal, multi_index));
      }));
  return std::move(result);
}

}  // namespace

template <typename ReturnT>
class HloEvaluator::TypedVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit TypedVisitor(HloEvaluator* p) : parent_(p) {}

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Unimplemented("unhandled HLO ops for HloEvaluator: %s.",
                         HloOpcodeString(hlo_instruction->opcode()).c_str());
  };

  // TODO(b/35950897): many of the stl functions used in the handlers are not
  // overloaded for every XLA primitive types.

  template <typename NativeT,
            typename std::enable_if<std::is_unsigned<NativeT>::value>::type* =
                nullptr>
  Status HandleAbs(HloInstruction* abs, HloInstruction* operand) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return elem_operand;
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_signed<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs, HloInstruction* operand) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return Status::OK();
  }

  Status HandleAbs(HloInstruction* abs, HloInstruction* operand) override {
    return HandleAbs<ReturnT>(abs, operand);
  };

  Status HandleCeil(HloInstruction* ceil, HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[ceil],
                        ElementWiseUnaryOp(ceil, [](ReturnT elem_operand) {
                          return std::ceil(elem_operand);
                        }));
    return Status::OK();
  };

  Status HandleCopy(HloInstruction* copy, HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[copy],
                        ElementWiseUnaryOp(copy, [](ReturnT elem_operand) {
                          return elem_operand;
                        }));
    return Status::OK();
  };

  template <PrimitiveType src_type, PrimitiveType dest_type>
  std::unique_ptr<Literal> ConvertIfTypesMatch(const Literal& src_literal) {
    DCHECK_EQ(src_type, src_literal.shape().element_type());
    return LiteralUtil::Convert<
        typename primitive_util::PrimitiveTypeToNative<src_type>::type,
        typename primitive_util::PrimitiveTypeToNative<dest_type>::type>(
        src_literal);
  }

  Status HandleConvert(HloInstruction* convert,
                       HloInstruction* operand) override {
    auto operand_literal = parent_->GetEvaluatedLiteralFor(operand);

    switch (operand->shape().element_type()) {
#define CONVERT_IF_TYPES_MATCH(src_type)                                \
  case (src_type):                                                      \
    parent_->evaluated_[convert] = LiteralUtil::Convert<                \
        typename primitive_util::PrimitiveTypeToNative<src_type>::type, \
        ReturnT>(operand_literal);                                      \
    break;
      CONVERT_IF_TYPES_MATCH(PRED)
      CONVERT_IF_TYPES_MATCH(S8)
      CONVERT_IF_TYPES_MATCH(S32)
      CONVERT_IF_TYPES_MATCH(S64)
      CONVERT_IF_TYPES_MATCH(U8)
      CONVERT_IF_TYPES_MATCH(U32)
      CONVERT_IF_TYPES_MATCH(U64)
      CONVERT_IF_TYPES_MATCH(F32)
      CONVERT_IF_TYPES_MATCH(F64)
#undef CONVERT_IF_TYPES_MATCH
      // Other types are not yet supported.
      default:
        LOG(FATAL) << "unimplemented operand type for HandleCovert: "
                   << PrimitiveType_Name(operand->shape().element_type());
    }

    return Status::OK();
  }

  Status HandleExp(HloInstruction* exp, HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[exp],
                        ElementWiseUnaryOp(exp, [](ReturnT elem_operand) {
                          return std::exp(elem_operand);
                        }));
    return Status::OK();
  };

  Status HandleFloor(HloInstruction* floor, HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[floor],
                        ElementWiseUnaryOp(floor, [](ReturnT elem_operand) {
                          return std::floor(elem_operand);
                        }));
    return Status::OK();
  };

  Status HandleLog(HloInstruction* log, HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[log],
                        ElementWiseUnaryOp(log, [](ReturnT elem_operand) {
                          return std::log(elem_operand);
                        }));
    return Status::OK();
  };

  Status HandleLogicalNot(HloInstruction* logical_not,
                          HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[logical_not],
        ElementWiseUnaryOp(logical_not,
                           [](ReturnT elem_operand) { return !elem_operand; }));
    return Status::OK();
  };

  Status HandleNegate(HloInstruction* negate,
                      HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[negate],
                        ElementWiseUnaryOp(negate, [](ReturnT elem_operand) {
                          return -elem_operand;
                        }));
    return Status::OK();
  };

  Status HandleSign(HloInstruction* sign, HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ReturnT elem_operand) {
                          return (ReturnT(0) < elem_operand) -
                                 (elem_operand < ReturnT(0));
                        }));
    return Status::OK();
  };

  Status HandleTanh(HloInstruction* tanh, HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[tanh],
                        ElementWiseUnaryOp(tanh, [](ReturnT elem_operand) {
                          return std::tanh(elem_operand);
                        }));
    return Status::OK();
  };

  Status HandleMultiply(HloInstruction* multiply, HloInstruction* lhs,
                        HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(multiply, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem * rhs_elem;
        }));
    return Status::OK();
  };

  Status HandleSubtract(HloInstruction* subtract, HloInstruction* lhs,
                        HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(subtract, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem - rhs_elem;
        }));
    return Status::OK();
  };

  Status HandleAdd(HloInstruction* add, HloInstruction* lhs,
                   HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[add],
        ElementWiseBinaryOp(add, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem + rhs_elem;
        }));
    return Status::OK();
  };

  Status HandleDivide(HloInstruction* divide, HloInstruction* lhs,
                      HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[divide],
        ElementWiseBinaryOp(divide, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem / rhs_elem;
        }));
    return Status::OK();
  };

  Status HandleMaximum(HloInstruction* maximum, HloInstruction* lhs,
                       HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[maximum],
        ElementWiseBinaryOp(maximum, [](ReturnT lhs, ReturnT rhs) {
          return std::fmax(lhs, rhs);
        }));
    return Status::OK();
  };

  Status HandleMinimum(HloInstruction* minimum, HloInstruction* lhs,
                       HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[minimum],
        ElementWiseBinaryOp(minimum, [](ReturnT lhs_el, ReturnT rhs_el) {
          return std::fmin(lhs_el, rhs_el);
        }));
    return Status::OK();
  };

  Status HandlePower(HloInstruction* power, HloInstruction* lhs,
                     HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[power],
        ElementWiseBinaryOp(power, [](ReturnT lhs_el, ReturnT rhs_el) {
          return std::pow(lhs_el, rhs_el);
        }));
    return Status::OK();
  };

  Status HandleRemainder(HloInstruction* remainder, HloInstruction* lhs,
                         HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[remainder],
        ElementWiseBinaryOp(remainder, [](ReturnT lhs_el, ReturnT rhs_el) {
          return std::fmod(lhs_el, rhs_el);
        }));
    return Status::OK();
  };

  Status HandleLogicalAnd(HloInstruction* logical_and, HloInstruction* lhs,
                          HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[logical_and],
        ElementWiseBinaryOp(logical_and, [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el && rhs_el;
        }));
    return Status::OK();
  };

  Status HandleLogicalOr(HloInstruction* logical_or, HloInstruction* lhs,
                         HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[logical_or],
        ElementWiseBinaryOp(logical_or, [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el || rhs_el;
        }));
    return Status::OK();
  };

  Status HandleClamp(HloInstruction* clamp, HloInstruction* min,
                     HloInstruction* arg, HloInstruction* max) override {
    std::function<ReturnT(ReturnT, ReturnT, ReturnT)> clamp_op =
        [](ReturnT low, ReturnT high, ReturnT value) {
          return std::fmax(low, std::fmin(value, high));
        };
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[clamp],
                        ElementWiseTernaryOp(clamp, std::move(clamp_op)));
    return Status::OK();
  };

  Status HandleSelect(HloInstruction* select, HloInstruction* pred,
                      HloInstruction* on_true,
                      HloInstruction* on_false) override {
    CHECK(!ShapeUtil::IsTuple(select->shape()));
    std::function<ReturnT(bool, ReturnT, ReturnT)> select_op =
        [](bool pred, ReturnT on_true, ReturnT on_false) {
          if (pred) {
            return on_true;
          }
          return on_false;
        };
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[select],
                        ElementWiseTernaryOp(select, std::move(select_op)));
    return Status::OK();
  };

  Status Preprocess(HloInstruction* hlo) override {
    VLOG(2) << hlo->ToString();
    return Status::OK();
  };

 private:
  StatusOr<std::unique_ptr<Literal>> ElementWiseUnaryOp(
      HloInstruction* instruction,
      const std::function<ReturnT(ReturnT)>& unary_op) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(instruction->operand(0));
    return ElementWiseUnaryOpImpl<ReturnT, ReturnT>(instruction, unary_op,
                                                    operand_literal);
  }

  StatusOr<std::unique_ptr<Literal>> ElementWiseBinaryOp(
      HloInstruction* instruction,
      const std::function<ReturnT(ReturnT, ReturnT)>& binary_op) {
    const auto shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
    // removed.
    if (!(ShapeUtil::SameDimensions(shape, rhs->shape()) &&
          ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()))) {
      return Unimplemented(
          "Implicit broadcasting is currently unsupported in HLO evaluator "
          "Shape Mismatch: %s vs %s vs %s: ",
          ShapeUtil::HumanString(shape).c_str(),
          ShapeUtil::HumanString(lhs->shape()).c_str(),
          ShapeUtil::HumanString(rhs->shape()).c_str());
    }

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    auto result = LiteralUtil::CreateFromShape(shape);

    TF_RETURN_IF_ERROR(LiteralUtil::Populate<ReturnT>(
        result.get(), [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return binary_op(LiteralUtil::Get<ReturnT>(lhs_literal, multi_index),
                           LiteralUtil::Get<ReturnT>(rhs_literal, multi_index));
        }));
    return std::move(result);
  }

  template <typename LhsType, typename RhsType, typename EhsType>
  StatusOr<std::unique_ptr<Literal>> ElementWiseTernaryOp(
      HloInstruction* instruction,
      const std::function<ReturnT(LhsType, RhsType, EhsType)>& ternary_op) {
    const auto shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);
    const auto* ehs = instruction->operand(2);

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
    // removed.
    if (!(ShapeUtil::SameDimensions(shape, lhs->shape()) &&
          ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()) &&
          ShapeUtil::SameDimensions(rhs->shape(), ehs->shape()))) {
      return Unimplemented(
          "Implicit broadcasting is currently unsupported in HLO evaluator "
          "Shape Mismatch: %s vs %s vs %s vs %s: ",
          ShapeUtil::HumanString(shape).c_str(),
          ShapeUtil::HumanString(lhs->shape()).c_str(),
          ShapeUtil::HumanString(rhs->shape()).c_str(),
          ShapeUtil::HumanString(ehs->shape()).c_str());
    }

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);
    const Literal& ehs_literal = parent_->GetEvaluatedLiteralFor(ehs);

    auto result = LiteralUtil::CreateFromShape(shape);

    TF_RETURN_IF_ERROR(LiteralUtil::Populate<ReturnT>(
        result.get(), [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return ternary_op(
              LiteralUtil::Get<LhsType>(lhs_literal, multi_index),
              LiteralUtil::Get<RhsType>(rhs_literal, multi_index),
              LiteralUtil::Get<EhsType>(ehs_literal, multi_index));
        }));

    return std::move(result);
  }

  HloEvaluator* parent_;
};

HloEvaluator::HloEvaluator() {
  typed_visitors_[PRED] = MakeUnique<TypedVisitor<bool>>(this);
  typed_visitors_[U8] = MakeUnique<TypedVisitor<uint8>>(this);
  typed_visitors_[U16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("unhandled primitive type: U16.");
  });
  typed_visitors_[U32] = MakeUnique<TypedVisitor<uint32>>(this);
  typed_visitors_[U64] = MakeUnique<TypedVisitor<uint64>>(this);
  typed_visitors_[S8] = MakeUnique<TypedVisitor<int8>>(this);
  typed_visitors_[S16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("unhandled primitive type: S16.");
  });
  typed_visitors_[S32] = MakeUnique<TypedVisitor<int32>>(this);
  typed_visitors_[S64] = MakeUnique<TypedVisitor<int64>>(this);
  typed_visitors_[F16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("unhandled primitive type: F16.");
  });
  typed_visitors_[F32] = MakeUnique<TypedVisitor<float>>(this);
  typed_visitors_[F64] = MakeUnique<TypedVisitor<double>>(this);
  typed_visitors_[TUPLE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("unhandled primitive type: TUPLE.");
  });
  typed_visitors_[OPAQUE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("unhandled primitive type: OPAQUE.");
  });
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloComputation* computation,
    tensorflow::gtl::ArraySlice<const Literal*> args) {
  arg_literals_ = args;
  evaluated_.clear();

  TF_RETURN_IF_ERROR(computation->Accept(this));
  return MakeUnique<Literal>(
      GetEvaluatedLiteralFor(computation->root_instruction()));
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction,
    tensorflow::gtl::ArraySlice<const Literal*> operands) {
  TF_RET_CHECK(hlo_query::AllOperandsAreParametersOrConstants(*instruction));
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

  arg_literals_ = operands;
  evaluated_.clear();

  // Evaluate operands of Parameter type against the input literals which
  // caches the evaluated literal results.
  for (const auto operand : instruction->operands()) {
    if (operand->opcode() == HloOpcode::kParameter) {
      const Literal* input_literal = arg_literals_[operand->parameter_number()];
      VLOG(2) << "Parameter operand evaluated to: "
              << LiteralUtil::ToString(*input_literal);
      TF_RET_CHECK(ShapeUtil::Equal(operand->shape(), input_literal->shape()));

      evaluated_[operand] = MakeUnique<Literal>(*input_literal);
    }
  }

  TF_RETURN_IF_ERROR(instruction->Visit(this));
  return MakeUnique<Literal>(GetEvaluatedLiteralFor(instruction));
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction) {
  TF_RET_CHECK(hlo_query::AllOperandsAreConstants(*instruction));
  TF_RET_CHECK(instruction->opcode() != HloOpcode::kParameter);
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

  arg_literals_.clear();
  evaluated_.clear();
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  return MakeUnique<Literal>(GetEvaluatedLiteralFor(instruction));
}

std::unique_ptr<Literal> HloEvaluator::TryEvaluate(
    HloInstruction* instruction) {
  auto result_or = Evaluate(instruction);
  if (!result_or.ok()) {
    VLOG(1) << "TryEvaluate failed:" << result_or.status();
    return nullptr;
  }

  return result_or.ConsumeValueOrDie();
}

Status HloEvaluator::HandleParameter(HloInstruction* parameter) {
  VLOG(2) << "HandleParameter: " << parameter->ToString();
  const Literal* input_literal = arg_literals_[parameter->parameter_number()];
  VLOG(2) << "Parameter evaluated to: "
          << LiteralUtil::ToString(*input_literal);
  DCHECK(ShapeUtil::Equal(parameter->shape(), input_literal->shape()));

  evaluated_[parameter] = MakeUnique<Literal>(*input_literal);
  return Status::OK();
}

Status HloEvaluator::HandleConstant(HloInstruction* constant,
                                    const Literal& literal) {
  VLOG(2) << "HandleConstant: " << constant->ToString();
  return Status::OK();
}

Status HloEvaluator::HandleReshape(HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(
      evaluated_[reshape],
      LiteralUtil::Reshape(GetEvaluatedLiteralFor(reshape->operand(0)),
                           AsInt64Slice(reshape->shape().dimensions())));
  return Status::OK();
}

Status HloEvaluator::HandleTranspose(HloInstruction* transpose) {
  evaluated_[transpose] = LiteralUtil::Transpose(
      GetEvaluatedLiteralFor(transpose->operand(0)), transpose->dimensions());
  return Status::OK();
}

Status HloEvaluator::HandleConcatenate(
    HloInstruction* concatenate,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  // The result concatenate dimension is going to be the sum of all concatenate
  // dimensions of the operands taking part of the operation.
  const Shape& reference_shape = operands[0]->shape();
  CHECK(!ShapeUtil::IsTuple(reference_shape));
  const int64 rank = ShapeUtil::Rank(reference_shape);
  const int64 concat_dim = concatenate->dimensions()[0];
  CHECK_GE(concat_dim, 0);
  CHECK_LT(concat_dim, rank);

  DimensionVector concat_dimensions(reference_shape.dimensions().begin(),
                                    reference_shape.dimensions().end());

  for (int64 i = 1; i < operands.size(); ++i) {
    const Shape& operand_shape = operands[i]->shape();
    CHECK(!ShapeUtil::IsTuple(operand_shape));
    // Accumulate the concat dimension from all tensors taking part to the
    // operation.
    concat_dimensions[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  auto result_literal = LiteralUtil::CreateFromDimensions(
      reference_shape.element_type(), concat_dimensions);
  DimensionVector source_indices(rank, 0);
  DimensionVector dest_indices(concat_dimensions.size(), 0);

  for (auto operand : operands) {
    const Shape& operand_shape = operand->shape();
    TF_RETURN_IF_ERROR(LiteralUtil::Copy(
        GetEvaluatedLiteralFor(operand), source_indices, result_literal.get(),
        dest_indices, AsInt64Slice(operand_shape.dimensions())));
    dest_indices[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  evaluated_[concatenate] = std::move(result_literal);
  return Status::OK();
}

Status HloEvaluator::HandleIsFinite(HloInstruction* is_finite,
                                    HloInstruction* operand) {
  if (!ShapeUtil::ElementIsFloating(operand->shape())) {
    return InvalidArgument(
        "expected element type in shape to be float for IsFinite op, got: %s",
        PrimitiveType_Name(operand->shape().element_type()).c_str());
  }

  switch (operand->shape().element_type()) {
    case F16:
      return Unimplemented("unhandled primitive type: F16.");
    case F32: {
      auto result_or = ElementWiseUnaryOpImpl<bool, float>(
          is_finite,
          [](float elem_operand) { return std::isfinite(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
    case F64: {
      auto result_or = ElementWiseUnaryOpImpl<bool, double>(
          is_finite,
          [](double elem_operand) { return std::isfinite(elem_operand); },
          GetEvaluatedLiteralFor(operand));
      TF_ASSIGN_OR_RETURN(evaluated_[is_finite], std::move(result_or));
      break;
    }
    default:
      LOG(FATAL) << "unknown/unhandled primitive type.";
  }

  return Status::OK();
}

Status HloEvaluator::HandleCompare(HloInstruction* compare, HloOpcode opcode,
                                   HloInstruction* lhs, HloInstruction* rhs) {
  // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
  // removed.
  if (!(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
        ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()))) {
    return Unimplemented(
        "Implicit broadcasting is currently unsupported in HLO evaluator "
        "Shape Mismatch: %s vs %s vs %s",
        ShapeUtil::HumanString(compare->shape()).c_str(),
        ShapeUtil::HumanString(lhs->shape()).c_str(),
        ShapeUtil::HumanString(rhs->shape()).c_str());
  }

  TF_RET_CHECK(lhs->shape().element_type() == rhs->shape().element_type());

  const Literal& lhs_literal = GetEvaluatedLiteralFor(lhs);
  const Literal& rhs_literal = GetEvaluatedLiteralFor(rhs);

  // Note here we switch on the operand's type.
  switch (lhs->shape().element_type()) {
    case PRED: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<bool>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U8: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint8>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U16:
      return Unimplemented("unhandled primitive type: U16.");
    case U32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint32>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case U64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<uint64>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S8: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int8>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S16:
      return Unimplemented("unhandled primitive type: S16.");
    case S32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int32>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case S64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<int64>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case F16:
      return Unimplemented("unhandled primitive type: F16.");
    case F32: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<float>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    case F64: {
      TF_ASSIGN_OR_RETURN(
          evaluated_[compare],
          Compare<double>(compare->shape(), opcode, lhs_literal, rhs_literal));
    } break;
    default:
      LOG(FATAL) << "unknown primitive type.";
  }

  return Status::OK();
}

Status HloEvaluator::HandleSlice(HloInstruction* slice,
                                 HloInstruction* operand) {
  const Shape& shape = slice->shape();
  auto literal = LiteralUtil::CreateFromDimensions(
      shape.element_type(), AsInt64Slice(shape.dimensions()));

  DimensionVector dest_indices(slice->slice_starts().size(), 0);

  TF_RETURN_IF_ERROR(LiteralUtil::Copy(
      GetEvaluatedLiteralFor(operand), slice->slice_starts(), literal.get(),
      dest_indices, AsInt64Slice(shape.dimensions())));

  evaluated_[slice] = std::move(literal);
  return Status::OK();
}

}  // namespace xla
