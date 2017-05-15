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
  };

  template <
      typename NativeT,
      typename std::enable_if<std::is_signed<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs, HloInstruction* operand) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return Status::OK();
  };

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

  Status HandleIsFinite(HloInstruction* is_finite,
                        HloInstruction* operand) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[is_finite],
                        ElementWiseUnaryOp(is_finite, [](ReturnT elem_operand) {
                          return std::isfinite(elem_operand);
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

  Status HandleCompare(HloInstruction* compare, HloOpcode opcode,
                       HloInstruction* lhs, HloInstruction* rhs) override {
    std::function<bool(ReturnT, ReturnT)> compare_op;
    switch (opcode) {
      case HloOpcode::kEq:
        compare_op = [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el == rhs_el;
        };
        break;
      case HloOpcode::kNe:
        compare_op = [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el != rhs_el;
        };
        break;
      case HloOpcode::kGe:
        compare_op = [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el >= rhs_el;
        };
        break;
      case HloOpcode::kGt:
        compare_op = [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el > rhs_el;
        };
        break;
      case HloOpcode::kLe:
        compare_op = [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el <= rhs_el;
        };
        break;
      case HloOpcode::kLt:
        compare_op = [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el < rhs_el;
        };
        break;
      default:
        LOG(FATAL) << "unhandled HLO opcode for conversion to Comparison: "
                   << HloOpcodeString(opcode);
    }

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast is
    // removed.
    if (!(ShapeUtil::SameDimensions(compare->shape(), rhs->shape()) &&
          ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()))) {
      return Unimplemented(
          "Compare operation with mismatched dimensions, likely due to "
          "broadcasting is unsupported.");
    }

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    auto result = LiteralUtil::CreateFromShape(compare->shape());
    std::vector<int64> multi_index(ShapeUtil::Rank(result->shape()), 0);
    do {
      LiteralUtil::Set<bool>(
          result.get(), multi_index,
          compare_op(LiteralUtil::Get<ReturnT>(lhs_literal, multi_index),
                     LiteralUtil::Get<ReturnT>(rhs_literal, multi_index)));
    } while (IndexUtil::BumpIndices(result->shape(), &multi_index));

    parent_->evaluated_[compare] = std::move(result);

    return Status::OK();
  };

  Status HandleMaximum(HloInstruction* maximum, HloInstruction* lhs,
                       HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[maximum],
        ElementWiseBinaryOp(maximum, [](ReturnT lhs, ReturnT rhs) {
          return std::max(lhs, rhs);
        }));
    return Status::OK();
  };

  Status HandleMinimum(HloInstruction* minimum, HloInstruction* lhs,
                       HloInstruction* rhs) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[minimum],
        ElementWiseBinaryOp(minimum, [](ReturnT lhs_el, ReturnT rhs_el) {
          return std::min(lhs_el, rhs_el);
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
          return std::remainder(lhs_el, rhs_el);
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
          return std::max(low, std::min(value, high));
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

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);

    auto result = LiteralUtil::CreateFromShape(shape);

    std::vector<int64> multi_index(ShapeUtil::Rank(result->shape()), 0);
    do {
      LiteralUtil::Set<ReturnT>(
          result.get(), multi_index,
          unary_op(LiteralUtil::Get<ReturnT>(operand_literal, multi_index)));
    } while (IndexUtil::BumpIndices(result->shape(), &multi_index));

    return std::move(result);
  };

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
    std::vector<int64> multi_index(ShapeUtil::Rank(result->shape()), 0);
    do {
      LiteralUtil::Set<ReturnT>(
          result.get(), multi_index,
          binary_op(LiteralUtil::Get<ReturnT>(lhs_literal, multi_index),
                    LiteralUtil::Get<ReturnT>(rhs_literal, multi_index)));
    } while (IndexUtil::BumpIndices(result->shape(), &multi_index));

    return std::move(result);
  };

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
    std::vector<int64> multi_index(ShapeUtil::Rank(result->shape()), 0);
    do {
      LiteralUtil::Set<ReturnT>(
          result.get(), multi_index,
          ternary_op(LiteralUtil::Get<LhsType>(lhs_literal, multi_index),
                     LiteralUtil::Get<RhsType>(rhs_literal, multi_index),
                     LiteralUtil::Get<EhsType>(ehs_literal, multi_index)));
    } while (IndexUtil::BumpIndices(result->shape(), &multi_index));

    return std::move(result);
  };

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
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloComputation* computation,
    tensorflow::gtl::ArraySlice<const Literal*> args) {
  arg_literals_ = args;
  evaluated_.clear();

  TF_RETURN_IF_ERROR(computation->Accept(this));
  return std::move(FindOrDie(evaluated_, computation->root_instruction()));
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction,
    tensorflow::gtl::ArraySlice<const Literal*> operands) {
  DCHECK(hlo_query::AllOperandsAreParametersOrConstants(*instruction));
  Shape shape = instruction->shape();
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape));

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
    } else if (operand->opcode() == HloOpcode::kConstant) {
      evaluated_[operand] = MakeUnique<Literal>(operand->literal());
    }
  }

  TF_RETURN_IF_ERROR(instruction->Visit(this));
  return std::move(FindOrDie(evaluated_, instruction));
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
  DCHECK(ShapeUtil::Equal(constant->shape(), literal.shape()));

  evaluated_[constant] = MakeUnique<Literal>(literal);
  return Status::OK();
}

}  // namespace xla
