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
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

template <typename T>
struct is_complex_t : public std::false_type {};

template <>
struct is_complex_t<complex64> : public std::true_type {};

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

  auto result = Literal::CreateFromShape(shape);
  TF_RETURN_IF_ERROR(result->Populate<bool>(
      [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return compare_op(lhs_literal.Get<OperandT>(multi_index),
                          rhs_literal.Get<OperandT>(multi_index));
      }));

  return std::move(result);
}

template <>
StatusOr<std::unique_ptr<Literal>> Compare<complex64>(
    const Shape& shape, HloOpcode opcode, const Literal& lhs_literal,
    const Literal& rhs_literal) {
  std::function<bool(complex64, complex64)> compare_op;
  switch (opcode) {
    case HloOpcode::kEq:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el == rhs_el;
      };
      break;
    case HloOpcode::kNe:
      compare_op = [](complex64 lhs_el, complex64 rhs_el) {
        return lhs_el != rhs_el;
      };
      break;
    default:
      LOG(FATAL) << "unhandled HLO opcode for conversion to Comparison: "
                 << HloOpcodeString(opcode);
  }

  auto result = Literal::CreateFromShape(shape);
  TF_RETURN_IF_ERROR(result->Populate<bool>(
      [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return compare_op(lhs_literal.Get<complex64>(multi_index),
                          rhs_literal.Get<complex64>(multi_index));
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

  auto result = Literal::CreateFromShape(shape);

  TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
      [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
        return unary_op(operand_literal.Get<NativeT>(multi_index));
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
  }

  // TODO(b/35950897): many of the stl functions used in the handlers are not
  // overloaded for every XLA primitive types.

  template <typename NativeT,
            typename std::enable_if<std::is_unsigned<NativeT>::value>::type* =
                nullptr>
  Status HandleAbs(HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return elem_operand;
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_signed<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return Status::OK();
  }

  Status HandleAbs(HloInstruction* abs) override {
    return HandleAbs<ReturnT>(abs);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRound(HloInstruction* round) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[round],
                        ElementWiseUnaryOp(round, [](ReturnT elem_operand) {
                          return std::round(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRound(HloInstruction* round) {
    return InvalidArgument("Unsupported type for Round");
  }

  Status HandleRound(HloInstruction* round) override {
    return HandleRound<ReturnT>(round);
  }

  Status HandleBroadcast(HloInstruction* broadcast) override {
    parent_->evaluated_[broadcast] =
        Literal::CreateFromShape(broadcast->shape());
    auto output = parent_->evaluated_[broadcast].get();
    auto operand_to_broadcast =
        parent_->GetEvaluatedLiteralFor(broadcast->operand(0));
    std::vector<int64> broadcast_indices(
        ShapeUtil::Rank(broadcast->operand(0)->shape()), 0);

    TF_RET_CHECK(broadcast->dimensions().size() ==
                 ShapeUtil::Rank(operand_to_broadcast.shape()))
        << "broadcast dimensions is of size: " << broadcast->dimensions().size()
        << " and rank of operand_to_broadcast is: "
        << ShapeUtil::Rank(operand_to_broadcast.shape());
    // Checks that operand's dimensions are the same as the broadcast's
    // dimensions along the dimensions to be broadcasted.
    for (int64 i = 0; i < broadcast->dimensions().size(); ++i) {
      TF_RET_CHECK(broadcast->shape().dimensions(broadcast->dimensions(i)) ==
                   operand_to_broadcast.shape().dimensions(i));
    }

    return output->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          for (int64 i = 0; i < broadcast->dimensions().size(); ++i) {
            broadcast_indices[i] = multi_index[broadcast->dimensions(i)];
          }
          return operand_to_broadcast.Get<ReturnT>(broadcast_indices);
        });
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCeil(HloInstruction* ceil) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[ceil],
                        ElementWiseUnaryOp(ceil, [](ReturnT elem_operand) {
                          return std::ceil(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCeil(HloInstruction* ceil) {
    return InvalidArgument("Unsupported type for Ceil");
  }

  Status HandleCeil(HloInstruction* ceil) override {
    return HandleCeil<ReturnT>(ceil);
  }

  Status HandleConvert(HloInstruction* convert) override {
    const HloInstruction* operand = convert->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), convert->shape()));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> result,
                        parent_->GetEvaluatedLiteralFor(operand).Convert(
                            convert->shape().element_type()));

    if (LayoutUtil::LayoutsInShapesEqual(result->shape(), convert->shape())) {
      parent_->evaluated_[convert] = std::move(result);
    } else {
      parent_->evaluated_[convert] =
          result->Relayout(convert->shape().layout());
    }
    return Status::OK();
  }

  Status HandleExp(HloInstruction* exp) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[exp],
                        ElementWiseUnaryOp(exp, [](ReturnT elem_operand) {
                          return std::exp(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleFloor(HloInstruction* floor) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[floor],
                        ElementWiseUnaryOp(floor, [](ReturnT elem_operand) {
                          return std::floor(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleFloor(HloInstruction* floor) {
    return InvalidArgument("Unsupported type for Floor");
  }

  Status HandleFloor(HloInstruction* floor) override {
    return HandleFloor<ReturnT>(floor);
  }

  Status HandleLog(HloInstruction* log) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[log],
                        ElementWiseUnaryOp(log, [](ReturnT elem_operand) {
                          return std::log(elem_operand);
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ReturnT elem_operand) {
                          return ~elem_operand;
                        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ReturnT elem_operand) {
                          return !elem_operand;
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ReturnT elem_operand) {
                          return !elem_operand;
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    return InvalidArgument("Unsupported type for Not");
  }

  Status HandleNot(HloInstruction* not_) override {
    return HandleNot<ReturnT>(not_);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_signed<NativeT>::value &&
                !std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    using type = typename std::make_unsigned<NativeT>::type;
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[negate],
                        ElementWiseUnaryOp(negate, [](ReturnT elem_operand) {
                          return NativeT(-type(elem_operand));
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                !std::is_signed<NativeT>::value ||
                std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[negate],
                        ElementWiseUnaryOp(negate, [](ReturnT elem_operand) {
                          return -elem_operand;
                        }));
    return Status::OK();
  }

  Status HandleNegate(HloInstruction* negate) override {
    return HandleNegate<ReturnT>(negate);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleSign(HloInstruction* sign) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ReturnT elem_operand) {
                          return (ReturnT(0) < elem_operand) -
                                 (elem_operand < ReturnT(0));
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleSign(HloInstruction* sign) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ReturnT elem_operand) {
                          auto abs_val = std::abs(elem_operand);
                          return 0 == abs_val ? ReturnT(0)
                                              : elem_operand / abs_val;
                        }));
    return Status::OK();
  }

  Status HandleSign(HloInstruction* sign) override {
    return HandleSign<ReturnT>(sign);
  }

  Status HandleTanh(HloInstruction* tanh) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[tanh],
                        ElementWiseUnaryOp(tanh, [](ReturnT elem_operand) {
                          return std::tanh(elem_operand);
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_signed<NativeT>::value &&
                !std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleMultiply(HloInstruction* multiply) {
    using type = typename std::make_unsigned<NativeT>::type;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(multiply, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return NativeT(type(lhs_elem) * type(rhs_elem));
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_unsigned<NativeT>::value ||
                              std::is_floating_point<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMultiply(HloInstruction* multiply) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(multiply, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem * rhs_elem;
        }));
    return Status::OK();
  }

  Status HandleMultiply(HloInstruction* multiply) override {
    return HandleMultiply<ReturnT>(multiply);
  }

  Status HandleSubtract(HloInstruction* subtract) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(subtract, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem - rhs_elem;
        }));
    return Status::OK();
  }

  Status HandleAdd(HloInstruction* add) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[add],
        ElementWiseBinaryOp(add, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem + rhs_elem;
        }));
    return Status::OK();
  }

  Status HandleDivide(HloInstruction* divide) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[divide],
        ElementWiseBinaryOp(divide, [](ReturnT lhs_elem, ReturnT rhs_elem) {
          return lhs_elem / rhs_elem;
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMaximum(HloInstruction* maximum) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[maximum],
        ElementWiseBinaryOp(maximum, [](ReturnT lhs, ReturnT rhs) {
          return std::fmax(lhs, rhs);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMaximum(HloInstruction* maximum) {
    return InvalidArgument("Unsupported type for Maximum");
  }

  Status HandleMaximum(HloInstruction* maximum) override {
    return HandleMaximum<ReturnT>(maximum);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMinimum(HloInstruction* minimum) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[minimum],
        ElementWiseBinaryOp(minimum, [](ReturnT lhs_el, ReturnT rhs_el) {
          return std::fmin(lhs_el, rhs_el);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleMinimum(HloInstruction* minimum) {
    return InvalidArgument("Unsupported type for Minimum");
  }

  Status HandleMinimum(HloInstruction* minimum) override {
    return HandleMinimum<ReturnT>(minimum);
  }

  Status HandlePower(HloInstruction* power) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[power],
        ElementWiseBinaryOp(power, [](ReturnT lhs_el, ReturnT rhs_el) {
          return std::pow(lhs_el, rhs_el);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[remainder],
        ElementWiseBinaryOp(remainder, [](ReturnT lhs_el, ReturnT rhs_el) {
          return std::fmod(lhs_el, rhs_el);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    return InvalidArgument("Unsupported type for Remainder");
  }

  Status HandleRemainder(HloInstruction* remainder) override {
    return HandleRemainder<ReturnT>(remainder);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleAnd(HloInstruction* and_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[and_],
        ElementWiseBinaryOp(and_, [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el & rhs_el;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleAnd(HloInstruction* and_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[and_],
        ElementWiseBinaryOp(and_, [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el && rhs_el;
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAnd(HloInstruction* and_) {
    return InvalidArgument("Unsupported type for And");
  }

  Status HandleAnd(HloInstruction* and_) override {
    return HandleAnd<ReturnT>(and_);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleOr(HloInstruction* or_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[or_],
        ElementWiseBinaryOp(or_, [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el | rhs_el;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleOr(HloInstruction* or_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[or_],
        ElementWiseBinaryOp(or_, [](ReturnT lhs_el, ReturnT rhs_el) {
          return lhs_el || rhs_el;
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleOr(HloInstruction* or_) {
    return InvalidArgument("Unsupported type for Or");
  }

  Status HandleOr(HloInstruction* or_) override {
    return HandleOr<ReturnT>(or_);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleShiftLeft(HloInstruction* shl) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[shl],
        ElementWiseBinaryOp(shl, [](NativeT lhs_elem, NativeT rhs_elem) {
          return lhs_elem << rhs_elem;
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<!std::is_integral<NativeT>::value ||
                                    std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleShiftLeft(HloInstruction*) {
    return InvalidArgument("Unsupported type for ShiftLeft");
  }

  Status HandleShiftLeft(HloInstruction* shl) override {
    return HandleShiftLeft<ReturnT>(shl);
  }
  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleShiftRightArithmetic(HloInstruction* shr) {
    typedef typename std::make_signed<NativeT>::type SignedT;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[shr],
        ElementWiseBinaryOp(shr, [](NativeT lhs_elem, NativeT rhs_elem) {
          return static_cast<NativeT>(static_cast<SignedT>(lhs_elem) >>
                                      rhs_elem);
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<!std::is_integral<NativeT>::value ||
                                    std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleShiftRightArithmetic(HloInstruction*) {
    return InvalidArgument("Unsupported type for ShiftRightArithmetic");
  }

  Status HandleShiftRightArithmetic(HloInstruction* shra) override {
    return HandleShiftRightArithmetic<ReturnT>(shra);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleShiftRightLogical(HloInstruction* shr) {
    typedef typename std::make_unsigned<NativeT>::type UnsignedT;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[shr],
        ElementWiseBinaryOp(shr, [](NativeT lhs_elem, NativeT rhs_elem) {
          return static_cast<NativeT>(static_cast<UnsignedT>(lhs_elem) >>
                                      rhs_elem);
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<!std::is_integral<NativeT>::value ||
                                    std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleShiftRightLogical(HloInstruction*) {
    return InvalidArgument("Unsupported type for ShiftRightLogical");
  }

  Status HandleShiftRightLogical(HloInstruction* shrl) override {
    return HandleShiftRightLogical<ReturnT>(shrl);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleClamp(HloInstruction* clamp) {
    std::function<ReturnT(ReturnT, ReturnT, ReturnT)> clamp_op =
        [](ReturnT low, ReturnT value, ReturnT high) {
          return std::fmax(low, std::fmin(value, high));
        };
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[clamp],
                        ElementWiseTernaryOp(clamp, std::move(clamp_op)));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleClamp(HloInstruction*) {
    return InvalidArgument("Unsupported type for Clamp");
  }

  Status HandleClamp(HloInstruction* clamp) override {
    return HandleClamp<ReturnT>(clamp);
  }

  Status HandleSelect(HloInstruction* select) override {
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
  }

  Status HandleReverse(HloInstruction* reverse) override {
    const auto result_shape = reverse->shape();
    const auto reverse_dimensions = reverse->dimensions();

    auto operand = reverse->operand(0);
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferReverseShape(operand->shape(),
                                                          reverse_dimensions));

    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    auto operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    auto result = Literal::CreateFromShape(result_shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> out_index) {
          std::vector<int64> from_index(out_index.begin(), out_index.end());
          for (const int64 dim : reverse_dimensions) {
            from_index[dim] = result_shape.dimensions(dim) - 1 - out_index[dim];
          }
          return operand_literal.Get<ReturnT>(from_index);
        }));

    parent_->evaluated_[reverse] = std::move(result);
    return Status::OK();
  }

  Status HandleConvolution(HloInstruction* conv) override {
    auto lhs = conv->operand(0);
    auto rhs = conv->operand(1);
    const auto& window = conv->window();
    const Shape& result_shape = conv->shape();
    const Shape& lhs_shape = lhs->shape();
    const Shape& rhs_shape = rhs->shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(lhs_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(rhs_shape));
    CHECK(ShapeUtil::IsArray(lhs_shape));
    CHECK(ShapeUtil::IsArray(rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, result_shape));

    const auto& dnums = conv->convolution_dimension_numbers();
    const int64 num_spatial_dims = dnums.output_spatial_dimensions_size();
    CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
    CHECK_GE(num_spatial_dims, 0);
    CHECK_EQ(window.dimensions_size(), num_spatial_dims);

    const auto lhs_rank = ShapeUtil::Rank(lhs_shape);
    const auto rhs_rank = ShapeUtil::Rank(rhs_shape);

    CHECK_EQ(num_spatial_dims + 2, lhs_rank);
    CHECK_EQ(num_spatial_dims + 2, rhs_rank);

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferConvolveShape(lhs_shape, rhs_shape,
                                                           window, dnums));
    CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    // Dimension number applicable for input (lhs).
    const int64 input_batch_dim = dnums.input_batch_dimension();
    const int64 input_z_dim = dnums.input_feature_dimension();
    // Dimension number applicable for kernel (rhs).
    const int64 kernel_input_z_dim = dnums.kernel_input_feature_dimension();
    const int64 kernel_output_z_dim = dnums.kernel_output_feature_dimension();
    // Dimension number applicable for output.
    const int64 output_batch_dim = dnums.output_batch_dimension();
    const int64 output_z_dim = dnums.output_feature_dimension();

    const int64 z_size = ShapeUtil::GetDimension(lhs_shape, input_z_dim);

    std::vector<int64> window_dimension_sizes;
    for (auto i : dnums.kernel_spatial_dimensions()) {
      window_dimension_sizes.push_back(ShapeUtil::GetDimension(rhs_shape, i));
    }

    const Shape& window_shape =
        ShapeUtil::MakeShape(rhs_shape.element_type(), window_dimension_sizes);

    DimensionVector lhs_index(lhs_rank);
    DimensionVector rhs_index(rhs_rank);
    DimensionVector rhs_spatial_index(dnums.kernel_spatial_dimensions_size());

    auto func = [&](tensorflow::gtl::ArraySlice<int64> out_index) {
      ReturnT result_val = static_cast<ReturnT>(0);

      std::fill(lhs_index.begin(), lhs_index.end(), 0);
      std::fill(rhs_index.begin(), rhs_index.end(), 0);
      std::fill(rhs_spatial_index.begin(), rhs_spatial_index.end(), 0);

      lhs_index[input_batch_dim] = out_index[output_batch_dim];
      rhs_index[kernel_output_z_dim] = out_index[output_z_dim];

      // Convolve input feature with kernel.
      do {
        for (int64 iz = 0; iz < z_size; ++iz) {
          lhs_index[input_z_dim] = iz;
          rhs_index[kernel_input_z_dim] = iz;

          // Find corresponding spatial dimension index for input (lhs).
          for (int64 ki = 0; ki < rhs_spatial_index.size(); ++ki) {
            // Spatial dimension number for input (lhs) and output.
            const int64 input_spatial_dim = dnums.input_spatial_dimensions(ki);
            const int64 output_spatial_dim =
                dnums.output_spatial_dimensions(ki);

            // Calculate lhs (input) index without taking base dilation into
            // account.
            const auto& window_dim = window.dimensions(ki);
            const int64 undilated_index =
                out_index[output_spatial_dim] * window_dim.stride() -
                window_dim.padding_low() +
                rhs_spatial_index[ki] * window_dim.window_dilation();
            // Skip if the lhs (input) index is to be dilated.
            if (undilated_index % window_dim.base_dilation() != 0) {
              goto cnt;
            }

            // Calculate the actual lhs (input) index after dilation.
            lhs_index[input_spatial_dim] =
                undilated_index / window_dim.base_dilation();

            // Skip if input index is not in bound.
            if (!(lhs_index[input_spatial_dim] >= 0 &&
                  lhs_index[input_spatial_dim] <
                      lhs_shape.dimensions(input_spatial_dim))) {
              goto cnt;
            }

            rhs_index[dnums.kernel_spatial_dimensions(ki)] =
                window_dim.window_reversal()
                    ? ((window_dim.size() - 1) - rhs_spatial_index[ki])
                    : rhs_spatial_index[ki];
          }

          result_val += lhs_literal.Get<ReturnT>(lhs_index) *
                        rhs_literal.Get<ReturnT>(rhs_index);
        }
      cnt : {}
      } while (IndexUtil::BumpIndices(window_shape, &rhs_spatial_index));

      return result_val;
    };

    auto result = Literal::CreateFromShape(result_shape);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(func));

    parent_->evaluated_[conv] = std::move(result);
    return Status::OK();
  }

  Status HandleDot(HloInstruction* dot) override {
    auto lhs = dot->operand(0);
    auto rhs = dot->operand(1);
    CHECK(ShapeUtil::IsArray(dot->shape()));
    CHECK(ShapeUtil::IsArray(lhs->shape()));
    CHECK(ShapeUtil::IsArray(rhs->shape()));

    // Dot only supports operands of rank 1 and 2.
    const auto dot_rank = ShapeUtil::Rank(dot->shape());
    const auto lhs_rank = ShapeUtil::Rank(lhs->shape());
    const auto rhs_rank = ShapeUtil::Rank(rhs->shape());
    CHECK(lhs_rank > 0 && lhs_rank <= 2);
    CHECK(rhs_rank > 0 && rhs_rank <= 2);
    CHECK_EQ(dot_rank, lhs_rank + rhs_rank - 2);

    CHECK(ShapeUtil::SameElementType(lhs->shape(), rhs->shape()));
    CHECK(ShapeUtil::SameElementType(lhs->shape(), dot->shape()));

    // Check contracted dimensions are the same.
    //
    // Determine the index of the contracted dimensions for input tensors.
    // dimensions -1 of lhs and dimension 0 of rhs are contracted.
    const int64 lhs_contracted_dimension =
        ShapeUtil::GetDimensionNumber(lhs->shape(), -1);
    const int64 rhs_contracted_dimension = 0;
    CHECK_EQ(lhs->shape().dimensions(lhs_contracted_dimension),
             rhs->shape().dimensions(rhs_contracted_dimension))
        << "lhs contracted dimension: "
        << lhs->shape().dimensions(lhs_contracted_dimension)
        << " rhs contracted dimension: "
        << rhs->shape().dimensions(rhs_contracted_dimension);
    const int64 contracted_dimension_size =
        lhs->shape().dimensions(lhs_contracted_dimension);

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    auto result = Literal::CreateFromShape(dot->shape());
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          ReturnT result_val = static_cast<ReturnT>(0);

          std::vector<int64> lhs_index(lhs_rank, 0);
          std::vector<int64> rhs_index(rhs_rank, 0);
          // Set index for non-contracted dimension for lhs and rhs.
          if (lhs_rank > 1) {
            lhs_index[0] = multi_index[0];
          }
          if (rhs_rank > 1) {
            rhs_index[1] = multi_index[multi_index.size() - 1];
          }

          // Accumulates resulting product along the contracted dimension.
          for (int64 i = 0; i < contracted_dimension_size; ++i) {
            lhs_index[lhs_contracted_dimension] = i;
            rhs_index[rhs_contracted_dimension] = i;

            result_val += lhs_literal.Get<ReturnT>(lhs_index) *
                          rhs_literal.Get<ReturnT>(rhs_index);
          }

          return result_val;
        }));

    parent_->evaluated_[dot] = std::move(result);
    return Status::OK();
  }

  Status HandlePad(HloInstruction* pad) override {
    CHECK(!ShapeUtil::IsTuple(pad->operand(0)->shape()));
    // Padding value must be scalar.
    CHECK(ShapeUtil::IsScalar(pad->operand(1)->shape()));
    CHECK_EQ(ShapeUtil::Rank(pad->operand(0)->shape()),
             pad->padding_config().dimensions_size());

    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferPadShape(
                            /*operand_shape=*/pad->operand(0)->shape(),
                            /*padding_value_shape=*/pad->operand(1)->shape(),
                            /*padding_config=*/pad->padding_config()));
    CHECK(ShapeUtil::Compatible(pad->shape(), inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(pad->shape())
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    // Create new HLO of padded shape with padding value.
    ReturnT scalar =
        parent_->GetEvaluatedLiteralFor(pad->operand(1)).Get<ReturnT>({});
    auto result = Literal::CreateFromShape(pad->shape());
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&scalar](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return scalar;
        }));

    auto evaluated_operand = parent_->GetEvaluatedLiteralFor(pad->operand(0));

    std::vector<int64> input_index(ShapeUtil::Rank(evaluated_operand.shape()),
                                   0);
    std::vector<int64> target_index(ShapeUtil::Rank(result->shape()), 0);

    // Loop through each element of the operand, assign them to the
    // corresponding index of the resulting padded literal.
    const PaddingConfig& pad_config = pad->padding_config();

    auto func = [&](const std::vector<int64>& input_index) {
      for (auto i = 0; i < input_index.size(); ++i) {
        // Interior padding occurs logically before edge padding, so in the case
        // of negative edge padding elements are removed from the
        // interior-padded operand.
        target_index[i] =
            pad_config.dimensions(i).edge_padding_low() +
            input_index[i] * (pad_config.dimensions(i).interior_padding() + 1);

        // Account for negative low and high padding: skip assignment if the
        // any target index is out of range.
        if (!(target_index[i] >= 0 &&
              target_index[i] < pad->shape().dimensions(i))) {
          return true;
        }
      }
      result->Set<ReturnT>(target_index,
                           evaluated_operand.Get<ReturnT>(input_index));
      return true;
    };

    std::vector<int64> zero_base(evaluated_operand.shape().dimensions_size(),
                                 0);
    std::vector<int64> step(evaluated_operand.shape().dimensions_size(), 1);

    ShapeUtil::ForEachIndex(
        evaluated_operand.shape(), zero_base,
        AsInt64Slice(evaluated_operand.shape().dimensions()), step, func);

    parent_->evaluated_[pad] = std::move(result);
    return Status::OK();
  }

  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override {
    auto operand = dynamic_slice->operand(0);
    auto start_indices = dynamic_slice->operand(1);
    auto result_shape = dynamic_slice->shape();
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferDynamicSliceShape(
                            operand->shape(), start_indices->shape(),
                            dynamic_slice->dynamic_slice_sizes()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    TF_RET_CHECK(
        primitive_util::IsIntegralType(start_indices->shape().element_type()));

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& start_indices_literal =
        parent_->GetEvaluatedLiteralFor(start_indices);

    switch (start_indices->shape().element_type()) {
      case S32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<int32>(operand_literal, start_indices_literal,
                                result_shape));
      } break;
      case S64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<int64>(operand_literal, start_indices_literal,
                                result_shape));
      } break;
      case U32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<uint32>(operand_literal, start_indices_literal,
                                 result_shape));
      } break;
      case U64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<uint64>(operand_literal, start_indices_literal,
                                 result_shape));
      } break;
      default:
        LOG(FATAL) << "HandleDynamicSlice: unhandled primitive type for "
                      "start_indices: "
                   << PrimitiveType_Name(start_indices->shape().element_type());
    }

    return Status::OK();
  }

  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override {
    auto operand = dynamic_update_slice->operand(0);
    auto update = dynamic_update_slice->operand(1);
    auto start_indices = dynamic_update_slice->operand(2);
    auto result_shape = dynamic_update_slice->shape();
    TF_ASSIGN_OR_RETURN(
        auto inferred_return_shape,
        ShapeInference::InferDynamicUpdateSliceShape(
            operand->shape(), update->shape(), start_indices->shape()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    TF_RET_CHECK(
        primitive_util::IsIntegralType(start_indices->shape().element_type()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, operand->shape()));

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& update_literal = parent_->GetEvaluatedLiteralFor(update);
    const Literal& start_indices_literal =
        parent_->GetEvaluatedLiteralFor(start_indices);

    switch (start_indices->shape().element_type()) {
      case S32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<int32>(operand_literal, update_literal,
                                      start_indices_literal));
      } break;
      case S64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<int64>(operand_literal, update_literal,
                                      start_indices_literal));
      } break;
      case U32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<uint32>(operand_literal, update_literal,
                                       start_indices_literal));
      } break;
      case U64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<uint64>(operand_literal, update_literal,
                                       start_indices_literal));
      } break;
      default:
        LOG(FATAL) << "HandleDynamicUpdateSlice: unhandled primitive type for "
                      "start_indices: "
                   << PrimitiveType_Name(start_indices->shape().element_type());
    }

    return Status::OK();
  }

  Status HandleReduce(HloInstruction* reduce) override {
    auto arg = reduce->operand(0);
    auto init_value = reduce->operand(1);
    tensorflow::gtl::ArraySlice<int64> dimensions(reduce->dimensions());
    HloComputation* function = reduce->to_apply();
    TF_RET_CHECK(ShapeUtil::Rank(reduce->shape()) ==
                 ShapeUtil::Rank(arg->shape()) - dimensions.size());
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferReduceShape(
                            /*arg=*/arg->shape(),
                            /*init_value=*/init_value->shape(),
                            /*dimensions_to_reduce=*/dimensions,
                            /*to_apply=*/function->ComputeProgramShape()));
    TF_RET_CHECK(ShapeUtil::Compatible(reduce->shape(), inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(reduce->shape())
        << "but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const Literal& arg_literal = parent_->GetEvaluatedLiteralFor(arg);
    VLOG(3) << "HandleReduce arg_literal: " << arg_literal.ToString();
    const Literal& init_literal = parent_->GetEvaluatedLiteralFor(init_value);
    VLOG(3) << "HandleReduce init_literal: " << init_literal.ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    auto result = Literal::CreateFromShape(reduce->shape());

    const auto arg_dimensions = AsInt64Slice(arg_literal.shape().dimensions());
    std::vector<int64> arg_dim_steps(arg_dimensions.size());
    std::vector<int64> arg_dim_counts(arg_dimensions.size());
    for (const int64 dim : dimensions) {
      arg_dim_steps[dim] = 1;
      arg_dim_counts[dim] = arg_dimensions[dim];
    }

    // Create mapping from result index to arg index.
    const int64 result_rank = ShapeUtil::Rank(result->shape());
    int64 result_dim = 0;
    std::vector<int64> result_to_arg_index(result_rank);
    for (int64 i = 0; i < arg_dimensions.size(); ++i) {
      if (arg_dim_steps[i] == 0) {
        result_to_arg_index[result_dim] = i;
        ++result_dim;
      }
    }

    // For each resulting dimension, calculate and assign computed value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          ReturnT result_val = init_scalar;

          std::vector<int64> base(arg_dimensions.size());
          for (int64 i = 0; i < multi_index.size(); ++i) {
            base[result_to_arg_index[i]] = multi_index[i];
          }

          auto func = [&](const std::vector<int64>& input_index) {
            auto curr_val = arg_literal.Get<ReturnT>(input_index);

            // Evaluate computation with specified literal operands.
            auto curr_val_literal = Literal::CreateR0<ReturnT>(curr_val);
            auto result_val_literal = Literal::CreateR0<ReturnT>(result_val);
            std::vector<const Literal*> args = {curr_val_literal.get(),
                                                result_val_literal.get()};

            // We need a new visitor for each evaluation, so that the same
            // computation can be visited more than once (with different
            // inputs).
            HloEvaluator embedded_evaluator;
            std::unique_ptr<Literal> computed_result =
                embedded_evaluator.Evaluate(*function, args)
                    .ConsumeValueOrDie();

            // Assign computed result to result_val.
            result_val = computed_result->Get<ReturnT>({});

            return true;
          };

          ShapeUtil::ForEachIndex(arg_literal.shape(), base, arg_dim_counts,
                                  arg_dim_steps, func);

          return result_val;
        }));

    parent_->evaluated_[reduce] = std::move(result);
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* reduce_window) override {
    auto operand = reduce_window->operand(0);
    const Window& window = reduce_window->window();
    HloComputation* function = reduce_window->to_apply();
    TF_ASSIGN_OR_RETURN(
        auto inferred_return_shape,
        ShapeInference::InferReduceWindowShape(
            /*operand_shape=*/reduce_window->operand(0)->shape(),
            /*init_value=*/reduce_window->operand(1)->shape(), window,
            /*to_apply_shape=*/function->ComputeProgramShape()));
    TF_RET_CHECK(
        ShapeUtil::Compatible(reduce_window->shape(), inferred_return_shape))
        << "return shape is set to: "
        << ShapeUtil::HumanStringWithLayout(reduce_window->shape())
        << "but is inferred to be: "
        << ShapeUtil::HumanStringWithLayout(inferred_return_shape);

    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(reduce_window->operand(0));
    VLOG(3) << "HandleReduceWindow arg_literal: " << operand_literal.ToString();
    const Literal& init_literal =
        parent_->GetEvaluatedLiteralFor(reduce_window->operand(1));
    VLOG(3) << "HandleReduceWindow init_literal: " << init_literal.ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    auto result = Literal::CreateFromShape(reduce_window->shape());

    // Creates a Shape object from window, for iteration below.
    std::vector<int64> window_dimension_sizes;
    for (const auto& window_dimension : window.dimensions()) {
      window_dimension_sizes.push_back(window_dimension.size());
    }
    const Shape window_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(), window_dimension_sizes);

    DimensionVector window_index(window.dimensions_size());
    DimensionVector operand_index(ShapeUtil::Rank(operand_literal.shape()));

    // For each resulting dimension, calculate and assign computed value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> output_index) {
          ReturnT result_val = init_scalar;

          std::fill(window_index.begin(), window_index.end(), 0);
          std::fill(operand_index.begin(), operand_index.end(), 0);

          do {
            // Set curr_val to 0 if out of bound (padded).
            ReturnT curr_val = static_cast<ReturnT>(0);
            bool out_of_bound = false;
            for (int i = 0; i < operand_index.size(); ++i) {
              operand_index[i] =
                  output_index[i] * window.dimensions(i).stride() +
                  window_index[i] - window.dimensions(i).padding_low();
              if (operand_index[i] < 0 ||
                  operand_index[i] >= operand_literal.shape().dimensions(i)) {
                out_of_bound = true;
                break;
              }
            }
            if (!out_of_bound) {
              curr_val = operand_literal.Get<ReturnT>(operand_index);
            }
            // Evaluate computation with specified literal operands.
            const auto curr_val_literal = Literal::CreateR0<ReturnT>(curr_val);
            const auto result_val_literal =
                Literal::CreateR0<ReturnT>(result_val);
            const std::vector<const Literal*> args = {curr_val_literal.get(),
                                                      result_val_literal.get()};
            // We need a new visitor for each evaluation, so that the same
            // computation can be visited more than once (with different
            // inputs).
            HloEvaluator embedded_evaluator;
            std::unique_ptr<Literal> computed_result =
                embedded_evaluator.Evaluate(*function, args)
                    .ConsumeValueOrDie();

            result_val = computed_result->Get<ReturnT>({});
          } while (IndexUtil::BumpIndices(window_shape, &window_index));

          return result_val;
        }));

    parent_->evaluated_[reduce_window] = std::move(result);
    return Status::OK();
  }

  Status HandleSlice(HloInstruction* slice) override {
    auto operand = slice->operand(0);
    const Shape& shape = slice->shape();
    TF_ASSIGN_OR_RETURN(auto inferred_return_shape,
                        ShapeInference::InferSliceShape(
                            operand->shape(), slice->slice_starts(),
                            slice->slice_limits(), slice->slice_strides()));
    TF_RET_CHECK(ShapeUtil::Compatible(shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const int64 rank = ShapeUtil::Rank(operand->shape());
    auto operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    auto func = [&](tensorflow::gtl::ArraySlice<int64> out_index) {
      DimensionVector operand_index(rank);
      for (int64 i = 0; i < rank; ++i) {
        operand_index[i] =
            slice->slice_starts(i) + out_index[i] * slice->slice_strides(i);
      }
      return operand_literal.Get<ReturnT>(operand_index);
    };

    auto result = Literal::CreateFromDimensions(
        shape.element_type(), AsInt64Slice(shape.dimensions()));
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(func));
    parent_->evaluated_[slice] = std::move(result);
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleSin(HloInstruction* sin) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sin],
                        ElementWiseUnaryOp(sin, [](ReturnT elem_operand) {
                          return std::sin(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_integral<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleSin(HloInstruction* sin) {
    return InvalidArgument("Unsupported type for Sin");
  }

  Status HandleSin(HloInstruction* sin) override {
    return HandleSin<ReturnT>(sin);
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleCos(HloInstruction* cos) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[cos],
                        ElementWiseUnaryOp(cos, [](ReturnT elem_operand) {
                          return std::cos(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_integral<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCos(HloInstruction* cos) {
    return InvalidArgument("Unsupported type for Cos");
  }

  Status HandleCos(HloInstruction* cos) override {
    return HandleCos<ReturnT>(cos);
  }

 private:
  template <typename IndexT>
  StatusOr<std::unique_ptr<Literal>> DynamicSlice(
      const Literal& operand_literal, const Literal& start_indices_literal,
      const Shape& result_shape) {
    const auto& start_indices_typed =
        start_indices_literal.GetArraySlice<IndexT>();
    std::vector<int64> start(start_indices_typed.begin(),
                             start_indices_typed.end());

    std::vector<int64> operand_indices(start.size());

    auto result = Literal::CreateFromShape(result_shape);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          for (int64 i = 0; i < operand_indices.size(); ++i) {
            CHECK_GE(multi_index[i] + start[i], 0);
            // Mod is only used here to be consistent with the existing
            // backends' behavior.
            operand_indices[i] = (multi_index[i] + start[i]) %
                                 operand_literal.shape().dimensions(i);
          }

          auto result = operand_literal.Get<ReturnT>(operand_indices);
          return result;
        }));

    return std::move(result);
  }

  template <typename IndexT>
  StatusOr<std::unique_ptr<Literal>> DynamicUpdateSlice(
      const Literal& operand_literal, const Literal& update_literal,
      const Literal& start_indices_literal) {
    const auto& start_indices_typed =
        start_indices_literal.GetArraySlice<IndexT>();
    const std::vector<int64> start(start_indices_typed.begin(),
                                   start_indices_typed.end());

    auto result = MakeUnique<Literal>(operand_literal);
    std::vector<int64> result_index(ShapeUtil::Rank(result->shape()), 0);

    auto func = [&](const std::vector<int64>& update_index) {
      std::transform(update_index.begin(), update_index.end(), start.begin(),
                     result_index.begin(), std::plus<int64>());

      result->Set<ReturnT>(result_index,
                           update_literal.Get<ReturnT>(update_index));
      return true;
    };

    std::vector<int64> base(update_literal.shape().dimensions_size(), 0);
    std::vector<int64> step(update_literal.shape().dimensions_size(), 1);
    ShapeUtil::ForEachIndex(update_literal.shape(), base,
                            AsInt64Slice(update_literal.shape().dimensions()),
                            step, func);

    return std::move(result);
  }

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

    auto result = Literal::CreateFromShape(shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return binary_op(lhs_literal.Get<ReturnT>(multi_index),
                           rhs_literal.Get<ReturnT>(multi_index));
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

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit
    // broadcast is removed.
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

    auto result = Literal::CreateFromShape(shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return ternary_op(lhs_literal.Get<LhsType>(multi_index),
                            rhs_literal.Get<RhsType>(multi_index),
                            ehs_literal.Get<EhsType>(multi_index));
        }));

    return std::move(result);
  }

  HloEvaluator* parent_;
};  // class HloEvaluator::TypedVisitor

HloEvaluator::HloEvaluator() {
  typed_visitors_[PRED] = MakeUnique<TypedVisitor<bool>>(this);
  typed_visitors_[U8] = MakeUnique<TypedVisitor<uint8>>(this);
  typed_visitors_[U16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: U16.");
  });
  typed_visitors_[U32] = MakeUnique<TypedVisitor<uint32>>(this);
  typed_visitors_[U64] = MakeUnique<TypedVisitor<uint64>>(this);
  typed_visitors_[S8] = MakeUnique<TypedVisitor<int8>>(this);
  typed_visitors_[S16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: S16.");
  });
  typed_visitors_[S32] = MakeUnique<TypedVisitor<int32>>(this);
  typed_visitors_[S64] = MakeUnique<TypedVisitor<int64>>(this);
  typed_visitors_[F16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: F16.");
  });
  typed_visitors_[F32] = MakeUnique<TypedVisitor<float>>(this);
  typed_visitors_[F64] = MakeUnique<TypedVisitor<double>>(this);
  typed_visitors_[C64] = MakeUnique<TypedVisitor<complex64>>(this);

  typed_visitors_[BF16] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: BF16.");
  });
  typed_visitors_[TUPLE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: TUPLE.");
  });
  typed_visitors_[OPAQUE] = MakeUnique<FunctionVisitor>([](HloInstruction*) {
    return Unimplemented("HloEvaluator: unhandled primitive type: OPAQUE.");
  });
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    const HloModule& module,
    tensorflow::gtl::ArraySlice<const Literal*> arg_literals) {
  XLA_VLOG_LINES(2, "HloEvaluator::Evaluate module:\n" + module.ToString());

  arg_literals_ = arg_literals;
  evaluated_.clear();

  TF_RETURN_IF_ERROR(module.entry_computation()->Accept(this));

  return MakeUnique<Literal>(
      GetEvaluatedLiteralFor(module.entry_computation()->root_instruction()));
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    const HloComputation& computation,
    tensorflow::gtl::ArraySlice<const Literal*> arg_literals) {
  XLA_VLOG_LINES(
      2, "HloEvaluator::Evaluate computation:\n" + computation.ToString());
  arg_literals_ = arg_literals;
  evaluated_.clear();

  TF_RETURN_IF_ERROR(computation.Accept(this));
  return MakeUnique<Literal>(
      GetEvaluatedLiteralFor(computation.root_instruction()));
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
              << input_literal->ToString();
      TF_RET_CHECK(ShapeUtil::Equal(operand->shape(), input_literal->shape()));

      evaluated_[operand] = MakeUnique<Literal>(*input_literal);
    }
  }

  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
  return MakeUnique<Literal>(GetEvaluatedLiteralFor(instruction));
}

StatusOr<std::unique_ptr<Literal>> HloEvaluator::Evaluate(
    HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kParameter) {
    return tensorflow::errors::FailedPrecondition(
        "Cannot evaluate a parameter.");
  }
  if (!hlo_query::AllOperandsAreConstants(*instruction)) {
    return tensorflow::errors::FailedPrecondition(
        "Not all operands are constants.");
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(instruction->shape()));

  arg_literals_.clear();
  evaluated_.clear();

  TF_RETURN_IF_ERROR(Preprocess(instruction));
  TF_RETURN_IF_ERROR(instruction->Visit(this));
  TF_RETURN_IF_ERROR(Postprocess(instruction));
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

StatusOr<std::unique_ptr<Literal>> HloEvaluator::EvaluateWithSubstitutions(
    const HloInstruction* instruction,
    const std::unordered_map<const HloInstruction*, const Literal*>&
        substitutions) {
  std::vector<std::unique_ptr<HloInstruction>> owned_operands;
  for (const HloInstruction* operand : instruction->operands()) {
    auto it = substitutions.find(operand);
    if (it == substitutions.end()) {
      owned_operands.push_back(operand->Clone());
    } else {
      owned_operands.push_back(
          HloInstruction::CreateConstant(it->second->CloneToUnique()));
    }
  }

  std::vector<HloInstruction*> operands;
  operands.reserve(owned_operands.size());
  for (auto& operand : owned_operands) {
    operands.push_back(operand.get());
  }

  std::unique_ptr<HloInstruction> cloned_instruction =
      instruction->CloneWithNewOperands(instruction->shape(), operands);
  auto result = Evaluate(cloned_instruction.get());

  // Clean up our cloned instructions before returning.
  cloned_instruction->DetachFromOperands();
  for (auto& operand : owned_operands) {
    operand->DetachFromOperands();
  }

  return result;
}

Status HloEvaluator::HandleParameter(HloInstruction* parameter) {
  const Literal* input_literal = arg_literals_[parameter->parameter_number()];
  VLOG(2) << "Parameter evaluated to: " << input_literal->ToString();
  DCHECK(ShapeUtil::Equal(parameter->shape(), input_literal->shape()));

  evaluated_[parameter] = MakeUnique<Literal>(*input_literal);
  return Status::OK();
}

Status HloEvaluator::HandleConstant(HloInstruction*) { return Status::OK(); }

Status HloEvaluator::HandleReshape(HloInstruction* reshape) {
  TF_ASSIGN_OR_RETURN(
      evaluated_[reshape],
      GetEvaluatedLiteralFor(reshape->operand(0))
          .Reshape(AsInt64Slice(reshape->shape().dimensions())));
  return Status::OK();
}

Status HloEvaluator::HandleTranspose(HloInstruction* transpose) {
  evaluated_[transpose] = GetEvaluatedLiteralFor(transpose->operand(0))
                              .Transpose(transpose->dimensions());
  return Status::OK();
}

Status HloEvaluator::HandleConcatenate(HloInstruction* concatenate) {
  tensorflow::gtl::ArraySlice<HloInstruction*> operands(
      concatenate->operands());
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

  auto result_literal = Literal::CreateFromDimensions(
      reference_shape.element_type(), concat_dimensions);
  DimensionVector source_indices(rank, 0);
  DimensionVector dest_indices(concat_dimensions.size(), 0);

  for (auto operand : operands) {
    const Shape& operand_shape = operand->shape();
    TF_RETURN_IF_ERROR(result_literal->Copy(
        GetEvaluatedLiteralFor(operand), source_indices, dest_indices,
        AsInt64Slice(operand_shape.dimensions())));
    dest_indices[concat_dim] +=
        ShapeUtil::GetDimension(operand_shape, concat_dim);
  }

  evaluated_[concatenate] = std::move(result_literal);
  return Status::OK();
}

Status HloEvaluator::HandleIsFinite(HloInstruction* is_finite) {
  auto operand = is_finite->operand(0);
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
      LOG(FATAL) << "HandleIsFinite: unknown/unhandled primitive type: "
                 << PrimitiveType_Name(operand->shape().element_type());
  }

  return Status::OK();
}

Status HloEvaluator::HandleCompare(HloInstruction* compare) {
  HloOpcode opcode = compare->opcode();
  auto lhs = compare->operand(0);
  auto rhs = compare->operand(1);
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
    case C64: {
      TF_ASSIGN_OR_RETURN(evaluated_[compare],
                          Compare<complex64>(compare->shape(), opcode,
                                             lhs_literal, rhs_literal));
    } break;
    default:
      LOG(FATAL) << "HandleCompare: unknown primitive type: "
                 << PrimitiveType_Name(lhs->shape().element_type());
  }

  return Status::OK();
}

Status HloEvaluator::HandleTuple(HloInstruction* tuple) {
  std::vector<const Literal*> operand_literals;
  for (auto operand : tuple->operands()) {
    operand_literals.push_back(&GetEvaluatedLiteralFor(operand));
  }

  evaluated_[tuple] = Literal::MakeTuple(operand_literals);
  return Status::OK();
}

Status HloEvaluator::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  const auto result_shape = get_tuple_element->shape();
  const int64 index = get_tuple_element->tuple_index();

  auto operand = get_tuple_element->operand(0);
  TF_ASSIGN_OR_RETURN(
      auto inferred_return_shape,
      ShapeInference::InferGetTupleElementShape(operand->shape(), index));
  TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
      << "return shape set to: " << ShapeUtil::HumanString(result_shape)
      << " but is inferred to be: "
      << ShapeUtil::HumanString(inferred_return_shape);

  const Literal& operand_tuple_literal = GetEvaluatedLiteralFor(operand);

  evaluated_[get_tuple_element] =
      MakeUnique<Literal>(operand_tuple_literal.tuple_literals(index));

  return Status::OK();
}

Status HloEvaluator::HandleCopy(HloInstruction* copy) {
  TF_RET_CHECK(ShapeUtil::Compatible(copy->shape(), copy->operand(0)->shape()));

  auto result = MakeUnique<Literal>(GetEvaluatedLiteralFor(copy->operand(0)));
  evaluated_[copy] = std::move(result);
  return Status::OK();
}

Status HloEvaluator::Preprocess(HloInstruction* hlo) {
  VLOG(2) << "About to visit HLO: " << hlo->ToString();
  return Status::OK();
}

Status HloEvaluator::Postprocess(HloInstruction* hlo) {
  VLOG(2) << "Finished visiting " << hlo->ToString()
          << "; evaluated value is: " << GetEvaluatedLiteralFor(hlo).ToString();
  return Status::OK();
}

}  // namespace xla
