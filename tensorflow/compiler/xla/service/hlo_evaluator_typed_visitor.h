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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_TYPED_VISITOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_TYPED_VISITOR_H_

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/gtl/optional.h"

namespace xla {

// TODO(b/79274244): We'd like these type traits to live inside of
// HloEvaluatorTypedVisitor so they don't pollute namespace xla, but that
// crashes clang in the frontend.
//
// Anyway this is relatively safe as-is because hlo_evaluator_typed_visitor.h is
// a "private" header that's not exposed outside of hlo_evaluator.cc.
template <typename T>
using is_complex_t = std::is_same<T, complex64>;
template <typename T>
using is_complex64_t = std::is_same<T, complex64>;

// It's UB to use std::sort with std::less<float>, because of NaNs. Define
// "safe" less functions which are actually strict weak orders.
template <
    typename NativeT,
    typename std::enable_if<std::is_integral<NativeT>::value>::type* = nullptr>
bool SafeLess(const NativeT& a, const NativeT& b) {
  return a < b;
}

template <typename NativeT,
          typename std::enable_if<
              std::is_floating_point<NativeT>::value ||
              std::is_same<NativeT, bfloat16>::value>::type* = nullptr>
bool SafeLess(const NativeT& a, const NativeT& b) {
  if (std::isnan(b)) {
    return !std::isnan(a);
  } else {
    return a < b;
  }
}

template <typename NativeT, typename std::enable_if<std::is_same<
                                NativeT, Eigen::half>::value>::type* = nullptr>
bool SafeLess(const NativeT& a, const NativeT& b) {
  if (Eigen::half_impl::isnan(b)) {
    return !Eigen::half_impl::isnan(a);
  } else {
    return a < b;
  }
}

// Templated DfsHloVisitor for use by HloEvaluator.
//
// Typically ReturnT here indicates the resulting literal type of each evaluated
// Handle* method of a TypedVisitor.  There are however a few notable exceptions
// to this rule, notably:
// - HandleCompare and HandleIsFinite: where the resulting literal type is
//   always boolean.
// These operations are handled outside of the parent HloEvaluator handlers
// instead of from within TypedVisitor.
//
// Type params:
//   - ReturnT: The type of input and output of each operation.
//   - ElementwiseT: The type in which internal computation are done.
//
// This a logically a private part of HloEvaluator.  It lives in this header
// file rather than in hlo_evaluator.cc because we use extern templates and a
// bunch of independent cc files to speed up compiling the many instantiations
// of this class.
template <typename ReturnT, typename ElementwiseT = ReturnT>
class HloEvaluatorTypedVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit HloEvaluatorTypedVisitor(HloEvaluator* p) : parent_(p) {}

  // The following higher-order functions convert a function with ElementwiseT
  // to a function with ReturnT.
  std::function<ReturnT(ReturnT)> ConvertUnaryFunction(
      const std::function<ElementwiseT(ElementwiseT)>& unary_op) {
    return [&unary_op](ReturnT arg) {
      return static_cast<ReturnT>(unary_op(static_cast<ElementwiseT>(arg)));
    };
  }
  std::function<ReturnT(ReturnT, ReturnT)> ConvertBinaryFunction(
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT)>&
          binary_op) {
    return [&binary_op](ReturnT arg1, ReturnT arg2) {
      return static_cast<ReturnT>(binary_op(static_cast<ElementwiseT>(arg1),
                                            static_cast<ElementwiseT>(arg2)));
    };
  }
  std::function<ReturnT(ReturnT, ReturnT, ReturnT)> ConvertTernaryFunction(
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT,
                                       ElementwiseT)>& ternary_op) {
    return [&ternary_op](ReturnT arg1, ReturnT arg2, ReturnT arg3) {
      return static_cast<ReturnT>(ternary_op(static_cast<ElementwiseT>(arg1),
                                             static_cast<ElementwiseT>(arg2),
                                             static_cast<ElementwiseT>(arg3)));
    };
  }

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Unimplemented("unhandled HLO ops for HloEvaluator: %s.",
                         HloOpcodeString(hlo_instruction->opcode()).c_str());
  }

  // TODO(b/35950897): many of the stl functions used in the handlers are not
  // overloaded for every XLA primitive type.

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
      typename std::enable_if<std::is_signed<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex64_t<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(abs->operand(0));
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[abs],
        (HloEvaluator::ElementWiseUnaryOpImpl<float, NativeT>(
            abs, [](NativeT elem_operand) { return std::abs(elem_operand); },
            operand_literal)));

    return Status::OK();
  }

  Status HandleAbs(HloInstruction* abs) override {
    // If the operand is of C64 type, the return type of abs will be F32.
    // However, ElementwiseT would still be the return type, F32, and thus
    // specifying the ElementwiseT explicitly as C64 is needed below.
    if (abs->operand(0)->shape().element_type() == C64) {
      return HandleAbs<complex64>(abs);
    }
    return HandleAbs<ElementwiseT>(abs);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRound(HloInstruction* round) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[round],
        ElementWiseUnaryOp(round, [](ElementwiseT elem_operand) {
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

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCeil(HloInstruction* ceil) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[ceil],
                        ElementWiseUnaryOp(ceil, [](ElementwiseT elem_operand) {
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

  Status HandleBitcastConvert(HloInstruction* convert) override {
    const HloInstruction* operand = convert->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), convert->shape()));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Literal> result,
                        parent_->GetEvaluatedLiteralFor(operand).BitcastConvert(
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
                        ElementWiseUnaryOp(exp, [](ElementwiseT elem_operand) {
                          return std::exp(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleExpm1(HloInstruction* expm1) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[expm1],
        ElementWiseUnaryOp(expm1, [](ElementwiseT elem_operand) {
          return std::expm1(elem_operand);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleExpm1(HloInstruction* floor) {
    return InvalidArgument("Unsupported type for Expm1");
  }

  Status HandleExpm1(HloInstruction* floor) override {
    return HandleExpm1<ReturnT>(floor);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleFloor(HloInstruction* floor) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[floor],
        ElementWiseUnaryOp(floor, [](ElementwiseT elem_operand) {
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
                        ElementWiseUnaryOp(log, [](ElementwiseT elem_operand) {
                          return std::log(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleLog1p(HloInstruction* expm1) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[expm1],
        ElementWiseUnaryOp(expm1, [](ElementwiseT elem_operand) {
          return std::log1p(elem_operand);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleLog1p(HloInstruction* floor) {
    return InvalidArgument("Unsupported type for Log1p");
  }

  Status HandleLog1p(HloInstruction* floor) override {
    return HandleLog1p<ReturnT>(floor);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return ~elem_operand;
                        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return !elem_operand;
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
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
    return HandleNot<ElementwiseT>(not_);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_signed<NativeT>::value &&
                !std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    using type = typename std::make_unsigned<NativeT>::type;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(negate, [](ElementwiseT elem_operand) {
          return NativeT(-type(elem_operand));
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                !std::is_signed<NativeT>::value ||
                std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(
            negate, [](ElementwiseT elem_operand) { return -elem_operand; }));
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
                        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
                          return (ElementwiseT(0) < elem_operand) -
                                 (elem_operand < ElementwiseT(0));
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleSign(HloInstruction* sign) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
                          auto abs_val = std::abs(elem_operand);
                          return 0 == abs_val ? ElementwiseT(0)
                                              : elem_operand / abs_val;
                        }));
    return Status::OK();
  }

  Status HandleSign(HloInstruction* sign) override {
    return HandleSign<ReturnT>(sign);
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[atan2],
                        ElementWiseBinaryOp(atan2, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                          return std::atan2(lhs_elem, rhs_elem);
                        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<!std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    return InvalidArgument("Unsupported type for Atan2");
  }

  Status HandleAtan2(HloInstruction* atan2) override {
    return HandleAtan2<ElementwiseT>(atan2);
  }

  Status HandleTanh(HloInstruction* tanh) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[tanh],
                        ElementWiseUnaryOp(tanh, [](ElementwiseT elem_operand) {
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
        ElementWiseBinaryOp(multiply,
                            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
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
        ElementWiseBinaryOp(multiply,
                            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                              return lhs_elem * rhs_elem;
                            }));
    return Status::OK();
  }

  Status HandleMultiply(HloInstruction* multiply) override {
    return HandleMultiply<ElementwiseT>(multiply);
  }

  Status HandleSubtract(HloInstruction* subtract) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(subtract,
                            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                              return lhs_elem - rhs_elem;
                            }));
    return Status::OK();
  }

  Status HandleAdd(HloInstruction* add) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[add],
                        ElementWiseBinaryOp(add, [](ElementwiseT lhs_elem,
                                                    ElementwiseT rhs_elem) {
                          return lhs_elem + rhs_elem;
                        }));
    return Status::OK();
  }

  Status HandleDivide(HloInstruction* divide) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[divide],
                        ElementWiseBinaryOp(divide, [](ElementwiseT lhs_elem,
                                                       ElementwiseT rhs_elem) {
                          return lhs_elem / rhs_elem;
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleMaximum(HloInstruction* maximum) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[maximum],
        ElementWiseBinaryOp(maximum, [](ElementwiseT lhs, ElementwiseT rhs) {
          return std::max(lhs, rhs);
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleMaximum(HloInstruction* maximum) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[maximum],
        ElementWiseBinaryOp(maximum, [](ElementwiseT lhs, ElementwiseT rhs) {
          return ((lhs >= rhs) || std::isnan(lhs)) ? lhs : rhs;
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
    return HandleMaximum<ElementwiseT>(maximum);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleMinimum(HloInstruction* minimum) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[minimum],
                        ElementWiseBinaryOp(minimum, [](ElementwiseT lhs_el,
                                                        ElementwiseT rhs_el) {
                          return std::min(lhs_el, rhs_el);
                        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleMinimum(HloInstruction* minimum) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[minimum],
        ElementWiseBinaryOp(minimum, [](ElementwiseT lhs_el,
                                        ElementwiseT rhs_el) {
          return ((lhs_el <= rhs_el) || std::isnan(lhs_el)) ? lhs_el : rhs_el;
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
    return HandleMinimum<ElementwiseT>(minimum);
  }

  Status HandlePower(HloInstruction* power) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[power],
                        ElementWiseBinaryOp(power, [](ElementwiseT lhs_el,
                                                      ElementwiseT rhs_el) {
                          return std::pow(lhs_el, rhs_el);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[remainder],
                        ElementWiseBinaryOp(remainder, [](ElementwiseT lhs_el,
                                                          ElementwiseT rhs_el) {
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
    return HandleRemainder<ElementwiseT>(remainder);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleAnd(HloInstruction* and_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[and_],
        ElementWiseBinaryOp(and_, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
          return lhs_el & rhs_el;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleAnd(HloInstruction* and_) {
    return InvalidArgument("Unsupported type for And");
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAnd(HloInstruction* and_) {
    return InvalidArgument("Unsupported type for And");
  }

  Status HandleAnd(HloInstruction* and_) override {
    return HandleAnd<ElementwiseT>(and_);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleOr(HloInstruction* or_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[or_],
        ElementWiseBinaryOp(or_, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
          return lhs_el | rhs_el;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleOr(HloInstruction* or_) {
    return InvalidArgument("Unsupported type for Or");
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleOr(HloInstruction* or_) {
    return InvalidArgument("Unsupported type for Or");
  }

  Status HandleOr(HloInstruction* or_) override {
    return HandleOr<ElementwiseT>(or_);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleXor(HloInstruction* xor_) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[xor_],
        ElementWiseBinaryOp(xor_, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
          return lhs_el ^ rhs_el;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleXor(HloInstruction* xor_) {
    return InvalidArgument("Unsupported type for Xor");
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleXor(HloInstruction* xor_) {
    return InvalidArgument("Unsupported type for Xor");
  }

  Status HandleXor(HloInstruction* xor_) override {
    return HandleXor<ElementwiseT>(xor_);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleShiftLeft(HloInstruction* shl) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[shl],
        ElementWiseBinaryOp(shl, [](NativeT lhs_elem, NativeT rhs_elem) {
          return IsShiftOutOfBounds<NativeT>(rhs_elem) ? 0
                                                       : (lhs_elem << rhs_elem);
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
    return HandleShiftLeft<ElementwiseT>(shl);
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
          SignedT lhs_signed = static_cast<SignedT>(lhs_elem);
          if (IsShiftOutOfBounds<NativeT>(rhs_elem)) {
            return lhs_signed < 0 ? static_cast<SignedT>(-1) : 0;
          } else {
            return lhs_signed >> rhs_elem;
          }
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
    return HandleShiftRightArithmetic<ElementwiseT>(shra);
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
          // If shift amount is greater than the number of bits, then return 0.
          if (IsShiftOutOfBounds<NativeT>(rhs_elem)) {
            return static_cast<NativeT>(0);
          }
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
    return HandleShiftRightLogical<ElementwiseT>(shrl);
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleClamp(HloInstruction* clamp) {
    std::function<ElementwiseT(ElementwiseT, ElementwiseT, ElementwiseT)>
        clamp_op = [](ElementwiseT low, ElementwiseT value, ElementwiseT high) {
          return std::fmin(high, std::fmax(value, low));
        };
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[clamp],
        ElementwiseTernaryOp(clamp,
                             std::move(ConvertTernaryFunction(clamp_op))));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleClamp(HloInstruction*) {
    return InvalidArgument("Unsupported type for Clamp");
  }

  Status HandleClamp(HloInstruction* clamp) override {
    return HandleClamp<ElementwiseT>(clamp);
  }

  Status HandleSelect(HloInstruction* select) override {
    CHECK(!ShapeUtil::IsScalar(select->operand(0)->shape()));
    CHECK(ShapeUtil::IsArray(select->shape()));
    std::function<ReturnT(bool, ReturnT, ReturnT)> select_op =
        [](bool pred, ReturnT on_true, ReturnT on_false) {
          if (pred) {
            return on_true;
          }
          return on_false;
        };
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[select],
                        ElementwiseTernaryOp(select, std::move(select_op)));
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

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    auto result = MakeUnique<Literal>(result_shape);

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

    std::vector<int64> window_dimension_sizes;
    for (auto i : dnums.kernel_spatial_dimensions()) {
      window_dimension_sizes.push_back(ShapeUtil::GetDimension(rhs_shape, i));
    }

    const Shape& window_shape =
        ShapeUtil::MakeShape(rhs_shape.element_type(), window_dimension_sizes);

    DimensionVector lhs_dim_multipliers = MakeDimMultipliers(lhs_shape);
    DimensionVector rhs_dim_multipliers = MakeDimMultipliers(rhs_shape);

    auto lhs_literal_data = lhs_literal.data<ReturnT>();
    auto rhs_literal_data = rhs_literal.data<ReturnT>();

    auto func = [&window_shape, &dnums, &lhs_shape, &rhs_shape, &window,
                 &lhs_dim_multipliers, &rhs_dim_multipliers, lhs_literal_data,
                 rhs_literal_data](
                    tensorflow::gtl::ArraySlice<int64> out_index) {
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

      ElementwiseT result_val = static_cast<ElementwiseT>(0);
      DimensionVector rhs_spatial_index(dnums.kernel_spatial_dimensions_size(),
                                        0);

      // Convolve input feature with kernel.
      do {
        for (int64 iz = 0; iz < z_size; ++iz) {
          int64 lhs_linear_index = 0;
          lhs_linear_index += out_index[output_batch_dim] *
                              lhs_dim_multipliers[input_batch_dim];
          lhs_linear_index += iz * lhs_dim_multipliers[input_z_dim];

          int64 rhs_linear_index = 0;
          rhs_linear_index += out_index[output_z_dim] *
                              rhs_dim_multipliers[kernel_output_z_dim];
          rhs_linear_index += iz * rhs_dim_multipliers[kernel_input_z_dim];

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
            // Skip if the lhs (input) index is to be dilated.  As an
            // optimization, skip this mod if there's no dilation.
            if (window_dim.base_dilation() > 1 &&
                undilated_index % window_dim.base_dilation() != 0) {
              goto cnt;
            }

            // Calculate the actual lhs (input) index after dilation.  As an
            // optimization, skip this integer divide if there's no dilation.
            int64 lhs_spatial_index;
            if (window_dim.base_dilation() > 1) {
              lhs_spatial_index = undilated_index / window_dim.base_dilation();
            } else {
              lhs_spatial_index = undilated_index;
            }
            lhs_linear_index +=
                lhs_spatial_index * lhs_dim_multipliers[input_spatial_dim];

            // Skip if input index is not in bounds.
            if (!(lhs_spatial_index >= 0 &&
                  lhs_spatial_index <
                      lhs_shape.dimensions(input_spatial_dim))) {
              goto cnt;
            }

            rhs_linear_index +=
                (window_dim.window_reversal()
                     ? ((window_dim.size() - 1) - rhs_spatial_index[ki])
                     : rhs_spatial_index[ki]) *
                rhs_dim_multipliers[dnums.kernel_spatial_dimensions(ki)];
          }

          result_val +=
              static_cast<ElementwiseT>(lhs_literal_data[lhs_linear_index]) *
              static_cast<ElementwiseT>(rhs_literal_data[rhs_linear_index]);
        }
      cnt : {}
      } while (IndexUtil::BumpIndices(window_shape, &rhs_spatial_index));

      return static_cast<ReturnT>(result_val);
    };

    auto result = MakeUnique<Literal>(result_shape);
    TF_RETURN_IF_ERROR(result->PopulateParallel<ReturnT>(func));

    parent_->evaluated_[conv] = std::move(result);
    return Status::OK();
  }

  Status HandleDot(HloInstruction* dot) override {
    auto lhs = dot->operand(0);
    auto rhs = dot->operand(1);
    CHECK(ShapeUtil::IsArray(dot->shape()));
    CHECK(ShapeUtil::IsArray(lhs->shape()));
    CHECK(ShapeUtil::IsArray(rhs->shape()));

    const auto& dnums = dot->dot_dimension_numbers();

    const auto lhs_rank = ShapeUtil::Rank(lhs->shape());
    const auto rhs_rank = ShapeUtil::Rank(rhs->shape());

    CHECK(ShapeUtil::SameElementType(lhs->shape(), rhs->shape()));
    CHECK(ShapeUtil::SameElementType(lhs->shape(), dot->shape()));

    // There must be 1 and only 1 Contracting dimension for lhs and rhs.
    CHECK_EQ(dnums.lhs_contracting_dimensions_size(), 1);
    CHECK_EQ(dnums.rhs_contracting_dimensions_size(), 1);
    const int64 lhs_contracting_dimension = dnums.lhs_contracting_dimensions(0);
    const int64 rhs_contracting_dimension = dnums.rhs_contracting_dimensions(0);
    // Contracted dimension sizes must be the same.
    CHECK_EQ(lhs->shape().dimensions(lhs_contracting_dimension),
             rhs->shape().dimensions(rhs_contracting_dimension))
        << "lhs contracted dimension: "
        << lhs->shape().dimensions(lhs_contracting_dimension)
        << " rhs contracted dimension: "
        << rhs->shape().dimensions(rhs_contracting_dimension);
    const int64 contracted_dimension_size =
        lhs->shape().dimensions(lhs_contracting_dimension);

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    CHECK_EQ(dnums.lhs_batch_dimensions_size(),
             dnums.rhs_batch_dimensions_size());

    DimensionVector lhs_index(lhs_rank);
    DimensionVector rhs_index(rhs_rank);

    // result_index_locations[i] contains one or two pointers to the locations
    // in lhs_index or rhs_index where the i'th result index should go.
    tensorflow::gtl::InlinedVector<std::pair<int64*, int64*>, kInlineRank>
        result_index_locations;
    result_index_locations.reserve(lhs_rank + rhs_rank - 2);

    // The first components in the output shape are the LHS and RHS batch
    // dimensions:
    for (int64 i = 0; i < dnums.lhs_batch_dimensions_size(); i++) {
      result_index_locations.push_back(
          {&lhs_index[dnums.lhs_batch_dimensions(i)],
           &rhs_index[dnums.rhs_batch_dimensions(i)]});
    }

    // Then we have the LHS and RHS non-contracting dimensions, if any:
    for (int64 i = 0; i < lhs_rank; i++) {
      if (i != lhs_contracting_dimension &&
          !ArrayContains(AsInt64Slice(dnums.lhs_batch_dimensions()), i)) {
        result_index_locations.push_back({&lhs_index[i], nullptr});
      }
    }
    for (int64 i = 0; i < rhs_rank; i++) {
      if (i != rhs_contracting_dimension &&
          !ArrayContains(AsInt64Slice(dnums.rhs_batch_dimensions()), i)) {
        result_index_locations.push_back({&rhs_index[i], nullptr});
      }
    }

    auto result = MakeUnique<Literal>(dot->shape());
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> result_index) {
          ElementwiseT result_val = static_cast<ElementwiseT>(0);

          for (int64 i = 0; i < result_index.size(); i++) {
            *result_index_locations[i].first = result_index[i];
            if (result_index_locations[i].second) {
              *result_index_locations[i].second = result_index[i];
            }
          }

          // Accumulates resulting product along the contracted dimension.
          for (int64 i = 0; i < contracted_dimension_size; ++i) {
            lhs_index[lhs_contracting_dimension] = i;
            rhs_index[rhs_contracting_dimension] = i;

            result_val +=
                static_cast<ElementwiseT>(lhs_literal.Get<ReturnT>(lhs_index)) *
                static_cast<ElementwiseT>(rhs_literal.Get<ReturnT>(rhs_index));
          }

          return static_cast<ReturnT>(result_val);
        }));

    parent_->evaluated_[dot] = std::move(result);
    return Status::OK();
  }

  Status HandlePad(HloInstruction* pad) override {
    CHECK(ShapeUtil::IsArray(pad->operand(0)->shape()));
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
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    // Create new HLO of padded shape with padding value.
    ReturnT scalar =
        parent_->GetEvaluatedLiteralFor(pad->operand(1)).Get<ReturnT>({});
    auto result = MakeUnique<Literal>(pad->shape());
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&scalar](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return scalar;
        }));

    const Literal& evaluated_operand =
        parent_->GetEvaluatedLiteralFor(pad->operand(0));

    std::vector<int64> input_index(ShapeUtil::Rank(evaluated_operand.shape()),
                                   0);
    std::vector<int64> target_index(ShapeUtil::Rank(result->shape()), 0);

    // Loop through each element of the operand, assign them to the
    // corresponding index of the resulting padded literal.
    const PaddingConfig& pad_config = pad->padding_config();

    auto func = [&](tensorflow::gtl::ArraySlice<int64> input_index) {
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
        << " but is inferred to be: "
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
        << " but is inferred to be: "
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

  template <typename NativeT>
  StatusOr<std::unique_ptr<Literal>> MapImpl(HloInstruction* map) {
    auto operands = map->operands();
    HloComputation* computation = map->to_apply();

    auto result = MakeUnique<Literal>(map->shape());

    HloEvaluator embedded_evaluator(parent_->max_loop_iterations_);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          std::vector<std::unique_ptr<Literal>> arg_literals;
          arg_literals.reserve(operands.size());

          // Construct scalar literal parameters to be passed to the map
          // computation.
          for (auto operand : operands) {
            const Literal& arg_literal =
                parent_->GetEvaluatedLiteralFor(operand);

            auto curr_val = arg_literal.Get<NativeT>(multi_index);
            auto curr_val_literal = LiteralUtil::CreateR0<NativeT>(curr_val);

            arg_literals.push_back(std::move(curr_val_literal));
          }

          std::unique_ptr<Literal> computed_result =
              embedded_evaluator
                  .Evaluate<std::unique_ptr<Literal>>(*computation,
                                                      arg_literals)
                  .ConsumeValueOrDie();
          // Clear visit states so that the we can use the evaluate again on
          // the same computation.
          embedded_evaluator.ResetVisitStates();

          return computed_result->Get<ReturnT>({});
        }));
    return std::move(result);
  }

  Status HandleMap(HloInstruction* map) override {
    switch (map->operand(0)->shape().element_type()) {
      case PRED: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<bool>(map));
        break;
      }
      case U8: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint8>(map));
        break;
      }
      case U32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint32>(map));
        break;
      }
      case U64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint64>(map));
        break;
      }
      case S8: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int8>(map));
        break;
      }
      case S32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int32>(map));
        break;
      }
      case S64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int64>(map));
        break;
      }
      case F16: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map],
                            MapImpl<Eigen::half>(map));
        break;
      }
      case F32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<float>(map));
        break;
      }
      case F64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<double>(map));
        break;
      }
      case C64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<complex64>(map));
        break;
      }
      default:
        LOG(FATAL) << "HandleMap: unhandled primitive type for "
                      "input operand: "
                   << PrimitiveType_Name(
                          map->operand(0)->shape().element_type());
    }

    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                !is_complex_t<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleSort(HloInstruction* sort) {
    auto keys = sort->operand(0);
    TF_RET_CHECK(ShapeUtil::Rank(keys->shape()) == 1)
        << "Sort is only supported for R1 shapes";
    TF_RET_CHECK(sort->operand_count() == 1)
        << "Typed visitor does not support key-value sort";

    const Literal& keys_literal = parent_->GetEvaluatedLiteralFor(keys);
    VLOG(3) << "HandleSort keys_literal: " << keys_literal.ToString();
    const auto& keys_data = keys_literal.data<ReturnT>();

    std::vector<ReturnT> result_data(keys_data.begin(), keys_data.end());
    std::sort(result_data.begin(), result_data.end(),
              [](const ReturnT& a, const ReturnT& b) {
                return SafeLess<ReturnT>(a, b);
              });
    auto result_literal = MakeUnique<Literal>(sort->shape());
    result_literal->PopulateR1(
        tensorflow::gtl::ArraySlice<ReturnT>(result_data));
    VLOG(3) << "HandleSort result_literal: " << result_literal->ToString();
    parent_->evaluated_[sort] = std::move(result_literal);
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<is_complex_t<NativeT>::value ||
                                    std::is_same<NativeT, bool>::value>::type* =
                nullptr>
  Status HandleSort(HloInstruction* sort) {
    return InvalidArgument("Unsupported type for Sort");
  }

  Status HandleSort(HloInstruction* sort) override {
    return HandleSort<ReturnT>(sort);
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
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const Literal& arg_literal = parent_->GetEvaluatedLiteralFor(arg);
    VLOG(3) << "HandleReduce arg_literal: " << arg_literal.ToString();
    const Literal& init_literal = parent_->GetEvaluatedLiteralFor(init_value);
    VLOG(3) << "HandleReduce init_literal: " << init_literal.ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    const auto arg_dimensions = AsInt64Slice(arg_literal.shape().dimensions());
    std::vector<int64> arg_dim_steps(arg_dimensions.size());
    std::vector<int64> arg_dim_counts(arg_dimensions.size());
    for (const int64 dim : dimensions) {
      arg_dim_steps[dim] = 1;
      arg_dim_counts[dim] = arg_dimensions[dim];
    }

    // Map each dimension in the result to a dimension in arg that isn't
    // being reduced.
    std::vector<int64> result_to_arg_index;
    for (int64 i = 0; i < arg_dimensions.size(); ++i) {
      if (arg_dim_steps[i] == 0) {
        result_to_arg_index.push_back(i);
      }
    }

    HloEvaluator embedded_evaluator(parent_->max_loop_iterations_);
    auto result = MakeUnique<Literal>(reduce->shape());
    // For each resulting dimension, calculate and assign computed value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          ReturnT result_val = init_scalar;

          std::vector<int64> base(arg_dimensions.size());
          for (int64 i = 0; i < multi_index.size(); ++i) {
            base[result_to_arg_index[i]] = multi_index[i];
          }

          // When the reduction is addition of floats, accumulate in a double
          // for better precision. Also, avoid creating Literals for the
          // intermediate results; it's much faster.
          if (ShapeUtil::ElementIsFloating(init_literal.shape()) &&
              IsScalarAdd(function)) {
            double computed_result = 0;
            auto func = [&](tensorflow::gtl::ArraySlice<int64> input_index) {
              computed_result += arg_literal.Get<float>(input_index);
              return true;
            };
            ShapeUtil::ForEachIndex(arg_literal.shape(), base, arg_dim_counts,
                                    arg_dim_steps, func);
            return static_cast<ReturnT>(computed_result);
          }
          auto func = [&](tensorflow::gtl::ArraySlice<int64> input_index) {
            auto curr_val = arg_literal.Get<ReturnT>(input_index);

            // Evaluate computation with specified literal operands.
            auto curr_val_literal = LiteralUtil::CreateR0<ReturnT>(curr_val);
            auto result_val_literal =
                LiteralUtil::CreateR0<ReturnT>(result_val);

            std::unique_ptr<Literal> computed_result =
                embedded_evaluator
                    .Evaluate<const Literal*>(
                        *function,
                        {result_val_literal.get(), curr_val_literal.get()})
                    .ConsumeValueOrDie();
            // Clear visit states so that we can use the evaluator again on
            // the same computation.
            embedded_evaluator.ResetVisitStates();
            // Assign computed result to result_val.
            result_val = computed_result->Get<ReturnT>({});
            return true;
          };
          // Computes one element of the result, reducing all dimensions that
          // contribute to that element.
          ShapeUtil::ForEachIndex(arg_literal.shape(), base, arg_dim_counts,
                                  arg_dim_steps, func);
          return result_val;
        }));

    parent_->evaluated_[reduce] = std::move(result);
    return Status::OK();
  }

  bool IsScalarAdd(HloComputation* computation) {
    HloInstruction* instruction = computation->root_instruction();
    if (instruction->opcode() == HloOpcode::kAdd &&
        computation->num_parameters() == 2) {
      const HloInstruction* lhs = instruction->operand(0);
      const HloInstruction* rhs = instruction->operand(1);
      return lhs->opcode() == HloOpcode::kParameter &&
             ShapeUtil::IsScalar(lhs->shape()) &&
             rhs->opcode() == HloOpcode::kParameter &&
             ShapeUtil::IsScalar(rhs->shape()) && lhs != rhs;
    }
    return false;
  }

  Status HandleSelectAndScatter(HloInstruction* select_and_scatter) override {
    auto operand = select_and_scatter->operand(0);
    auto source = select_and_scatter->operand(1);
    const Window& window = select_and_scatter->window();

    const Literal& init_literal =
        parent_->GetEvaluatedLiteralFor(select_and_scatter->operand(2));
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    auto result = MakeUnique<Literal>(select_and_scatter->shape());

    // Initialize result array with the init value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> output_index) {
          return init_scalar;
        }));

    std::vector<int64> window_dimension_sizes;
    for (const auto& window_dimension : window.dimensions()) {
      window_dimension_sizes.push_back(window_dimension.size());
    }
    const Shape window_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(), window_dimension_sizes);

    HloComputation* select = select_and_scatter->select();
    HloComputation* scatter = select_and_scatter->scatter();

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& source_literal = parent_->GetEvaluatedLiteralFor(source);

    int64 rank = ShapeUtil::Rank(operand_literal.shape());

    HloEvaluator embedded_evaluator(parent_->max_loop_iterations_);
    DimensionVector source_index(rank, 0);

    // Used in the dual IterateThroughWindow lambdas below. Hoisted to avoid
    // dynamic memory allocations.
    auto curr_val_literal = LiteralUtil::CreateR0<ReturnT>(ReturnT());
    auto selected_val_literal = LiteralUtil::CreateR0<ReturnT>(ReturnT());
    auto source_literal_scatter = LiteralUtil::CreateR0<ReturnT>(ReturnT());
    auto scattered_literal = LiteralUtil::CreateR0<ReturnT>(ReturnT());
    do {
      // For each element in `source`, we place a window in `operand`. For each
      // window placement, we iterate inside the window twice:
      //
      // 1. Find the selected index by applying `select` function to all
      // elements. E.g., If the `select` function is GreaterEqual, the first
      // iteration through the window finds the biggest value and returns its
      // index.
      //
      // 2. Using the selected index, scatter value from `source` to result. We
      // do this by iterating through the window, and compare each index with
      // the selected index.
      tensorflow::gtl::optional<ReturnT> selected_val;
      tensorflow::gtl::optional<std::vector<int64>> selected_index;

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](const std::vector<int64>& operand_index) {
            auto curr_val = operand_literal.Get<ReturnT>(operand_index);
            if (!selected_val) {
              selected_val = curr_val;
              selected_index = operand_index;
            }
            curr_val_literal->Set({}, curr_val);
            selected_val_literal->Set({}, *selected_val);
            std::unique_ptr<Literal> computed_result =
                embedded_evaluator
                    .Evaluate<const Literal*>(
                        *select,
                        {selected_val_literal.get(), curr_val_literal.get()})
                    .ConsumeValueOrDie();
            bool selected = !computed_result->Get<bool>({});
            if (selected) {
              selected_val = curr_val;
              selected_index = operand_index;
            }
            embedded_evaluator.ResetVisitStates();
          });

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](const std::vector<int64>& operand_index) {
            if (std::equal(operand_index.begin(), operand_index.end(),
                           selected_index->begin())) {
              auto source = source_literal.Get<ReturnT>(source_index);
              auto scattered = result->Get<ReturnT>(operand_index);
              source_literal_scatter->Set({}, source);
              scattered_literal->Set({}, scattered);
              std::unique_ptr<Literal> computed_result =
                  embedded_evaluator
                      .Evaluate<const Literal*>(*scatter,
                                                {source_literal_scatter.get(),
                                                 scattered_literal.get()})
                      .ConsumeValueOrDie();
              result->Set(operand_index, computed_result->Get<ReturnT>({}));
              // Clear visit states so that the we can use the evaluator again
              // on the same computation.
              embedded_evaluator.ResetVisitStates();
            }
          });
    } while (IndexUtil::BumpIndices(source->shape(), &source_index));

    parent_->evaluated_[select_and_scatter] = std::move(result);
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
        << " but is inferred to be: "
        << ShapeUtil::HumanStringWithLayout(inferred_return_shape);

    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(reduce_window->operand(0));
    VLOG(3) << "HandleReduceWindow arg_literal: " << operand_literal.ToString();
    const Literal& init_literal =
        parent_->GetEvaluatedLiteralFor(reduce_window->operand(1));
    VLOG(3) << "HandleReduceWindow init_literal: " << init_literal.ToString();
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    // Creates a Shape object from window, for iteration below.
    std::vector<int64> window_dimension_sizes;
    for (const auto& window_dimension : window.dimensions()) {
      window_dimension_sizes.push_back(window_dimension.size());
    }
    const Shape window_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(), window_dimension_sizes);

    DimensionVector window_index(window.dimensions_size());
    DimensionVector operand_index(ShapeUtil::Rank(operand_literal.shape()));

    HloEvaluator embedded_evaluator(parent_->max_loop_iterations_);
    auto result = MakeUnique<Literal>(reduce_window->shape());
    // For each resulting dimension, calculate and assign computed value.
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> output_index) {
          ReturnT result_val = init_scalar;

          std::fill(window_index.begin(), window_index.end(), 0);
          std::fill(operand_index.begin(), operand_index.end(), 0);

          IterateThroughWindow(
              window_shape, window, operand_literal.shape(), output_index,
              [&](const std::vector<int64>& operand_index) {
                auto curr_val = operand_literal.Get<ReturnT>(operand_index);

                // Evaluate computation with specified literal operands.
                const auto curr_val_literal =
                    LiteralUtil::CreateR0<ReturnT>(curr_val);
                const auto result_val_literal =
                    LiteralUtil::CreateR0<ReturnT>(result_val);
                std::unique_ptr<Literal> computed_result =
                    embedded_evaluator
                        .Evaluate<const Literal*>(
                            *function,
                            {result_val_literal.get(), curr_val_literal.get()})
                        .ConsumeValueOrDie();

                // Clear visit states so that the we can use the evaluate again
                // on the same computation.
                embedded_evaluator.ResetVisitStates();

                result_val = computed_result->Get<ReturnT>({});
              });

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
    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    auto func = [&](tensorflow::gtl::ArraySlice<int64> out_index) {
      DimensionVector operand_index(rank);
      for (int64 i = 0; i < rank; ++i) {
        operand_index[i] =
            slice->slice_starts(i) + out_index[i] * slice->slice_strides(i);
      }
      return operand_literal.Get<ReturnT>(operand_index);
    };

    auto result = LiteralUtil::CreateFromDimensions(
        shape.element_type(), AsInt64Slice(shape.dimensions()));
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(func));
    parent_->evaluated_[slice] = std::move(result);
    return Status::OK();
  }

  // Enable CLZ only for int32, uint32, int64 and uint64.
  template <
      typename NativeT,
      typename std::enable_if<
          (std::is_floating_point<NativeT>::value ||
           std::is_integral<NativeT>::value || is_complex_t<NativeT>::value) &&
          !(std::is_same<NativeT, uint32>::value ||
            std::is_same<NativeT, int32>::value ||
            std::is_same<NativeT, int64>::value ||
            std::is_same<NativeT, uint64>::value)>::type* = nullptr>
  Status HandleClz(HloInstruction* clz) {
    return InvalidArgument("Unsupported type for Clz");
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_same<NativeT, uint32>::value ||
                std::is_same<NativeT, int32>::value>::type* = nullptr>
  Status HandleClz(HloInstruction* clz) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[clz],
                        ElementWiseUnaryOp(clz, [](ElementwiseT elem_operand) {
                          return 31 - tensorflow::Log2Floor(elem_operand);
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_same<NativeT, uint64>::value ||
                std::is_same<NativeT, int64>::value>::type* = nullptr>
  Status HandleClz(HloInstruction* clz) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[clz],
                        ElementWiseUnaryOp(clz, [](ElementwiseT elem_operand) {
                          return 63 - tensorflow::Log2Floor64(elem_operand);
                        }));
    return Status::OK();
  }

  Status HandleClz(HloInstruction* clz) override {
    return HandleClz<ElementwiseT>(clz);
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleSin(HloInstruction* sin) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sin],
                        ElementWiseUnaryOp(sin, [](ElementwiseT elem_operand) {
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
    return HandleSin<ElementwiseT>(sin);
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleCos(HloInstruction* cos) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[cos],
                        ElementWiseUnaryOp(cos, [](ElementwiseT elem_operand) {
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
    return HandleCos<ElementwiseT>(cos);
  }

  template <typename NativeT, typename std::enable_if<std::is_same<
                                  float, NativeT>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[reduce_precision],
        ElementWiseUnaryOp(reduce_precision, [reduce_precision](
                                                 ElementwiseT elem) {
          uint32_t value_as_int = tensorflow::bit_cast<uint32_t>(elem);
          const uint32_t mantissa_bits = reduce_precision->mantissa_bits();
          const uint32_t exponent_bits = reduce_precision->exponent_bits();

          // Code is based on the CPU/GPU implementation in LLVM-emitting code.
          //
          // Bits in float type:
          //   mantissa : bits [0:22]
          //   exponent : bits [23:30]
          //   sign     : bits [31]
          if (mantissa_bits < 23) {
            const uint32_t last_mantissa_bit_mask = 1u << (23 - mantissa_bits);

            // Compute rounding bias for round-to-nearest with ties to even.
            // This is equal to a base value of 0111... plus one bit if the last
            // remaining mantissa bit is 1.
            const uint32_t base_rounding_bias =
                (last_mantissa_bit_mask >> 1) - 1;
            const uint32_t x_last_mantissa_bit =
                (value_as_int & last_mantissa_bit_mask) >> (23 - mantissa_bits);
            const uint32_t x_rounding_bias =
                x_last_mantissa_bit + base_rounding_bias;

            // Add rounding bias, and mask out truncated bits.  Note that the
            // case where adding the rounding bias overflows into the exponent
            // bits is correct; the non-masked mantissa bits will all be zero,
            // and the exponent will be incremented by one.
            const uint32_t truncation_mask = ~(last_mantissa_bit_mask - 1);
            value_as_int = value_as_int + x_rounding_bias;
            value_as_int = value_as_int & truncation_mask;
          }
          if (exponent_bits < 8) {
            // Masks for f32 values.
            const uint32_t f32_sign_bit_mask = 1u << 31;
            const uint32_t f32_exp_bits_mask = 0xffu << 23;

            // An exponent of 2^(n-1)-1 -- that is, 0111... with the zero in the
            // most- significant bit -- is equal to 1.0f for all exponent sizes.
            // Adding 2^(n-1)-1 to this gives us the highest non-infinite
            // exponent for a bit- size of n, and subtracting 2^(n-1)-1 from
            // this gives us the lowest' exponent (corresponding to 0.0f).
            //
            // Thus, the f32 exponent corresponding to the highest non-infinite
            // exponent for a bit size of n is (2^7-1) + 2^(n-1)-1, and the f32
            // exponent corresponding to the lowest exponent for a bit size of n
            // is (2^7-1) - 2^(n-1)-1.
            //
            // Note that we have already checked that exponents_bits >= 1.
            const uint32_t f32_exponent_bias = (1 << 7) - 1;
            const uint32_t reduced_exponent_bias =
                (1 << (exponent_bits - 1)) - 1;
            const uint32_t reduced_max_exponent =
                f32_exponent_bias + reduced_exponent_bias;
            const uint32_t reduced_min_exponent =
                f32_exponent_bias - reduced_exponent_bias;

            // Do we overflow or underflow?
            const uint32_t x_exponent = value_as_int & f32_exp_bits_mask;
            const bool x_overflows = x_exponent > (reduced_max_exponent << 23);
            const bool x_underflows =
                x_exponent <= (reduced_min_exponent << 23);

            // Compute appropriately-signed values of zero and infinity.
            const uint32_t x_signed_zero = value_as_int & f32_sign_bit_mask;
            const uint32_t x_signed_inf = x_signed_zero | f32_exp_bits_mask;

            // Force to zero or infinity if overflow or underflow.  (Note that
            // this truncates all denormal values to zero, rather than rounding
            // them.)
            value_as_int = x_overflows ? x_signed_inf : value_as_int;
            value_as_int = x_underflows ? x_signed_zero : value_as_int;
          }

          float reduced_result = tensorflow::bit_cast<float>(value_as_int);
          if (std::isnan(elem)) {
            reduced_result = mantissa_bits > 0
                                 ? elem
                                 : std::numeric_limits<float>::infinity();
          }
          return reduced_result;
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_same<
                                  double, NativeT>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return InvalidArgument("Double not supported for reduce precision");
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_integral<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return InvalidArgument("Unsupported type for reduce precision");
  }

  Status HandleReducePrecision(HloInstruction* reduce_precision) override {
    return HandleReducePrecision<ElementwiseT>(reduce_precision);
  }

 private:
  // Creates a vector of multipliers which can be used to create a linear index
  // into shape.
  //
  // Given the multidimensional index {i1, ..., iN} and
  // M = MakeDimMultipliers(shape), the corresponding linear index LI is simply
  //
  //   LI = i1 * M[1] + i2 * M[2] + ... + iN * M[N].
  //
  // This lets you calculate LI given the multidimensional indices in any order.
  static DimensionVector MakeDimMultipliers(const Shape& shape) {
    DimensionVector v(ShapeUtil::Rank(shape));
    int64 scale = 1;
    for (auto dim : LayoutUtil::MinorToMajor(shape)) {
      v[dim] = scale;
      scale *= shape.dimensions(dim);
    }
    return v;
  }

  // For one particular placement of a window in a base shape (the placement is
  // represented as `window_count_index`), iterates inside the window.
  // Translates the window index into base index. If the base index is within
  // bound, call `f` with the base index.
  static void IterateThroughWindow(
      const Shape& window_shape, const Window& window, const Shape& base_shape,
      const tensorflow::gtl::ArraySlice<int64>& window_count_index,
      const std::function<void(const std::vector<int64>&)>& f) {
    const int64 rank = ShapeUtil::Rank(base_shape);
    DimensionVector window_index(rank);
    std::fill(window_index.begin(), window_index.end(), 0);
    do {
      std::vector<int64> base_index(rank);
      bool out_of_bound = false;
      for (int64 i = 0; i < rank; ++i) {
        base_index[i] = window_count_index[i] * window.dimensions(i).stride() +
                        window_index[i] - window.dimensions(i).padding_low();
        if (base_index[i] < 0 || base_index[i] >= base_shape.dimensions(i)) {
          out_of_bound = true;
          break;
        }
      }
      if (!out_of_bound) {
        f(base_index);
      }
    } while (IndexUtil::BumpIndices(window_shape, &window_index));
  }

  template <typename IndexT>
  StatusOr<std::unique_ptr<Literal>> DynamicSlice(
      const Literal& operand_literal, const Literal& start_indices_literal,
      const Shape& result_shape) {
    auto start_indices_typed = start_indices_literal.data<IndexT>();
    std::vector<int64> start(start_indices_typed.begin(),
                             start_indices_typed.end());

    // Clamp the start indices so the slice is in-bounds w.r.t the operand.

    // TODO(b/74360564): This is implementation defined behavior, but is
    // currently respected by all implementations. Change this if we ever decide
    // to officially document different behavior.
    for (int64 i = 0; i < start.size(); ++i) {
      start[i] = std::min<int64>(
          std::max(int64{0}, start[i]),
          operand_literal.shape().dimensions(i) - result_shape.dimensions(i));
    }

    std::vector<int64> operand_indices(start.size());
    auto result = MakeUnique<Literal>(result_shape);
    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          for (int64 i = 0; i < operand_indices.size(); ++i) {
            CHECK_GE(multi_index[i] + start[i], 0);
            operand_indices[i] = multi_index[i] + start[i];
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
    auto result = operand_literal.CloneToUnique();
    auto start_indices_typed = start_indices_literal.data<IndexT>();
    const auto rank = ShapeUtil::Rank(result->shape());
    std::vector<int64> start(start_indices_typed.begin(),
                             start_indices_typed.end());
    // Clamp the update start indices so the slice is in-bounds w.r.t the
    // operand.

    // TODO(b/74360564): This is implementation defined behavior, but is
    // currently respected by all implementations. Change this if we ever decide
    // to oficially document different behavior.
    for (int64 i = 0; i < rank; ++i) {
      start[i] = std::min<int64>(
          std::max<int64>(0, start[i]),
          result->shape().dimensions(i) - update_literal.shape().dimensions(i));
    }
    std::vector<int64> result_index(rank, 0);

    auto func = [&](tensorflow::gtl::ArraySlice<int64> update_index) {
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
      const std::function<ElementwiseT(ElementwiseT)>& unary_op) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(instruction->operand(0));
    TF_ASSIGN_OR_RETURN(
        auto result_literal,
        (HloEvaluator::ElementWiseUnaryOpImpl<ReturnT, ReturnT>(
            instruction, ConvertUnaryFunction(unary_op), operand_literal)));

    return std::move(result_literal);
  }

  StatusOr<std::unique_ptr<Literal>> ElementWiseBinaryOp(
      HloInstruction* instruction,
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT)>&
          binary_op) {
    const auto shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);

    // TODO(b/35950897, b/27796129): add DCHECK back once implicit broadcast
    // is removed.
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

    auto result = MakeUnique<Literal>(shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return ConvertBinaryFunction(binary_op)(
              lhs_literal.Get<ReturnT>(multi_index),
              rhs_literal.Get<ReturnT>(multi_index));
        }));
    return std::move(result);
  }

  template <typename LhsType, typename RhsType, typename EhsType>
  StatusOr<std::unique_ptr<Literal>> ElementwiseTernaryOp(
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

    auto result = MakeUnique<Literal>(shape);

    TF_RETURN_IF_ERROR(result->Populate<ReturnT>(
        [&](tensorflow::gtl::ArraySlice<int64> multi_index) {
          return ternary_op(lhs_literal.Get<LhsType>(multi_index),
                            rhs_literal.Get<RhsType>(multi_index),
                            ehs_literal.Get<EhsType>(multi_index));
        }));

    return std::move(result);
  }

  template <typename NativeT>
  static bool IsShiftOutOfBounds(NativeT rhs) {
    typedef typename std::make_unsigned<NativeT>::type UnsignedT;
    UnsignedT lhs_size_unsigned = sizeof(NativeT) * CHAR_BIT;
    UnsignedT rhs_unsigned = static_cast<UnsignedT>(rhs);
    return rhs_unsigned >= lhs_size_unsigned;
  }

  HloEvaluator* parent_;
};

// These extern templates prevent users of this class from implicitly
// instantiating it.  We explicitly instantiate this class in the various
// hlo_evaluator_typed_visitor*.cc files.
extern template class HloEvaluatorTypedVisitor<bool>;
extern template class HloEvaluatorTypedVisitor<uint8>;
extern template class HloEvaluatorTypedVisitor<uint32>;
extern template class HloEvaluatorTypedVisitor<uint64>;
extern template class HloEvaluatorTypedVisitor<int8>;
extern template class HloEvaluatorTypedVisitor<int32>;
extern template class HloEvaluatorTypedVisitor<int64>;
extern template class HloEvaluatorTypedVisitor<Eigen::half, float>;
extern template class HloEvaluatorTypedVisitor<float>;
extern template class HloEvaluatorTypedVisitor<double>;
extern template class HloEvaluatorTypedVisitor<complex64>;
extern template class HloEvaluatorTypedVisitor<bfloat16, float>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_TYPED_VISITOR_H_
