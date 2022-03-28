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

#include <bitset>
#include <cmath>
#include <limits>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/meta/type_traits.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {

// TODO(b/79274244): We'd like these type traits to live inside of
// HloEvaluatorTypedVisitor so they don't pollute namespace xla, but that
// crashes clang in the frontend.
//
// Anyway this is relatively safe as-is because hlo_evaluator_typed_visitor.h is
// a "private" header that's not exposed outside of hlo_evaluator.cc.
//
// Not using an alias template to work around MSVC 14.00 bug.
template <typename T>
struct is_complex_t : absl::disjunction<std::is_same<T, complex64>,
                                        std::is_same<T, complex128>> {};

namespace detail {
template <typename T>
using unsigned_promoted_type_t =
    std::make_unsigned_t<decltype(std::declval<T>() + std::declval<T>())>;
}

// ToArithmeticSafeType(T t):
//  - converts `t` to an unsigned integer at least as wide as `int` if T is an
//    integer, and
//  - otherwise returns `t` unchanged.
//
// It's UB in C++ to under/overflow a signed integer, so we wrap all arithmetic
// in this type to force 2's complement behavior.
template <typename T,
          typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
detail::unsigned_promoted_type_t<T> ToArithmeticSafeType(T t) {
  return static_cast<detail::unsigned_promoted_type_t<T>>(t);
}
template <typename T,
          typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
T ToArithmeticSafeType(T t) {
  return std::move(t);
}

// UintWithSize<N> gets an unsigned integer with the given size in bytes.
template <size_t Bytes>
struct UintWithSize {};

template <>
struct UintWithSize<1> {
  using type = uint8_t;
};

template <>
struct UintWithSize<2> {
  using type = uint16_t;
};

template <>
struct UintWithSize<4> {
  using type = uint32_t;
};

template <>
struct UintWithSize<8> {
  using type = uint64_t;
};

// Templated DfsHloVisitor for use by HloEvaluator.
//
// Typically ReturnT here indicates the resulting literal type of each evaluated
// Handle* method of a TypedVisitor.  There are however a few exceptions to this
// rule, notably:
// - HandleCompare and HandleIsFinite: where the resulting literal type is
//   always boolean.
// - HandleImag and HandleReal: where the resulting literal type is always float
//   and the operand is always complex, or real in the case of HandleReal.
// These operations are handled outside of the parent HloEvaluator handlers
// instead of from within TypedVisitor.
//
// Type params:
//   - ReturnT: The type of input and output of each operation.
//   - ElementwiseT: The type in which internal computation are done.
//
// This is logically a private part of HloEvaluator.  It lives in this header
// file rather than in hlo_evaluator.cc because we use extern templates and a
// bunch of independent cc files to speed up compiling the many instantiations
// of this class.
template <typename ReturnT, typename ElementwiseT = ReturnT>
class HloEvaluatorTypedVisitor : public DfsHloVisitorWithDefault {
 private:
  Status UnsupportedTypeError(HloInstruction* instruction) {
    return InvalidArgument(
        "Unsupported type for %s: %s", HloOpcodeString(instruction->opcode()),
        PrimitiveType_Name(instruction->shape().element_type()));
  }

  // Get the value in the given literal static_cast as a double.
  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  double GetAsDouble(const Literal& literal,
                     absl::Span<const int64_t> input_index) {
    return static_cast<double>(literal.Get<NativeT>(input_index));
  }

  // Specialization for complex types. In this case it is not possible to
  // static_cast value to a double so just CHECK fail. This method is not used
  // at run-time, but must be available at compile-time to keep the compiler
  // happy.
  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  double GetAsDouble(const Literal& literal,
                     absl::Span<const int64_t> input_index) {
    LOG(FATAL) << "Trying to get complex literal as double: "
               << literal.ToString();
  }

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
                         HloOpcodeString(hlo_instruction->opcode()));
  }

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
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(abs->operand(0));
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[abs],
        (HloEvaluator::ElementWiseUnaryOpImpl<typename NativeT::value_type,
                                              NativeT>(
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
    } else if (abs->operand(0)->shape().element_type() == C128) {
      return HandleAbs<complex128>(abs);
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
    return UnsupportedTypeError(round);
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
    return UnsupportedTypeError(ceil);
  }

  Status HandleCeil(HloInstruction* ceil) override {
    return HandleCeil<ReturnT>(ceil);
  }

  Status HandleConvert(HloInstruction* convert) override {
    const HloInstruction* operand = convert->operand(0);
    TF_RET_CHECK(ShapeUtil::SameDimensions(operand->shape(), convert->shape()));
    TF_ASSIGN_OR_RETURN(Literal result,
                        parent_->GetEvaluatedLiteralFor(operand).Convert(
                            convert->shape().element_type()));
    parent_->evaluated_[convert] = std::move(result);
    return Status::OK();
  }

  Status HandleBitcastConvert(HloInstruction* convert) override {
    const HloInstruction* operand = convert->operand(0);
    TF_ASSIGN_OR_RETURN(Literal result,
                        parent_->GetEvaluatedLiteralFor(operand).BitcastConvert(
                            convert->shape()));

    parent_->evaluated_[convert] = std::move(result);
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
  Status HandleExpm1(HloInstruction* expm1) {
    return UnsupportedTypeError(expm1);
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
    return UnsupportedTypeError(floor);
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
  Status HandleLog1p(HloInstruction* log1p) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[log1p],
        ElementWiseUnaryOp(log1p, [](ElementwiseT elem_operand) {
          return std::log1p(elem_operand);
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleLog1p(HloInstruction* log1p) {
    return UnsupportedTypeError(log1p);
  }

  Status HandleLog1p(HloInstruction* log1p) override {
    return HandleLog1p<ReturnT>(log1p);
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
    return UnsupportedTypeError(not_);
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

  Status HandleLogistic(HloInstruction* logistic) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[logistic],
        ElementWiseUnaryOp(logistic, [](ElementwiseT elem_operand) {
          return static_cast<ElementwiseT>(1) /
                 (static_cast<ElementwiseT>(1) + std::exp(-elem_operand));
        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleSign(HloInstruction* sign) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
                          return (ElementwiseT(0) < elem_operand) -
                                 (elem_operand < ElementwiseT(0));
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_same<NativeT, bfloat16>::value ||
                std::is_same<NativeT, Eigen::half>::value ||
                std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleSign(HloInstruction* sign) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sign],
                        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
                          return std::isnan(elem_operand)
                                     ? elem_operand
                                     : std::copysign(
                                           elem_operand != ElementwiseT(0),
                                           elem_operand);
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

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[atan2],
        ElementWiseBinaryOp(atan2, [](ElementwiseT y, ElementwiseT x) {
          // atan2(y,x) = -i * log((x + i * y)/sqrt(x**2+y**2))
          auto i = ElementwiseT(0.0, 1.0);
          return (-i) * (std::log((x + i * y) / std::sqrt(x * x + y * y)));
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!std::is_floating_point<NativeT>::value &&
                              !is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    return UnsupportedTypeError(atan2);
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

  Status HandleMultiply(HloInstruction* multiply) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(
            multiply, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              return ElementwiseT(ToArithmeticSafeType(lhs_elem) *
                                  ToArithmeticSafeType(rhs_elem));
            }));
    return Status::OK();
  }

  Status HandleSubtract(HloInstruction* subtract) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(
            subtract, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              return ElementwiseT(ToArithmeticSafeType(lhs_elem) -
                                  ToArithmeticSafeType(rhs_elem));
            }));
    return Status::OK();
  }

  Status HandleAdd(HloInstruction* add) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[add],
                        ElementWiseBinaryOp(add, [](ElementwiseT lhs_elem,
                                                    ElementwiseT rhs_elem) {
                          return ElementwiseT(ToArithmeticSafeType(lhs_elem) +
                                              ToArithmeticSafeType(rhs_elem));
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_floating_point<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleDivide(HloInstruction* divide) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[divide],
                        ElementWiseBinaryOp(divide, [](ElementwiseT lhs_elem,
                                                       ElementwiseT rhs_elem) {
                          return lhs_elem / rhs_elem;
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_signed<NativeT>::value &&
                                    std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleDivide(HloInstruction* divide) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[divide],
        ElementWiseBinaryOp(
            divide,
            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) -> ElementwiseT {
              if (rhs_elem == 0) {
                return static_cast<ElementwiseT>(-1);
              }
              if (rhs_elem == -1 &&
                  lhs_elem == std::numeric_limits<ElementwiseT>::min()) {
                return lhs_elem;
              }
              return lhs_elem / rhs_elem;
            }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_unsigned<NativeT>::value>::type* =
                nullptr>
  Status HandleDivide(HloInstruction* divide) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[divide],
                        ElementWiseBinaryOp(divide, [](ElementwiseT lhs_elem,
                                                       ElementwiseT rhs_elem) {
                          return rhs_elem == 0
                                     ? std::numeric_limits<ElementwiseT>::max()
                                     : (lhs_elem / rhs_elem);
                        }));
    return Status::OK();
  }

  Status HandleDivide(HloInstruction* divide) override {
    return HandleDivide<ElementwiseT>(divide);
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
    return UnsupportedTypeError(maximum);
  }

  Status HandleMaximum(HloInstruction* maximum) override {
    return HandleMaximum<ElementwiseT>(maximum);
  }

  template <typename NativeT,
            typename std::enable_if<std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleMinimum(HloInstruction* minimum) {
    VLOG(2) << "Evaluating minimum\n";
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
    return UnsupportedTypeError(minimum);
  }

  Status HandleMinimum(HloInstruction* minimum) override {
    return HandleMinimum<ElementwiseT>(minimum);
  }

  Status HandlePower(HloInstruction* power) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[power],
        ElementWiseBinaryOp(
            power, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
              return lhs_el == ElementwiseT(0) && rhs_el == ElementwiseT(0)
                         ? static_cast<ElementwiseT>(1)
                         : std::pow(lhs_el, rhs_el);
            }));
    return Status::OK();
  }

  Status HandleSqrt(HloInstruction* sqrt) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sqrt],
                        ElementWiseUnaryOp(sqrt, [](ElementwiseT elem_operand) {
                          return std::sqrt(elem_operand);
                        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCbrt(HloInstruction* cbrt) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[cbrt],
        ElementWiseUnaryOp(cbrt, [](ElementwiseT elem_operand) -> ElementwiseT {
          return std::pow(elem_operand, static_cast<ElementwiseT>(1.0 / 3.0));
          return elem_operand.real() < 0
                     ? -std::pow(-elem_operand,
                                 static_cast<ElementwiseT>(1.0 / 3.0))
                     : std::pow(elem_operand,
                                static_cast<ElementwiseT>(1.0 / 3.0));
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<!is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleCbrt(HloInstruction* cbrt) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[cbrt],
                        ElementWiseUnaryOp(cbrt, [](ElementwiseT elem_operand) {
                          return std::cbrt(elem_operand);
                        }));
    return Status::OK();
  }

  Status HandleCbrt(HloInstruction* cbrt) override {
    return HandleCbrt<ElementwiseT>(cbrt);
  }

  Status HandleRsqrt(HloInstruction* rsqrt) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[rsqrt],
        ElementWiseUnaryOp(rsqrt, [](ElementwiseT elem_operand) {
          return static_cast<ElementwiseT>(1) / std::sqrt(elem_operand);
        }));
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<std::is_floating_point<
                                  NativeT>::value>::type* = nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[remainder],
                        ElementWiseBinaryOp(remainder, [](ElementwiseT lhs_el,
                                                          ElementwiseT rhs_el) {
                          return std::fmod(lhs_el, rhs_el);
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_unsigned<NativeT>::value>::type* =
                nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[remainder],
                        ElementWiseBinaryOp(remainder, [](ElementwiseT lhs_el,
                                                          ElementwiseT rhs_el) {
                          return rhs_el == 0 ? lhs_el : (lhs_el % rhs_el);
                        }));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<std::is_signed<NativeT>::value &&
                                    std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[remainder],
        ElementWiseBinaryOp(
            remainder,
            [](ElementwiseT lhs_el, ElementwiseT rhs_el) -> ElementwiseT {
              if (rhs_el == 0) {
                return lhs_el;
              }
              if (rhs_el == -1 &&
                  lhs_el == std::numeric_limits<ElementwiseT>::min()) {
                return 0;
              }
              return lhs_el % rhs_el;
            }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleRemainder(HloInstruction* remainder) {
    return UnsupportedTypeError(remainder);
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
    return UnsupportedTypeError(and_);
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleAnd(HloInstruction* and_) {
    return UnsupportedTypeError(and_);
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
    return UnsupportedTypeError(or_);
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
    return UnsupportedTypeError(xor_);
  }

  template <
      typename NativeT,
      typename std::enable_if<is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleXor(HloInstruction* xor_) {
    return UnsupportedTypeError(xor_);
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
  Status HandleShiftLeft(HloInstruction* shift) {
    return UnsupportedTypeError(shift);
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
  Status HandleShiftRightArithmetic(HloInstruction* shift) {
    return UnsupportedTypeError(shift);
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
  Status HandleShiftRightLogical(HloInstruction* shift) {
    return UnsupportedTypeError(shift);
  }

  Status HandleShiftRightLogical(HloInstruction* shrl) override {
    return HandleShiftRightLogical<ElementwiseT>(shrl);
  }

  // Special case for integral type due to MSVC's std::isnan being unable to
  // handle integral type.
  template <typename NativeT,
            typename std::enable_if<!is_complex_t<NativeT>::value &&
                                    std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleClamp(HloInstruction* clamp) {
    std::function<ElementwiseT(ElementwiseT, ElementwiseT, ElementwiseT)>
        clamp_op = [](ElementwiseT low, ElementwiseT value, ElementwiseT high) {
          return static_cast<ElementwiseT>(
              std::min(high, std::max(value, low)));
        };
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[clamp],
        ElementwiseTernaryOp(clamp,
                             std::move(ConvertTernaryFunction(clamp_op))));
    return Status::OK();
  }

  template <typename NativeT,
            typename std::enable_if<!is_complex_t<NativeT>::value &&
                                    !std::is_integral<NativeT>::value>::type* =
                nullptr>
  Status HandleClamp(HloInstruction* clamp) {
    std::function<ElementwiseT(ElementwiseT, ElementwiseT, ElementwiseT)>
        clamp_op = [](ElementwiseT low, ElementwiseT value, ElementwiseT high) {
          if (std::isnan(low) || std::isnan(high) || std::isnan(value)) {
            return static_cast<ElementwiseT>(NAN);
          }
          return static_cast<ElementwiseT>(
              std::min<NativeT>(high, std::max<NativeT>(value, low)));
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
  Status HandleClamp(HloInstruction* clamp) {
    return UnsupportedTypeError(clamp);
  }

  Status HandleClamp(HloInstruction* clamp) override {
    return HandleClamp<ElementwiseT>(clamp);
  }

  Status HandleSelect(HloInstruction* select) override {
    CHECK(!ShapeUtil::IsScalar(select->operand(0)->shape()));
    CHECK(select->shape().IsArray());
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
    Literal result(result_shape);

    TF_RETURN_IF_ERROR(
        result.Populate<ReturnT>([&](absl::Span<const int64_t> out_index) {
          std::vector<int64_t> from_index(out_index.begin(), out_index.end());
          for (const int64_t dim : reverse_dimensions) {
            from_index[dim] = result_shape.dimensions(dim) - 1 - out_index[dim];
          }
          return operand_literal.Get<ReturnT>(from_index);
        }));

    parent_->evaluated_[reverse] = std::move(result);
    return Status::OK();
  }

  Status HandleConvolutionWithLiterals(HloInstruction* conv,
                                       const Literal& lhs_literal,
                                       const Literal& rhs_literal) {
    const auto& window = conv->window();
    const Shape& result_shape = conv->shape();
    const Shape& lhs_shape = lhs_literal.shape();
    const Shape& rhs_shape = rhs_literal.shape();

    TF_CHECK_OK(ShapeUtil::ValidateShape(lhs_shape));
    TF_CHECK_OK(ShapeUtil::ValidateShape(rhs_shape));
    CHECK(lhs_shape.IsArray());
    CHECK(rhs_shape.IsArray());
    CHECK(ShapeUtil::SameElementType(lhs_shape, rhs_shape));
    CHECK(ShapeUtil::SameElementType(lhs_shape, result_shape));

    const auto& dnums = conv->convolution_dimension_numbers();
    const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
    CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
    CHECK_GE(num_spatial_dims, 0);
    CHECK_EQ(window.dimensions_size(), num_spatial_dims);

    std::vector<int64_t> window_dimension_sizes;
    for (auto i : dnums.kernel_spatial_dimensions()) {
      window_dimension_sizes.push_back(ShapeUtil::GetDimension(rhs_shape, i));
    }

    const Shape& window_shape =
        ShapeUtil::MakeShape(rhs_shape.element_type(), window_dimension_sizes);

    DimensionVector lhs_dim_multipliers = MakeDimMultipliers(lhs_shape);
    DimensionVector rhs_dim_multipliers = MakeDimMultipliers(rhs_shape);

    auto lhs_literal_data = lhs_literal.data<ReturnT>();
    auto rhs_literal_data = rhs_literal.data<ReturnT>();

    const int64_t feature_group_count = conv->feature_group_count();
    const int64_t batch_group_count = conv->batch_group_count();

    auto func = [&window_shape, &dnums, &lhs_shape, &rhs_shape, &window,
                 &lhs_dim_multipliers, &rhs_dim_multipliers, lhs_literal_data,
                 rhs_literal_data, feature_group_count,
                 batch_group_count](const absl::Span<const int64_t> out_index) {
      // Dimension number applicable for input (lhs).
      const int64_t input_batch_dim = dnums.input_batch_dimension();
      const int64_t input_z_dim = dnums.input_feature_dimension();
      // Dimension number applicable for kernel (rhs).
      const int64_t kernel_input_z_dim = dnums.kernel_input_feature_dimension();
      const int64_t kernel_output_z_dim =
          dnums.kernel_output_feature_dimension();
      // Dimension number applicable for output.
      const int64_t output_batch_dim = dnums.output_batch_dimension();
      const int64_t output_z_dim = dnums.output_feature_dimension();

      const int64_t input_z_size =
          ShapeUtil::GetDimension(lhs_shape, input_z_dim);

      const int64_t input_batch_size =
          ShapeUtil::GetDimension(lhs_shape, input_batch_dim);

      const int64_t batch_group_size = input_batch_size / batch_group_count;

      // The size of an input feature group.
      const int64_t input_feature_group_size =
          input_z_size / feature_group_count;

      const int64_t output_z_size =
          ShapeUtil::GetDimension(rhs_shape, kernel_output_z_dim);
      // The output feature dimension is a concatenation of convolution results
      // from the different groups.
      const int64_t output_feature_group_size =
          output_z_size / feature_group_count;

      // Calculate the group index to which the current output index
      // belongs.
      const int64_t feature_group_index =
          out_index[output_z_dim] / output_feature_group_size;

      const int64_t depthwise_multiplier =
          batch_group_count > 1 ? output_z_size / input_batch_size : 1;
      const int64_t batch_group_index =
          out_index[output_z_dim] / depthwise_multiplier;

      ElementwiseT result_val = static_cast<ElementwiseT>(0);
      DimensionVector rhs_spatial_index(dnums.kernel_spatial_dimensions_size(),
                                        0);

      // Convolve input feature with kernel.
      // The mechanism indexes into the correct LHS (input) and RHS (kernel)
      // locations and accumulates multiplications for a given output index.
      do {
        // Find corresponding spatial dimension index for input (lhs).
        int64_t lhs_linear_spatial_index = 0;
        int64_t rhs_linear_spatial_index = 0;
        for (int64_t ki = 0; ki < rhs_spatial_index.size(); ++ki) {
          // Spatial dimension number for input (lhs) and output.
          const int64_t input_spatial_dim = dnums.input_spatial_dimensions(ki);
          const int64_t output_spatial_dim =
              dnums.output_spatial_dimensions(ki);

          // Calculate lhs (input) index without taking base dilation into
          // account.
          const auto& window_dim = window.dimensions(ki);
          const int64_t undilated_index =
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
          int64_t lhs_spatial_index;
          if (window_dim.base_dilation() > 1) {
            lhs_spatial_index = undilated_index / window_dim.base_dilation();
          } else {
            lhs_spatial_index = undilated_index;
          }

          // Skip if input index is not in bounds.
          if (!(lhs_spatial_index >= 0 &&
                lhs_spatial_index < lhs_shape.dimensions(input_spatial_dim))) {
            goto cnt;
          }

          lhs_linear_spatial_index +=
              lhs_spatial_index * lhs_dim_multipliers[input_spatial_dim];
          rhs_linear_spatial_index +=
              (window_dim.window_reversal()
                   ? ((window_dim.size() - 1) - rhs_spatial_index[ki])
                   : rhs_spatial_index[ki]) *
              rhs_dim_multipliers[dnums.kernel_spatial_dimensions(ki)];
        }

        for (int64_t rhs_iz = 0; rhs_iz < input_feature_group_size; ++rhs_iz) {
          const int64_t iz =
              feature_group_index * input_feature_group_size + rhs_iz;

          int64_t lhs_linear_index = lhs_linear_spatial_index;
          lhs_linear_index += out_index[output_batch_dim] *
                              lhs_dim_multipliers[input_batch_dim];

          // We are scraping only the diagonal elements in the resultant
          // convolution output when batch_group_count is greater than 1,
          // where 1 is the default. No scraping is done in that case.
          // This approach works out automatically for 'groups' in batches
          // with group_size > 1, because we already descend down the batch
          // dimension for the 'output_batch_dim' above.
          lhs_linear_index +=
              ((batch_group_index * batch_group_size) % input_batch_size) *
              lhs_dim_multipliers[input_batch_dim];

          lhs_linear_index += iz * lhs_dim_multipliers[input_z_dim];
          int64_t rhs_linear_index = rhs_linear_spatial_index;

          rhs_linear_index += out_index[output_z_dim] *
                              rhs_dim_multipliers[kernel_output_z_dim];
          rhs_linear_index += rhs_iz * rhs_dim_multipliers[kernel_input_z_dim];

          result_val +=
              static_cast<ElementwiseT>(lhs_literal_data[lhs_linear_index]) *
              static_cast<ElementwiseT>(rhs_literal_data[rhs_linear_index]);
        }
      cnt : {}
      } while (IndexUtil::BumpIndices(window_shape,
                                      absl::MakeSpan(rhs_spatial_index)));

      return static_cast<ReturnT>(result_val);
    };

    Literal result(result_shape);
    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(func));

    parent_->evaluated_[conv] = std::move(result);
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
    CHECK(lhs_shape.IsArray());
    CHECK(rhs_shape.IsArray());

    const auto& dnums = conv->convolution_dimension_numbers();
    const int64_t num_spatial_dims = dnums.output_spatial_dimensions_size();
    CHECK_EQ(num_spatial_dims, dnums.input_spatial_dimensions_size());
    CHECK_EQ(num_spatial_dims, dnums.kernel_spatial_dimensions_size());
    CHECK_GE(num_spatial_dims, 0);
    CHECK_EQ(window.dimensions_size(), num_spatial_dims);

    const auto lhs_rank = lhs_shape.rank();
    const auto rhs_rank = rhs_shape.rank();

    CHECK_EQ(num_spatial_dims + 2, lhs_rank);
    CHECK_EQ(num_spatial_dims + 2, rhs_rank);

    TF_ASSIGN_OR_RETURN(
        auto inferred_return_shape,
        ShapeInference::InferConvolveShape(
            lhs_shape, rhs_shape, conv->feature_group_count(),
            conv->batch_group_count(), window, dnums,
            /*preferred_element_type=*/conv->shape().element_type()));
    CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);
    const bool lhs_same = ShapeUtil::SameElementType(lhs_shape, result_shape);
    const bool rhs_same = ShapeUtil::SameElementType(rhs_shape, result_shape);
    if (rhs_same && lhs_same) {
      return HandleConvolutionWithLiterals(conv, lhs_literal, rhs_literal);
    }
    if (rhs_same) {
      return HandleConvolutionWithLiterals(
          conv, lhs_literal.Convert(result_shape.element_type()).ValueOrDie(),
          rhs_literal);
    }
    if (lhs_same) {
      return HandleConvolutionWithLiterals(
          conv, lhs_literal,
          rhs_literal.Convert(result_shape.element_type()).ValueOrDie());
    }
    return HandleConvolutionWithLiterals(
        conv, lhs_literal.Convert(result_shape.element_type()).ValueOrDie(),
        rhs_literal.Convert(result_shape.element_type()).ValueOrDie());
  }

  Status HandleDot(HloInstruction* dot) override {
    if (dot->dot_dimension_numbers().rhs_contracting_dimensions_size() == 1 &&
        parent_->use_fast_path_ &&
        ShapeUtil::SameElementType(dot->operand(0)->shape(), dot->shape()) &&
        ShapeUtil::SameElementType(dot->operand(1)->shape(), dot->shape())) {
      return HandleDot<ElementwiseT>(dot);
    }
    return HandleDotSlowPath(dot);
  }

  template <typename NativeT, typename std::enable_if<std::is_same<
                                  NativeT, float>::value>::type* = nullptr>
  Status HandleDot(HloInstruction* dot) {
    const HloInstruction* lhs = dot->operand(0);
    const HloInstruction* rhs = dot->operand(1);
    CHECK(dot->shape().IsArray());
    CHECK(lhs->shape().IsArray());
    CHECK(rhs->shape().IsArray());

    const auto& dnums = dot->dot_dimension_numbers();

    const int64_t lhs_rank = lhs->shape().rank();
    const int64_t rhs_rank = rhs->shape().rank();

    CHECK(ShapeUtil::SameElementType(lhs->shape(), rhs->shape()));
    CHECK(ShapeUtil::SameElementType(lhs->shape(), dot->shape()));

    // There must be 1 and only 1 Contracting dimension for lhs and rhs.
    const int64_t lhs_contracting_dimension =
        dnums.lhs_contracting_dimensions(0);
    const int64_t rhs_contracting_dimension =
        dnums.rhs_contracting_dimensions(0);
    // Contracted dimension sizes must be the same.
    CHECK_EQ(lhs->shape().dimensions(lhs_contracting_dimension),
             rhs->shape().dimensions(rhs_contracting_dimension))
        << "lhs contracted dimension: "
        << lhs->shape().dimensions(lhs_contracting_dimension)
        << " rhs contracted dimension: "
        << rhs->shape().dimensions(rhs_contracting_dimension);

    // The fast path is for a simple rank 2 dot with default layout operands.
    if (lhs_rank != 2 || rhs_rank != 2 || lhs_contracting_dimension != 1 ||
        rhs_contracting_dimension != 0 ||
        !LayoutUtil::Equal(lhs->shape().layout(),
                           LayoutUtil::GetDefaultLayoutForR2()) ||
        !LayoutUtil::Equal(rhs->shape().layout(),
                           LayoutUtil::GetDefaultLayoutForR2()) ||
        !LayoutUtil::Equal(dot->shape().layout(),
                           LayoutUtil::GetDefaultLayoutForR2())) {
      return HandleDotSlowPath(dot);
    }

    const PrimitiveType native_ty =
        primitive_util::NativeToPrimitiveType<NativeT>();
    Literal lhs_literal =
        parent_->GetEvaluatedLiteralFor(lhs).Convert(native_ty).ValueOrDie();
    Literal rhs_literal =
        parent_->GetEvaluatedLiteralFor(rhs).Convert(native_ty).ValueOrDie();
    const int64_t contracted_dimension_size =
        lhs->shape().dimensions(lhs_contracting_dimension);
    Array2D<NativeT> lhs_array(lhs->shape().dimensions(0),
                               contracted_dimension_size);
    lhs_array.SetValues(lhs_literal.data<NativeT>());
    Array2D<NativeT> rhs_array(contracted_dimension_size,
                               rhs->shape().dimensions(1));
    rhs_array.SetValues(rhs_literal.data<NativeT>());
    std::unique_ptr<Array2D<NativeT>> result_array =
        HloEvaluator::MatmulArray2D(lhs_array, rhs_array);
    Literal result(ShapeUtil::MakeShape(native_ty, dot->shape().dimensions()));
    result.PopulateR2FromArray2D(*result_array);
    parent_->evaluated_[dot] =
        std::move(result).Convert(dot->shape().element_type()).ValueOrDie();
    return Status::OK();
  }

  template <typename NativeT, typename std::enable_if<!std::is_same<
                                  NativeT, float>::value>::type* = nullptr>
  Status HandleDot(HloInstruction* dot) {
    return HandleDotSlowPath(dot);
  }

  Status HandleDotSlowPathWithLiterals(HloInstruction* dot,
                                       const Literal& lhs_literal,
                                       const Literal& rhs_literal) {
    const auto& dnums = dot->dot_dimension_numbers();

    const auto lhs_rank = lhs_literal.shape().rank();
    const auto rhs_rank = rhs_literal.shape().rank();

    CHECK(ShapeUtil::SameElementType(lhs_literal.shape(), rhs_literal.shape()));
    CHECK(ShapeUtil::SameElementType(lhs_literal.shape(), dot->shape()));

    CHECK_EQ(dnums.lhs_batch_dimensions_size(),
             dnums.rhs_batch_dimensions_size());

    DimensionVector lhs_non_contracting_dims;
    DimensionVector rhs_non_contracting_dims;
    for (int64_t i = 0; i < lhs_rank; i++) {
      if (!absl::c_linear_search(dnums.lhs_contracting_dimensions(), i) &&
          !absl::c_linear_search(dnums.lhs_batch_dimensions(), i)) {
        lhs_non_contracting_dims.push_back(i);
      }
    }
    for (int64_t i = 0; i < rhs_rank; i++) {
      if (!absl::c_linear_search(dnums.rhs_contracting_dimensions(), i) &&
          !absl::c_linear_search(dnums.rhs_batch_dimensions(), i)) {
        rhs_non_contracting_dims.push_back(i);
      }
    }

    absl::InlinedVector<int64_t, InlineRank()> contracting_dim_sizes;
    contracting_dim_sizes.reserve(dnums.lhs_contracting_dimensions_size());
    DimensionVector lhs_contracting_dims;
    DimensionVector rhs_contracting_dims;
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); ++i) {
      const int64_t lhs_dnum = dnums.lhs_contracting_dimensions(i);
      const int64_t rhs_dnum = dnums.rhs_contracting_dimensions(i);
      lhs_contracting_dims.push_back(lhs_dnum);
      rhs_contracting_dims.push_back(rhs_dnum);
      const int64_t dim_size = lhs_literal.shape().dimensions(lhs_dnum);
      contracting_dim_sizes.push_back(dim_size);
    }
    const int64_t total_contraction_size = Product(contracting_dim_sizes);
    Literal result(dot->shape());
    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&](absl::Span<const int64_t> result_index) {
          // Locations in LHS and RHS that we read from.
          DimensionVector lhs_index(lhs_rank);
          DimensionVector rhs_index(rhs_rank);

          // First come the batch dimensions.
          int64_t idx = 0;
          for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); i++) {
            lhs_index[dnums.lhs_batch_dimensions(i)] = result_index[idx];
            rhs_index[dnums.rhs_batch_dimensions(i)] = result_index[idx];
            idx++;
          }

          // Next we have non-contracting dimensions, if any.
          for (int64_t i = 0; i < lhs_non_contracting_dims.size(); i++) {
            lhs_index[lhs_non_contracting_dims[i]] = result_index[idx++];
          }
          for (int64_t i = 0; i < rhs_non_contracting_dims.size(); i++) {
            rhs_index[rhs_non_contracting_dims[i]] = result_index[idx++];
          }

          // Accumulate resulting product along the contracting dimensions.
          ElementwiseT result_val = static_cast<ElementwiseT>(0);
          for (int64_t k = 0; k < total_contraction_size; k++) {
            ElementwiseT lhs_val(lhs_literal.Get<ReturnT>(lhs_index));
            ElementwiseT rhs_val(rhs_literal.Get<ReturnT>(rhs_index));
            result_val +=
                ToArithmeticSafeType(lhs_val) * ToArithmeticSafeType(rhs_val);

            // If there are no contracting dimensions, do not try to count down
            // from -1 to 0; that's an infinite loop.
            if (!contracting_dim_sizes.empty()) {
              for (int64_t i = contracting_dim_sizes.size() - 1; i >= 0; --i) {
                lhs_index[lhs_contracting_dims[i]]++;
                rhs_index[rhs_contracting_dims[i]]++;
                if (lhs_index[lhs_contracting_dims[i]] !=
                    contracting_dim_sizes[i]) {
                  break;
                }
                lhs_index[lhs_contracting_dims[i]] = 0;
                rhs_index[rhs_contracting_dims[i]] = 0;
              }
            }
          }

          return static_cast<ReturnT>(result_val);
        }));

    parent_->evaluated_[dot] = std::move(result);
    return Status::OK();
  }

  Status HandleDotSlowPath(HloInstruction* dot) {
    auto lhs = dot->operand(0);
    auto rhs = dot->operand(1);
    CHECK(dot->shape().IsArray());
    CHECK(lhs->shape().IsArray());
    CHECK(rhs->shape().IsArray());
    const bool lhs_same =
        ShapeUtil::SameElementType(lhs->shape(), dot->shape());
    const bool rhs_same =
        ShapeUtil::SameElementType(rhs->shape(), dot->shape());
    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);
    if (lhs_same && rhs_same) {
      return HandleDotSlowPathWithLiterals(dot, lhs_literal, rhs_literal);
    }
    if (lhs_same) {
      return HandleDotSlowPathWithLiterals(
          dot, lhs_literal,
          rhs_literal.Convert(dot->shape().element_type()).ValueOrDie());
    }
    if (rhs_same) {
      return HandleDotSlowPathWithLiterals(
          dot, lhs_literal.Convert(dot->shape().element_type()).ValueOrDie(),
          rhs_literal);
    }
    return HandleDotSlowPathWithLiterals(
        dot, lhs_literal.Convert(dot->shape().element_type()).ValueOrDie(),
        rhs_literal.Convert(dot->shape().element_type()).ValueOrDie());
  }

  Status HandlePad(HloInstruction* pad) override {
    CHECK(pad->operand(0)->shape().IsArray());
    // Padding value must be scalar.
    CHECK(ShapeUtil::IsScalar(pad->operand(1)->shape()));
    CHECK_EQ(pad->operand(0)->shape().rank(),
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
    Literal result(pad->shape());
    TF_RETURN_IF_ERROR(result.Populate<ReturnT>(
        [&scalar](absl::Span<const int64_t> multi_index) { return scalar; }));

    const Literal& evaluated_operand =
        parent_->GetEvaluatedLiteralFor(pad->operand(0));

    std::vector<int64_t> input_index(evaluated_operand.shape().rank(), 0);
    std::vector<int64_t> target_index(result.shape().rank(), 0);

    // Loop through each element of the operand, assign them to the
    // corresponding index of the resulting padded literal.
    const PaddingConfig& pad_config = pad->padding_config();

    auto func = [&](absl::Span<const int64_t> input_index) {
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
      result.Set<ReturnT>(target_index,
                          evaluated_operand.Get<ReturnT>(input_index));
      return true;
    };

    std::vector<int64_t> zero_base(evaluated_operand.shape().dimensions_size(),
                                   0);
    std::vector<int64_t> step(evaluated_operand.shape().dimensions_size(), 1);

    ShapeUtil::ForEachIndex(evaluated_operand.shape(), zero_base,
                            evaluated_operand.shape().dimensions(), step, func);

    parent_->evaluated_[pad] = std::move(result);
    return Status::OK();
  }

  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override {
    auto operand = dynamic_slice->operand(0);
    auto start_indices = dynamic_slice->operand(1);
    auto result_shape = dynamic_slice->shape();
    TF_ASSIGN_OR_RETURN(
        auto inferred_return_shape,
        ShapeInference::InferDynamicSliceShape(
            operand->shape(),
            Cast<HloDynamicSliceInstruction>(dynamic_slice)->index_shapes(),
            dynamic_slice->dynamic_slice_sizes()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    TF_RET_CHECK(
        primitive_util::IsIntegralType(start_indices->shape().element_type()));

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);

    switch (start_indices->shape().element_type()) {
      case S32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<int32_t>(
                operand_literal,
                absl::MakeConstSpan(dynamic_slice->operands()).subspan(1),
                result_shape));
      } break;
      case S64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<int64_t>(
                operand_literal,
                absl::MakeConstSpan(dynamic_slice->operands()).subspan(1),
                result_shape));
      } break;
      case U32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<uint32_t>(
                operand_literal,
                absl::MakeConstSpan(dynamic_slice->operands()).subspan(1),
                result_shape));
      } break;
      case U64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_slice],
            DynamicSlice<uint64_t>(
                operand_literal,
                absl::MakeConstSpan(dynamic_slice->operands()).subspan(1),
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
            operand->shape(), update->shape(),
            Cast<HloDynamicUpdateSliceInstruction>(dynamic_update_slice)
                ->index_shapes()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(result_shape)
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    TF_RET_CHECK(
        primitive_util::IsIntegralType(start_indices->shape().element_type()));
    TF_RET_CHECK(ShapeUtil::Compatible(result_shape, operand->shape()));

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& update_literal = parent_->GetEvaluatedLiteralFor(update);

    switch (start_indices->shape().element_type()) {
      case S32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<int32_t>(
                operand_literal, update_literal,
                absl::MakeConstSpan(dynamic_update_slice->operands())
                    .subspan(2)));
      } break;
      case S64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<int64_t>(
                operand_literal, update_literal,
                absl::MakeConstSpan(dynamic_update_slice->operands())
                    .subspan(2)));
      } break;
      case U32: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<uint32_t>(
                operand_literal, update_literal,
                absl::MakeConstSpan(dynamic_update_slice->operands())
                    .subspan(2)));
      } break;
      case U64: {
        TF_ASSIGN_OR_RETURN(
            parent_->evaluated_[dynamic_update_slice],
            DynamicUpdateSlice<uint64_t>(
                operand_literal, update_literal,
                absl::MakeConstSpan(dynamic_update_slice->operands())
                    .subspan(2)));
      } break;
      default:
        LOG(FATAL) << "HandleDynamicUpdateSlice: unhandled primitive type for "
                      "start_indices: "
                   << PrimitiveType_Name(start_indices->shape().element_type());
    }

    return Status::OK();
  }

  template <typename NativeT>
  StatusOr<Literal> MapImpl(HloInstruction* map) {
    auto operands = map->operands();
    HloComputation* computation = map->to_apply();

    Literal result(map->shape());

    HloEvaluator embedded_evaluator(parent_->max_loop_iterations_);
    TF_RETURN_IF_ERROR(
        result.Populate<ReturnT>([&](absl::Span<const int64_t> multi_index) {
          std::vector<Literal> arg_literals;
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

          Literal computed_result =
              embedded_evaluator.Evaluate(*computation, arg_literals)
                  .ConsumeValueOrDie();
          // Clear visit states so that the we can use the evaluate again on
          // the same computation.
          embedded_evaluator.ResetVisitStates();

          return computed_result.Get<ReturnT>({});
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
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint8_t>(map));
        break;
      }
      case U16: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint16_t>(map));
        break;
      }
      case U32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint32_t>(map));
        break;
      }
      case U64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<uint64_t>(map));
        break;
      }
      case S8: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int8_t>(map));
        break;
      }
      case S16: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int16_t>(map));
        break;
      }
      case S32: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int32_t>(map));
        break;
      }
      case S64: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<int64_t>(map));
        break;
      }
      case F16: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map],
                            MapImpl<Eigen::half>(map));
        break;
      }
      case BF16: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<bfloat16>(map));
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
      case C128: {
        TF_ASSIGN_OR_RETURN(parent_->evaluated_[map], MapImpl<complex128>(map));
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

  Status HandleSort(HloInstruction* sort) override {
    return UnsupportedTypeError(sort);
  }

  Status HandleSelectAndScatter(HloInstruction* select_and_scatter) override {
    auto operand = select_and_scatter->operand(0);
    auto source = select_and_scatter->operand(1);
    const Window& window = select_and_scatter->window();

    const Literal& init_literal =
        parent_->GetEvaluatedLiteralFor(select_and_scatter->operand(2));
    TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
    auto init_scalar = init_literal.Get<ReturnT>({});

    Literal result(select_and_scatter->shape());

    // Initialize result array with the init value.
    TF_RETURN_IF_ERROR(result.Populate<ReturnT>(
        [&](absl::Span<const int64_t> output_index) { return init_scalar; }));

    std::vector<int64_t> window_dimension_sizes;
    for (const auto& window_dimension : window.dimensions()) {
      window_dimension_sizes.push_back(window_dimension.size());
    }
    const Shape window_shape = ShapeUtil::MakeShape(
        operand->shape().element_type(), window_dimension_sizes);

    HloComputation* select = select_and_scatter->select();
    HloComputation* scatter = select_and_scatter->scatter();

    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    const Literal& source_literal = parent_->GetEvaluatedLiteralFor(source);

    int64_t rank = operand_literal.shape().rank();

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
      absl::optional<ReturnT> selected_val;
      absl::optional<std::vector<int64_t>> selected_index;

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](const std::vector<int64_t>& operand_index) {
            auto curr_val = operand_literal.Get<ReturnT>(operand_index);
            if (!selected_val) {
              selected_val = curr_val;
              selected_index = operand_index;
            }
            curr_val_literal.Set({}, curr_val);
            selected_val_literal.Set({}, *selected_val);
            Literal computed_result =
                embedded_evaluator
                    .Evaluate(*select,
                              {&selected_val_literal, &curr_val_literal})
                    .ConsumeValueOrDie();
            bool selected = !computed_result.Get<bool>({});
            if (selected) {
              selected_val = curr_val;
              selected_index = operand_index;
            }
            embedded_evaluator.ResetVisitStates();
          });

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](const std::vector<int64_t>& operand_index) {
            if (std::equal(operand_index.begin(), operand_index.end(),
                           selected_index->begin())) {
              auto source = source_literal.Get<ReturnT>(source_index);
              auto scattered = result.Get<ReturnT>(operand_index);
              source_literal_scatter.Set({}, source);
              scattered_literal.Set({}, scattered);
              Literal computed_result =
                  embedded_evaluator
                      .Evaluate(*scatter,
                                {&source_literal_scatter, &scattered_literal})
                      .ConsumeValueOrDie();
              result.Set(operand_index, computed_result.Get<ReturnT>({}));
              // Clear visit states so that the we can use the evaluator again
              // on the same computation.
              embedded_evaluator.ResetVisitStates();
            }
          });
    } while (
        IndexUtil::BumpIndices(source->shape(), absl::MakeSpan(source_index)));

    parent_->evaluated_[select_and_scatter] = std::move(result);
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* reduce_window) override {
    auto* reduce_window_instr = Cast<HloReduceWindowInstruction>(reduce_window);
    const Window& window = reduce_window->window();
    HloComputation* function = reduce_window->to_apply();
    TF_ASSIGN_OR_RETURN(
        auto inferred_return_shape,
        ShapeInference::InferReduceWindowShape(
            reduce_window_instr->input_shapes(),
            reduce_window_instr->init_value_shapes(), window,
            /*to_apply_shape=*/function->ComputeProgramShape()));
    TF_RET_CHECK(
        ShapeUtil::Compatible(reduce_window->shape(), inferred_return_shape))
        << "return shape is set to: "
        << ShapeUtil::HumanStringWithLayout(reduce_window->shape())
        << " but is inferred to be: "
        << ShapeUtil::HumanStringWithLayout(inferred_return_shape);

    absl::InlinedVector<const Literal*, 2> input_literal_vec, init_literal_vec;
    auto input_arrays = reduce_window_instr->inputs();
    auto init_values = reduce_window_instr->init_values();
    int64_t num_args = input_arrays.size();
    for (int i = 0; i < num_args; ++i) {
      const Literal& input_literal =
          parent_->GetEvaluatedLiteralFor(input_arrays[i]);
      VLOG(3) << "HandleReduceWindow arg_literal: " << input_literal.ToString();
      input_literal_vec.push_back(&input_literal);
      const Literal& init_literal =
          parent_->GetEvaluatedLiteralFor(init_values[i]);
      VLOG(3) << "HandleReduceWindow init_literal: " << init_literal.ToString();
      TF_RET_CHECK(ShapeUtil::IsScalar(init_literal.shape()));
      init_literal_vec.push_back(&init_literal);
    }
    // Creates a Shape object from window, for iteration below.
    absl::InlinedVector<int64_t, 2> window_dimension_sizes;
    for (const auto& window_dimension : window.dimensions()) {
      window_dimension_sizes.push_back(window_dimension.size());
    }
    const Shape window_shape = ShapeUtil::MakeShape(
        input_arrays[0]->shape().element_type(), window_dimension_sizes);

    HloEvaluator embedded_evaluator(parent_->max_loop_iterations_);
    // For each resulting dimension, calculate and assign computed value.
    auto evaluate_impl =
        [&](absl::Span<const int64_t> output_index) -> std::vector<Literal> {
      std::vector<Literal> computed_result;
      computed_result.reserve(init_literal_vec.size());
      for (const auto* init : init_literal_vec) {
        computed_result.push_back(init->Clone());
      }
      IterateThroughWindow(
          window_shape, window, input_literal_vec[0]->shape(), output_index,
          [&](absl::Span<const int64_t> operand_index) -> void {
            absl::InlinedVector<const Literal*, 2> args;
            for (auto& curr_result_val : computed_result) {
              VLOG(2) << "Pushing:" << curr_result_val.ToString() << "\n";
              args.push_back(&curr_result_val);
            }
            absl::InlinedVector<Literal, 2> curr_val_literal_vec(
                input_literal_vec.size());
            for (const auto* input_literal : input_literal_vec) {
              // Evaluate computation with specified literal operands.
              curr_val_literal_vec.push_back(Literal(ShapeUtil::MakeShape(
                  input_literal->shape().element_type(), {})));
              TF_CHECK_OK(curr_val_literal_vec.back().CopyElementFrom(
                  *input_literal, operand_index, {}));
              VLOG(2) << "Pushing:" << curr_val_literal_vec.back().ToString()
                      << "\n";
              args.push_back(&curr_val_literal_vec.back());
            }
            computed_result[0] = embedded_evaluator.Evaluate(*function, args)
                                     .ConsumeValueOrDie();
            VLOG(2) << "Computed result:" << computed_result[0].ToString()
                    << "\n";
            // Clear visit states so that the we can use the evaluate again
            // on the same computation.
            embedded_evaluator.ResetVisitStates();
            if (inferred_return_shape.IsTuple()) {
              computed_result = computed_result[0].DecomposeTuple();
            }
          });
      VLOG(2) << "Final result size:" << computed_result.size() << "\n";
      for (const auto& res : computed_result) {
        VLOG(2) << res.ToString() << "\n";
      }
      return computed_result;
    };
    Literal result(inferred_return_shape);
    if (inferred_return_shape.IsTuple()) {
      absl::InlinedVector<Literal, 1> results(num_args);
      for (int64_t i = 0; i < num_args; ++i) {
        results[i] = Literal(inferred_return_shape.tuple_shapes(i));
      }
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
          inferred_return_shape.tuple_shapes(0),
          [&](absl::Span<const int64_t> output_index) -> StatusOr<bool> {
            std::vector<Literal> computed_result_vec =
                evaluate_impl(output_index);
            for (int i = 0; i < computed_result_vec.size(); ++i) {
              TF_RETURN_IF_ERROR(results[i].CopyElementFrom(
                  computed_result_vec[i], {}, output_index));
            }
            return true;
          }));
      result = Literal::MoveIntoTuple(absl::MakeSpan(results));
      VLOG(2) << "Final result is:" << result.ToString() << "\n";
    } else {
      TF_RETURN_IF_ERROR(
          result.Populate<ReturnT>([&](absl::Span<const int64_t> output_index) {
            return evaluate_impl(output_index)[0].template Get<ReturnT>({});
          }));
    }
    VLOG(2) << "Final result is:" << result.ToString() << "\n";
    parent_->evaluated_[reduce_window] = std::move(result);
    return Status::OK();
  }

  // Reshapes the scatter indices input to have a trailing degenerate `1`
  // dimension if necessary.  Hands over the ownership of the newly created
  // literal (if there is one) to `reshaped_indices`.
  StatusOr<std::reference_wrapper<const Literal>> ReshapedScatterIndices(
      int64_t index_vector_dim, const Literal& indices,
      Literal* reshaped_indices) {
    if (indices.shape().dimensions_size() != index_vector_dim) {
      return std::cref(indices);
    }

    std::vector<int64_t> new_shape(indices.shape().dimensions().begin(),
                                   indices.shape().dimensions().end());
    new_shape.push_back(1);
    TF_ASSIGN_OR_RETURN(*reshaped_indices, indices.Reshape(new_shape));
    return std::cref(*reshaped_indices);
  }

  // Returns an ShapeUtil::IndexIterationSpace that iterates over the update
  // scatter dimensions while keeping the rest of the update dimensions clamped
  // to 0.
  ShapeUtil::IndexIterationSpace IterationSpaceForUpdateScatterIndices(
      const Shape& updates_shape, const ScatterDimensionNumbers& dim_numbers) {
    int64_t updates_rank = updates_shape.dimensions_size();
    std::vector<int64_t> index_base(updates_rank, 0);
    std::vector<int64_t> index_count(updates_rank, 1);
    for (int64_t i = 0; i < updates_rank; i++) {
      bool is_update_scatter_dim =
          !absl::c_binary_search(dim_numbers.update_window_dims(), i);
      if (is_update_scatter_dim) {
        index_count[i] = updates_shape.dimensions(i);
      }
    }
    return {std::move(index_base), std::move(index_count),
            std::vector<int64_t>(updates_rank, 1)};
  }

  // Return an ShapeUtil::IndexIterationSpace that iterates over the update
  // window dimensions while keeping the rest of the update dimensions clamped
  // to 0.
  ShapeUtil::IndexIterationSpace IterationSpaceForUpdateWindowIndices(
      const Shape& updates_shape, const ScatterDimensionNumbers& dim_numbers) {
    int64_t updates_rank = updates_shape.dimensions_size();
    std::vector<int64_t> index_base(updates_rank, 0);
    std::vector<int64_t> index_count(updates_rank, 1);
    for (int64_t i = 0; i < updates_rank; i++) {
      bool is_update_window_dim =
          absl::c_binary_search(dim_numbers.update_window_dims(), i);
      if (is_update_window_dim) {
        index_count[i] = updates_shape.dimensions(i);
      }
    }
    return {std::move(index_base), std::move(index_count),
            std::vector<int64_t>(updates_rank, 1)};
  }

  // This functor computes the contribution of scatter_indices to an input index
  // corresponding to an update index.  That is, given an update index I, it
  // picks out the scatter indices in I and uses them to look up a scatter
  // index, S, from the scatter indices tensor, and expands S into the input
  // space according to scatter_dims_to_operand_dims.
  //
  // This is similar to the class HloEvaluator::OutputGatherIndexToInputIndex
  // that does the corresponding function for Gather.
  class UpdateScatterIndexToInputIndex {
   public:
    // The constructor does some setup work that is amortized across all
    // iterations.
    explicit UpdateScatterIndexToInputIndex(
        const ScatterDimensionNumbers* dim_numbers, const Shape& input_shape,
        const Shape& updates_shape, const Literal* scatter_indices)
        : dim_numbers_(*dim_numbers), scatter_indices_(*scatter_indices) {
      for (int64_t i = 0; i < updates_shape.dimensions_size(); i++) {
        update_dim_is_scatter_dims_.push_back(
            !absl::c_binary_search(dim_numbers_.update_window_dims(), i));
      }

      for (int64_t i = 0; i < input_shape.dimensions_size(); i++) {
        int64_t index_of_input_dim_in_index_vector =
            FindIndex(dim_numbers_.scatter_dims_to_operand_dims(), i);
        if (index_of_input_dim_in_index_vector ==
            dim_numbers_.scatter_dims_to_operand_dims_size()) {
          input_dim_value_to_index_vector_.push_back(-1);
        } else {
          input_dim_value_to_index_vector_.push_back(
              index_of_input_dim_in_index_vector);
        }
      }

      index_vector_index_.resize(scatter_indices_.shape().dimensions_size());
      input_index_.resize(input_shape.dimensions_size());
      int64_t index_vector_size =
          scatter_indices_.shape().dimensions(dim_numbers_.index_vector_dim());
      index_vector_.resize(index_vector_size);
    }

    // Returns the contribution of scatter_indices to the input index
    // corresponding to update_index.  See scatter_inner_loop_body.
    //
    // This is conceptually  a stateless transformation from update_index to the
    // scatter input index, but:
    //
    //  - Instead of allocating memory to represent the scatter input index on
    //    every invocation we reuse the same storage for the result
    //    (input_index_), mutating it in place.
    //  - Instead of allocating buffers for temporary values like
    //    index_vector_index_ and index_vector on every invocation, we reuse the
    //    same storage for all invocations.
    //
    // This returns a Span into memory owned by the class.
    StatusOr<absl::Span<const int64_t>> operator()(
        absl::Span<const int64_t> update_index) {
      PropagateUpdateIndexScatterDimsToIndexVectorIndex(update_index);
      TF_RETURN_IF_ERROR(FetchIndexVector());
      PropagateIndexVectorToInputIndex();
      return absl::Span<const int64_t>(input_index_);
    }

   private:
    // Propagates the scatter index dimensions from the update index into
    // index_vector_index_ by mutating index_vector_index_ in place.  Does not
    // update the dim_numbers.index_vector_dim() dimension -- that's the
    // dimension we iterate over in FetchIndexVector.
    void PropagateUpdateIndexScatterDimsToIndexVectorIndex(
        absl::Span<const int64_t> update_index) {
      int64_t index_vector_index_i = 0;
      for (int64_t i = 0, e = update_index.size(); i < e; i++) {
        if (!update_dim_is_scatter_dims_[i]) {
          continue;
        }

        if (index_vector_index_i == dim_numbers_.index_vector_dim()) {
          index_vector_index_i++;
        }

        index_vector_index_[index_vector_index_i++] = update_index[i];
      }
    }

    // Populates index_vector_ by iterating over scatter_indices_ according to
    // index_vector_index_.
    Status FetchIndexVector() {
      int64_t index_vector_dim = dim_numbers_.index_vector_dim();
      for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
        index_vector_index_[index_vector_dim] = i;
        index_vector_[i] =
            *scatter_indices_.GetIntegralAsS64(index_vector_index_);
      }
      return Status::OK();
    }

    // Populates input_index_.
    void PropagateIndexVectorToInputIndex() {
      for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
        if (input_dim_value_to_index_vector_[i] != -1) {
          input_index_[i] = index_vector_[input_dim_value_to_index_vector_[i]];
        }

        // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
        // remains 0, as set by the constructor.
      }
    }

    // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
    // of the input index from the index vector.  See
    // PropagateIndexVectorToInputIndex.
    std::vector<int64_t> input_dim_value_to_index_vector_;

    // update_dim_is_scatter_dims_[i] is true iff the update index i is a
    // scatter dimension.
    std::vector<bool> update_dim_is_scatter_dims_;

    // The buffer into which we construct an index into scatter_indices_ to
    // fetch the index vector.
    std::vector<int64_t> index_vector_index_;

    // The index vector fetched from scatter_indices_.
    std::vector<int64_t> index_vector_;

    // The result computed by this functor.  operator() returns a Span
    // into this vector.
    std::vector<int64_t> input_index_;

    const ScatterDimensionNumbers& dim_numbers_;
    const Literal& scatter_indices_;
  };

  // This functor computes the contribution of the window indices in an update
  // index to an input index.  That is, given an update index I it picks out the
  // update window indices in I and expands it into a window index into the
  // input shape.
  //
  // This is similar to the class HloEvaluator::OutputWindowIndexToInputIndex
  // that does the corresponding function for Gather.
  class UpdateWindowIndexToInputIndex {
   public:
    // The constructor does some setup work that is amortized across all
    // iterations.
    explicit UpdateWindowIndexToInputIndex(
        const ScatterDimensionNumbers& dim_numbers, const Shape& input_shape,
        const Shape& updates_shape) {
      std::vector<int64_t> window_index_to_update_index;
      int64_t update_index_count = 0;
      for (int64_t i = 0; i < updates_shape.dimensions_size(); i++) {
        if (absl::c_binary_search(dim_numbers.update_window_dims(), i)) {
          window_index_to_update_index.push_back(update_index_count++);
        } else {
          update_index_count++;
        }
      }

      int64_t window_dim_count = 0;
      for (int64_t i = 0; i < input_shape.dimensions_size(); i++) {
        if (absl::c_binary_search(dim_numbers.inserted_window_dims(), i)) {
          input_dim_value_to_update_index_.push_back(-1);
        } else {
          input_dim_value_to_update_index_.push_back(
              window_index_to_update_index[window_dim_count++]);
        }
      }

      input_index_.resize(input_shape.dimensions_size());
    }

    // Returns the contribution of the window indices to the input index
    // corresponding to update_index.  See scatter_inner_loop_body.
    //
    // This is conceptually a stateless transformation from update_index to the
    // window input index, but instead of allocating memory to represent the
    // scatter input index on every invocation we reuse the same storage for the
    // result (input_index_), mutating it in place.
    //
    // This returns a Span into memory owned by the class.
    StatusOr<absl::Span<const int64_t>> operator()(
        absl::Span<const int64_t> update_index) {
      PropagateUpdateIndexWindowDimsToInputIndex(update_index);
      return absl::Span<const int64_t>(input_index_);
    }

    // Returns for a given 'input_dim' the corresponding update dimension index,
    // or -1 if 'input_dim' is an elided window dimension.
    int64_t input_dim_value_to_update_index(int64_t input_dim) {
      return input_dim_value_to_update_index_[input_dim];
    }

   private:
    // Propagates window dimensions from the update index to input_index_ by
    // mutating input_index_ in place.
    void PropagateUpdateIndexWindowDimsToInputIndex(
        absl::Span<const int64_t> update_index) {
      for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
        if (input_dim_value_to_update_index_[i] != -1) {
          input_index_[i] = update_index[input_dim_value_to_update_index_[i]];
        }

        // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
        // remains 0, as set by the constructor.
      }
    }

    // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
    // of the input index from the update index. See
    // PropagateUpdateIndexWindowDimsToInputIndex.
    std::vector<int64_t> input_dim_value_to_update_index_;

    // The result computed by this functor.  operator() returns a Span
    // into this vector.
    std::vector<int64_t> input_index_;
  };

  Status HandleScatter(HloInstruction* scatter) override {
    const ScatterDimensionNumbers& dim_numbers =
        scatter->scatter_dimension_numbers();
    const Literal& operand =
        parent_->GetEvaluatedLiteralFor(scatter->operand(0));
    Literal reshaped_scatter_indices;
    TF_ASSIGN_OR_RETURN(const Literal& scatter_indices,
                        ReshapedScatterIndices(dim_numbers.index_vector_dim(),
                                               parent_->GetEvaluatedLiteralFor(
                                                   scatter->operand(1)),
                                               &reshaped_scatter_indices));
    const Literal& updates =
        parent_->GetEvaluatedLiteralFor(scatter->operand(2));
    const Shape& updates_shape = updates.shape();
    const Shape& operand_shape = operand.shape();

    ShapeUtil::IndexIterationSpace scatter_indices_iteration_space =
        IterationSpaceForUpdateScatterIndices(updates_shape, dim_numbers);
    ShapeUtil::IndexIterationSpace window_indices_iteration_space =
        IterationSpaceForUpdateWindowIndices(updates_shape, dim_numbers);

    std::vector<int64_t> input_index(operand_shape.dimensions_size());
    std::vector<int64_t> update_index(updates_shape.dimensions_size());

    UpdateScatterIndexToInputIndex update_scatter_index_to_input_index(
        &scatter->scatter_dimension_numbers(), /*input_shape=*/operand_shape,
        updates_shape, &scatter_indices);
    UpdateWindowIndexToInputIndex update_window_index_to_input_index(
        scatter->scatter_dimension_numbers(), /*input_shape=*/operand_shape,
        updates_shape);

    // Initialize the result with the operand. This makes it easier to handle
    // the updates even when the indices are repeated.
    Literal result = operand.Clone();
    HloEvaluator embedded_evaluator;
    auto scatter_inner_loop_body =
        [&](absl::Span<const int64_t> update_window_index,
            absl::Span<const int64_t> input_scatter_index,
            absl::Span<const int64_t> update_scatter_index) -> StatusOr<bool> {
      TF_ASSIGN_OR_RETURN(
          absl::Span<const int64_t> input_window_index,
          update_window_index_to_input_index(update_window_index));
      for (int i = 0, e = update_index.size(); i < e; i++) {
        update_index[i] = update_scatter_index[i] + update_window_index[i];
        DCHECK_LT(update_index[i], updates_shape.dimensions(i));
      }
      for (int i = 0, e = input_scatter_index.size(); i < e; i++) {
        int64_t update_dim =
            update_window_index_to_input_index.input_dim_value_to_update_index(
                i);
        // If 'update_dim' is -1, it means 'i' is an elided window dim. This
        // means we set the iteration index to 0, so for the purpose of the
        // following calculations we can consider the update dimension size to
        // be 1.
        int64_t update_dim_size =
            update_dim == -1 ? 1 : updates_shape.dimensions(update_dim);
        // If any part of the update region is out-of-bounds, then do not
        // perform any update on the input.
        if ((input_scatter_index[i] < 0) ||
            (input_scatter_index[i] >
             operand_shape.dimensions(i) - update_dim_size)) {
          return true;
        }
      }
      for (int i = 0, e = input_index.size(); i < e; i++) {
        input_index[i] = input_scatter_index[i] + input_window_index[i];
      }

      auto result_value_literal =
          LiteralUtil::CreateR0<ReturnT>(result.Get<ReturnT>(input_index));
      auto update_value_literal =
          LiteralUtil::CreateR0<ReturnT>(updates.Get<ReturnT>(update_index));
      Literal updated_result =
          embedded_evaluator
              .Evaluate(*scatter->to_apply(),
                        {&result_value_literal, &update_value_literal})
              .ConsumeValueOrDie();
      // Clear visit states so that the we can use the evaluate again on the
      // same computation.
      embedded_evaluator.ResetVisitStates();
      result.Set<ReturnT>(input_index, updated_result.Get<ReturnT>({}));
      return true;
    };

    auto scatter_outer_loop_body =
        [&](absl::Span<const int64_t> update_scatter_index) -> StatusOr<bool> {
      TF_ASSIGN_OR_RETURN(
          absl::Span<const int64_t> input_scatter_index,
          update_scatter_index_to_input_index(update_scatter_index));
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
          updates_shape, window_indices_iteration_space,
          [&](absl::Span<const int64_t> update_window_index) {
            return scatter_inner_loop_body(
                update_window_index, input_scatter_index, update_scatter_index);
          }));
      return true;
    };

    TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
        updates_shape, scatter_indices_iteration_space,
        scatter_outer_loop_body));
    parent_->evaluated_[scatter] = std::move(result);
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

    const int64_t rank = operand->shape().rank();
    const Literal& operand_literal = parent_->GetEvaluatedLiteralFor(operand);
    auto func = [&](absl::Span<const int64_t> out_index) {
      DimensionVector operand_index(rank);
      for (int64_t i = 0; i < rank; ++i) {
        operand_index[i] =
            slice->slice_starts(i) + out_index[i] * slice->slice_strides(i);
      }
      return operand_literal.Get<ReturnT>(operand_index);
    };

    Literal result(shape);
    TF_RETURN_IF_ERROR(result.Populate<ReturnT>(func));
    parent_->evaluated_[slice] = std::move(result);
    return Status::OK();
  }

  // Enable CLZ only for int32_t, uint32_t, int64_t and uint64_t.
  template <typename NativeT,
            typename std::enable_if<
                (!std::is_integral<NativeT>::value ||
                 std::is_same<NativeT, bool>::value)>::type* = nullptr>
  Status HandleClz(HloInstruction* clz) {
    return UnsupportedTypeError(clz);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandleClz(HloInstruction* clz) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[clz],
        ElementWiseUnaryOp(clz, [](ElementwiseT elem_operand) {
          using UnsignedElementwiseT = std::make_unsigned_t<ElementwiseT>;
          return (std::numeric_limits<UnsignedElementwiseT>::digits - 1) -
                 Log2Floor<UnsignedElementwiseT>(elem_operand);
        }));
    return Status::OK();
  }

  Status HandleClz(HloInstruction* clz) override {
    return HandleClz<ElementwiseT>(clz);
  }

  template <typename NativeT,
            typename std::enable_if<
                (!std::is_integral<NativeT>::value ||
                 std::is_same<NativeT, bool>::value)>::type* = nullptr>
  Status HandlePopulationCount(HloInstruction* popcnt) {
    return UnsupportedTypeError(popcnt);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_integral<NativeT>::value &&
                !std::is_same<NativeT, bool>::value>::type* = nullptr>
  Status HandlePopulationCount(HloInstruction* popcnt) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[popcnt],
        ElementWiseUnaryOp(popcnt, [](ElementwiseT elem_operand) {
          return std::bitset<CHAR_BIT * sizeof elem_operand>(elem_operand)
              .count();
        }));
    return Status::OK();
  }

  Status HandlePopulationCount(HloInstruction* popcnt) override {
    return HandlePopulationCount<ElementwiseT>(popcnt);
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
    return UnsupportedTypeError(sin);
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
    return UnsupportedTypeError(cos);
  }

  Status HandleCos(HloInstruction* cos) override {
    return HandleCos<ElementwiseT>(cos);
  }

  template <typename NativeT,
            typename std::enable_if<
                std::is_same<NativeT, float>::value ||
                std::is_same<NativeT, double>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[reduce_precision],
        ElementWiseUnaryOp(reduce_precision, [&](ElementwiseT elem) {
          const uint32_t src_mantissa_bits =
              std::numeric_limits<NativeT>::digits - 1;
          const uint32_t src_exponent_bits =
              8 * sizeof(NativeT) - src_mantissa_bits - 1;
          const uint32_t dest_mantissa_bits = reduce_precision->mantissa_bits();
          const uint32_t dest_exponent_bits = reduce_precision->exponent_bits();

          using Uint = typename UintWithSize<sizeof(NativeT)>::type;
          Uint value_as_int = absl::bit_cast<Uint>(elem);

          // Code is based on the CPU/GPU implementation in LLVM-emitting code.
          //
          // Bits in float32 type:
          //   mantissa : bits [0:22]
          //   exponent : bits [23:30]
          //   sign     : bits [31]
          if (dest_mantissa_bits < src_mantissa_bits) {
            const Uint last_mantissa_bit_mask =
                Uint{1} << (src_mantissa_bits - dest_mantissa_bits);

            // Compute rounding bias for round-to-nearest with ties to even.
            // This is equal to a base value of 0111... plus one bit if the last
            // remaining mantissa bit is 1.
            const Uint base_rounding_bias = (last_mantissa_bit_mask >> 1) - 1;
            const Uint x_last_mantissa_bit =
                (value_as_int & last_mantissa_bit_mask) >>
                (src_mantissa_bits - dest_mantissa_bits);
            const Uint x_rounding_bias =
                x_last_mantissa_bit + base_rounding_bias;

            // Add rounding bias, and mask out truncated bits.  Note that the
            // case where adding the rounding bias overflows into the exponent
            // bits is correct; the non-masked mantissa bits will all be zero,
            // and the exponent will be incremented by one.
            const Uint truncation_mask = ~(last_mantissa_bit_mask - 1);
            value_as_int = value_as_int + x_rounding_bias;
            value_as_int = value_as_int & truncation_mask;
          }
          if (dest_exponent_bits < src_exponent_bits) {
            // Masks for f32 values.
            const Uint sign_bit_mask = Uint{1} << 8 * sizeof(NativeT) - 1;
            const Uint exp_bits_mask = (Uint{1 << src_exponent_bits} - 1)
                                       << src_mantissa_bits;

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
            const Uint exponent_bias = (Uint{1} << (src_exponent_bits - 1)) - 1;
            const Uint reduced_exponent_bias =
                (1 << (dest_exponent_bits - 1)) - 1;
            const Uint reduced_max_exponent =
                exponent_bias + reduced_exponent_bias;
            const Uint reduced_min_exponent =
                exponent_bias - reduced_exponent_bias;

            // Do we overflow or underflow?
            const Uint x_exponent = value_as_int & exp_bits_mask;
            const bool x_overflows =
                x_exponent > (reduced_max_exponent << src_mantissa_bits);
            const bool x_underflows =
                x_exponent <= (reduced_min_exponent << src_mantissa_bits);

            // Compute appropriately-signed values of zero and infinity.
            const Uint x_signed_zero = value_as_int & sign_bit_mask;
            const Uint x_signed_inf = x_signed_zero | exp_bits_mask;

            // Force to zero or infinity if overflow or underflow.  (Note that
            // this truncates all denormal values to zero, rather than rounding
            // them.)
            value_as_int = x_overflows ? x_signed_inf : value_as_int;
            value_as_int = x_underflows ? x_signed_zero : value_as_int;
          }

          NativeT reduced_result = absl::bit_cast<NativeT>(value_as_int);
          if (std::isnan(elem)) {
            reduced_result = dest_mantissa_bits > 0
                                 ? elem
                                 : std::numeric_limits<NativeT>::infinity();
          }
          return reduced_result;
        }));
    return Status::OK();
  }

  template <
      typename NativeT,
      typename std::enable_if<std::is_integral<NativeT>::value ||
                              is_complex_t<NativeT>::value>::type* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return UnsupportedTypeError(reduce_precision);
  }

  Status HandleReducePrecision(HloInstruction* reduce_precision) override {
    return HandleReducePrecision<ElementwiseT>(reduce_precision);
  }

  template <
      typename NativeT,
      typename std::enable_if<
          std::is_same<NativeT, bfloat16>::value ||
          std::is_same<NativeT, Eigen::half>::value ||
          std::is_integral<NativeT>::value || is_complex_t<NativeT>::value ||
          std::is_floating_point<NativeT>::value>::type* = nullptr>
  Status HandleIota(HloInstruction* instruction) {
    auto* iota = Cast<HloIotaInstruction>(instruction);

    Literal result(iota->shape());
    ShapeUtil::ForEachIndex(iota->shape(), [&](absl::Span<const int64_t> idx) {
      result.Set(idx, static_cast<NativeT>(idx[iota->iota_dimension()]));
      return true;
    });
    parent_->evaluated_[iota] = std::move(result);
    return Status::OK();
  }
  template <
      typename NativeT,
      typename std::enable_if<
          !(std::is_same<NativeT, bfloat16>::value ||
            std::is_same<NativeT, Eigen::half>::value ||
            std::is_integral<NativeT>::value || is_complex_t<NativeT>::value ||
            std::is_floating_point<NativeT>::value)>::type* = nullptr>
  Status HandleIota(HloInstruction* iota) {
    return UnsupportedTypeError(iota);
  }
  Status HandleIota(HloInstruction* iota) override {
    return HandleIota<ReturnT>(iota);
  }

  template <typename NativeT,
            typename std::enable_if<
                !(std::is_integral<NativeT>::value ||
                  std::is_floating_point<NativeT>::value)>::type* = nullptr>
  Status HandleRng(HloInstruction* random) {
    return UnsupportedTypeError(random);
  }
  template <typename NativeT,
            typename std::enable_if<
                (std::is_floating_point<NativeT>::value)>::type* = nullptr>
  Status HandleRng(HloInstruction* random) {
    RandomDistribution distribution = random->random_distribution();
    const auto result_shape = random->shape();
    Literal result(result_shape);

    switch (distribution) {
      case RNG_UNIFORM: {
        const Literal& low =
            parent_->GetEvaluatedLiteralFor(random->operand(0));
        const Literal& high =
            parent_->GetEvaluatedLiteralFor(random->operand(1));

        // std::uniform_real_distribution(a, b) can sometimes return a value
        // equal to b.  Unclear if this is a spec bug or an implementation bug
        // or WAI [0] [1] [2].  Anyway for our purposes we want a half-open
        // interval, so we have to re-sample if we get `b` out.
        //
        // [0] https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63176
        // [1] https://bugs.llvm.org/show_bug.cgi?id=18767
        // [2] http://open-std.org/JTC1/SC22/WG21/docs/lwg-active.html#2524
        auto low_val = low.Get<NativeT>({});
        auto high_val = high.Get<NativeT>({});
        std::uniform_real_distribution<NativeT> generator(low_val, high_val);
        TF_RETURN_IF_ERROR(result.Populate<NativeT>(
            [&](absl::Span<const int64_t> /*indexes*/) {
              while (true) {
                NativeT v = generator(parent_->engine_);
                if (v != high_val) {
                  return v;
                }
              }
            }));
        break;
      }
      case RNG_NORMAL: {
        const Literal& mean =
            parent_->GetEvaluatedLiteralFor(random->operand(0));
        const Literal& stddev =
            parent_->GetEvaluatedLiteralFor(random->operand(1));

        std::normal_distribution<NativeT> generator(mean.Get<NativeT>({}),
                                                    stddev.Get<NativeT>({}));

        TF_RETURN_IF_ERROR(result.Populate<NativeT>(
            [&](absl::Span<const int64_t> /*indexes*/) {
              return generator(parent_->engine_);
            }));
        break;
      }
      default:
        return UnimplementedStrCat("The distribution ",
                                   RandomDistribution_Name(distribution),
                                   " is not implemented.");
    }
    parent_->evaluated_[random] = std::move(result);
    return Status::OK();
  }
  template <typename NativeT,
            typename std::enable_if<(std::is_integral<NativeT>::value)>::type* =
                nullptr>
  Status HandleRng(HloInstruction* random) {
    RandomDistribution distribution = random->random_distribution();
    const auto result_shape = random->shape();
    Literal result(result_shape);

    switch (distribution) {
      case RNG_UNIFORM: {
        const Literal& low =
            parent_->GetEvaluatedLiteralFor(random->operand(0));
        const Literal& high =
            parent_->GetEvaluatedLiteralFor(random->operand(1));

        // Note std::uniform_int_distribution assumes interval is closed, i.e.,
        // [low, high], but we want [low, high) instead. Hence high-1 is used as
        // the upper range.
        std::uniform_int_distribution<int64_t> generator(
            low.Get<NativeT>({}), high.Get<NativeT>({}) - 1);

        TF_RETURN_IF_ERROR(result.Populate<NativeT>(
            [&](absl::Span<const int64_t> /*indexes*/) {
              return static_cast<NativeT>(generator(parent_->engine_));
            }));
        break;
      }
      case RNG_NORMAL: {
        return Unimplemented(
            "Normal distribution is not supported for integral types.");
      }
      default:
        return UnimplementedStrCat("The distribution ",
                                   RandomDistribution_Name(distribution),
                                   " is not implemented.");
    }
    parent_->evaluated_[random] = std::move(result);
    return Status::OK();
  }
  Status HandleRng(HloInstruction* random) override {
    return HandleRng<ReturnT>(random);
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
    DimensionVector v(shape.rank());
    int64_t scale = 1;
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
      const absl::Span<const int64_t> window_count_index,
      const std::function<void(const std::vector<int64_t>&)>& f) {
    const int64_t rank = base_shape.rank();
    DimensionVector window_index(rank);
    std::fill(window_index.begin(), window_index.end(), 0);
    do {
      std::vector<int64_t> base_index(rank);
      bool out_of_bound = false;
      for (int64_t i = 0; i < rank; ++i) {
        // Padding is applied to the dilated base. Say that padding is 3 and
        // dilation is 2 for some dimension. After applying base dilation and
        // padding, the dimension looks like:
        // P P P E D D E D D ... E D D E P P P
        // where E are the elements and D are the holes. So, the elements are
        // located in indices: padding + k*base_dilation for k = {0, 1, 2, ...}.
        // We are accessing elements in the transformed base at indices:
        // window_count_index * stride + window_index * window_dilation.
        // Solving for k gives us
        // (win_count_i * stride + win_i * win_dilation - pad) / base_dilation
        // When this is a natural number, we index an original element.
        // Otherwise, we index a 0 (pad or hole), and we don't need to apply
        // the callback f.
        base_index[i] =
            window_count_index[i] * window.dimensions(i).stride() +
            window_index[i] * window.dimensions(i).window_dilation() -
            window.dimensions(i).padding_low();
        if (base_index[i] % window.dimensions(i).base_dilation() != 0) {
          out_of_bound = true;
          break;
        }
        base_index[i] /= window.dimensions(i).base_dilation();
        if (base_index[i] < 0 || base_index[i] >= base_shape.dimensions(i)) {
          out_of_bound = true;
          break;
        }
      }
      if (!out_of_bound) {
        f(base_index);
      }
    } while (
        IndexUtil::BumpIndices(window_shape, absl::MakeSpan(window_index)));
  }

  template <typename IndexT>
  StatusOr<Literal> DynamicSlice(
      const Literal& operand_literal,
      absl::Span<HloInstruction* const> start_indices,
      const Shape& result_shape) {
    std::vector<int64_t> start;

    for (HloInstruction* index : start_indices) {
      start.push_back(
          parent_->GetEvaluatedLiteralFor(index).GetFirstElement<IndexT>());
    }

    // Clamp the start indices so the slice is in-bounds w.r.t the operand.
    for (int64_t i = 0; i < start.size(); ++i) {
      start[i] = std::min<int64_t>(
          std::max(int64_t{0}, start[i]),
          operand_literal.shape().dimensions(i) - result_shape.dimensions(i));
    }

    std::vector<int64_t> operand_indices(start.size());
    Literal result(result_shape);
    TF_RETURN_IF_ERROR(
        result.Populate<ReturnT>([&](absl::Span<const int64_t> multi_index) {
          for (int64_t i = 0; i < operand_indices.size(); ++i) {
            CHECK_GE(multi_index[i] + start[i], 0);
            operand_indices[i] = multi_index[i] + start[i];
          }

          auto result = operand_literal.Get<ReturnT>(operand_indices);
          return result;
        }));

    return std::move(result);
  }

  template <typename IndexT>
  StatusOr<Literal> DynamicUpdateSlice(
      const Literal& operand_literal, const Literal& update_literal,
      absl::Span<HloInstruction* const> start_indices) {
    auto result = operand_literal.Clone();
    const auto rank = result.shape().rank();
    std::vector<int64_t> start;
    for (HloInstruction* index : start_indices) {
      start.push_back(
          parent_->GetEvaluatedLiteralFor(index).GetFirstElement<IndexT>());
    }

    // Clamp the update start indices so the slice is in-bounds w.r.t the
    // operand.
    for (int64_t i = 0; i < rank; ++i) {
      start[i] = std::min<int64_t>(
          std::max<int64_t>(0, start[i]),
          result.shape().dimensions(i) - update_literal.shape().dimensions(i));
    }
    std::vector<int64_t> result_index(rank, 0);

    auto func = [&](absl::Span<const int64_t> update_index) {
      std::transform(update_index.begin(), update_index.end(), start.begin(),
                     result_index.begin(), std::plus<int64_t>());
      result.Set<ReturnT>(result_index,
                          update_literal.Get<ReturnT>(update_index));
      return true;
    };

    std::vector<int64_t> base(update_literal.shape().dimensions_size(), 0);
    std::vector<int64_t> step(update_literal.shape().dimensions_size(), 1);
    ShapeUtil::ForEachIndex(update_literal.shape(), base,
                            update_literal.shape().dimensions(), step, func);

    return std::move(result);
  }

  StatusOr<Literal> ElementWiseUnaryOp(
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

  StatusOr<Literal> ElementWiseBinaryOp(
      HloInstruction* instruction,
      const std::function<ElementwiseT(ElementwiseT, ElementwiseT)>&
          binary_op) {
    const auto shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, rhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);

    Literal result(shape);

    TF_RETURN_IF_ERROR(
        result.Populate<ReturnT>([&](absl::Span<const int64_t> multi_index) {
          return ConvertBinaryFunction(binary_op)(
              lhs_literal.Get<ReturnT>(multi_index),
              rhs_literal.Get<ReturnT>(multi_index));
        }));
    return std::move(result);
  }

  template <typename LhsType, typename RhsType, typename EhsType>
  StatusOr<Literal> ElementwiseTernaryOp(
      HloInstruction* instruction,
      const std::function<ReturnT(LhsType, RhsType, EhsType)>& ternary_op) {
    const auto shape = instruction->shape();
    const auto* lhs = instruction->operand(0);
    const auto* rhs = instruction->operand(1);
    const auto* ehs = instruction->operand(2);
    TF_RET_CHECK(ShapeUtil::SameDimensions(shape, lhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(lhs->shape(), rhs->shape()));
    TF_RET_CHECK(ShapeUtil::SameDimensions(rhs->shape(), ehs->shape()));

    const Literal& lhs_literal = parent_->GetEvaluatedLiteralFor(lhs);
    const Literal& rhs_literal = parent_->GetEvaluatedLiteralFor(rhs);
    const Literal& ehs_literal = parent_->GetEvaluatedLiteralFor(ehs);

    Literal result(shape);

    TF_RETURN_IF_ERROR(
        result.Populate<ReturnT>([&](absl::Span<const int64_t> multi_index) {
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
extern template class HloEvaluatorTypedVisitor<uint8_t>;
extern template class HloEvaluatorTypedVisitor<uint32_t>;
extern template class HloEvaluatorTypedVisitor<uint64_t>;
extern template class HloEvaluatorTypedVisitor<int8_t>;
extern template class HloEvaluatorTypedVisitor<int32_t>;
extern template class HloEvaluatorTypedVisitor<int64_t>;
extern template class HloEvaluatorTypedVisitor<Eigen::half, float>;
extern template class HloEvaluatorTypedVisitor<float>;
extern template class HloEvaluatorTypedVisitor<double>;
extern template class HloEvaluatorTypedVisitor<complex64>;
extern template class HloEvaluatorTypedVisitor<complex128>;
extern template class HloEvaluatorTypedVisitor<bfloat16, float>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_EVALUATOR_TYPED_VISITOR_H_
