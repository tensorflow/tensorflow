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

#include <algorithm>
#include <bitset>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
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
struct is_complex_t : std::disjunction<std::is_same<T, complex64>,
                                       std::is_same<T, complex128>> {};

template <typename T>
inline constexpr bool is_complex_v = is_complex_t<T>::value;

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
template <typename T>
auto ToArithmeticSafeType(T t) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<detail::unsigned_promoted_type_t<T>>(t);
  }
  if constexpr (!std::is_integral_v<T>) {
    return std::move(t);
  }
}

// UintWithSize<N> gets an unsigned integer with the given size in bytes.
template <size_t kBytes>
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

template <size_t kBytes>
using UintWithSizeType = typename UintWithSize<kBytes>::type;

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
  template <typename NativeT>
  double GetAsDouble(const Literal& literal,
                     absl::Span<const int64_t> input_index) {
    // Specialization for complex types. In this case it is not possible to
    // static_cast value to a double so just CHECK fail. This method is not used
    // at run-time, but must be available at compile-time to keep the compiler
    // happy.
    if (is_complex_v<NativeT>) {
      LOG(FATAL) << "Trying to get complex literal as double: "
                 << literal.ToString();
    }
    return static_cast<double>(literal.Get<NativeT>(input_index));
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
            typename std::enable_if_t<std::is_unsigned_v<NativeT>>* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return elem_operand;
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_signed_v<NativeT>>* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  Status HandleAbs(HloInstruction* abs) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(abs->operand(0));
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[abs],
        (HloEvaluator::ElementWiseUnaryOpImpl<typename NativeT::value_type,
                                              NativeT>(
            abs, [](NativeT elem_operand) { return std::abs(elem_operand); },
            operand_literal)));

    return OkStatus();
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

  template <typename NativeT,
            typename std::enable_if_t<!is_complex_v<NativeT>>* = nullptr>
  Status HandleRound(HloInstruction* round) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[round],
        ElementWiseUnaryOp(round, [](ElementwiseT elem_operand) {
          return std::round(elem_operand);
        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  Status HandleRound(HloInstruction* round) {
    return UnsupportedTypeError(round);
  }

  Status HandleRound(HloInstruction* round) override {
    return HandleRound<ReturnT>(round);
  }

  template <typename NativeT,
            typename std::enable_if_t<!is_complex_v<NativeT>>* = nullptr>
  Status HandleRoundNearestEven(HloInstruction* round) {
    // TODO(b/228138251): Add support for rounding to nearest even.
    return UnsupportedTypeError(round);
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  Status HandleRoundNearestEven(HloInstruction* round) {
    return UnsupportedTypeError(round);
  }

  Status HandleRoundNearestEven(HloInstruction* round) override {
    return HandleRoundNearestEven<ReturnT>(round);
  }

  template <typename NativeT,
            typename std::enable_if_t<!is_complex_v<NativeT>>* = nullptr>
  Status HandleCeil(HloInstruction* ceil) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[ceil],
                        ElementWiseUnaryOp(ceil, [](ElementwiseT elem_operand) {
                          return std::ceil(elem_operand);
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
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
    return OkStatus();
  }

  Status HandleBitcastConvert(HloInstruction* convert) override {
    const HloInstruction* operand = convert->operand(0);
    TF_ASSIGN_OR_RETURN(Literal result,
                        parent_->GetEvaluatedLiteralFor(operand).BitcastConvert(
                            convert->shape()));

    parent_->evaluated_[convert] = std::move(result);
    return OkStatus();
  }

  Status HandleExp(HloInstruction* exp) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[exp],
                        ElementWiseUnaryOp(exp, [](ElementwiseT elem_operand) {
                          return std::exp(elem_operand);
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<!is_complex_v<NativeT>>* = nullptr>
  Status HandleExpm1(HloInstruction* expm1) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[expm1],
        ElementWiseUnaryOp(expm1, [](ElementwiseT elem_operand) {
          return std::expm1(elem_operand);
        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  Status HandleExpm1(HloInstruction* expm1) {
    return UnsupportedTypeError(expm1);
  }

  Status HandleExpm1(HloInstruction* floor) override {
    return HandleExpm1<ReturnT>(floor);
  }

  template <typename NativeT,
            typename std::enable_if_t<!is_complex_v<NativeT>>* = nullptr>
  Status HandleFloor(HloInstruction* floor) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[floor],
        ElementWiseUnaryOp(floor, [](ElementwiseT elem_operand) {
          return std::floor(elem_operand);
        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
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
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<!is_complex_v<NativeT>>* = nullptr>
  Status HandleLog1p(HloInstruction* log1p) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[log1p],
        ElementWiseUnaryOp(log1p, [](ElementwiseT elem_operand) {
          return std::log1p(elem_operand);
        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  Status HandleLog1p(HloInstruction* log1p) {
    return UnsupportedTypeError(log1p);
  }

  Status HandleLog1p(HloInstruction* log1p) override {
    return HandleLog1p<ReturnT>(log1p);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_integral_v<NativeT> &&
                                  !std::is_same_v<NativeT, bool>>* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return ~elem_operand;
                        }));
    return OkStatus();
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return !elem_operand;
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_same_v<NativeT, bool>>* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[not_],
                        ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
                          return !elem_operand;
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  Status HandleNot(HloInstruction* not_) {
    return UnsupportedTypeError(not_);
  }

  Status HandleNot(HloInstruction* not_) override {
    return HandleNot<ElementwiseT>(not_);
  }

  template <
      typename NativeT,
      typename std::enable_if_t<std::is_signed_v<NativeT> &&
                                !std::is_floating_point_v<NativeT>>* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    using type = std::make_unsigned_t<NativeT>;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(negate, [](ElementwiseT elem_operand) {
          return NativeT(-type(elem_operand));
        }));
    return OkStatus();
  }

  template <typename NativeT, typename std::enable_if_t<
                                  !std::is_signed_v<NativeT> ||
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  Status HandleNegate(HloInstruction* negate) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(
            negate, [](ElementwiseT elem_operand) { return -elem_operand; }));
    return OkStatus();
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
    return OkStatus();
  }

  Status HandleSign(HloInstruction* sign) override {
    using NativeT = ReturnT;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[sign],
        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
          if constexpr (std::is_integral_v<NativeT>) {
            return (ElementwiseT(0) < elem_operand) -
                   (elem_operand < ElementwiseT(0));
          }
          if constexpr (std::is_same_v<NativeT, bfloat16> ||
                        std::is_same_v<NativeT, Eigen::half> ||
                        std::is_floating_point_v<NativeT>) {
            return std::isnan(elem_operand)
                       ? elem_operand
                       : std::copysign(elem_operand != ElementwiseT(0),
                                       elem_operand);
          }
          if constexpr (is_complex_v<NativeT>) {
            auto abs_val = std::abs(elem_operand);
            return 0 == abs_val ? ElementwiseT(0) : elem_operand / abs_val;
          }
        }));
    return OkStatus();
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[atan2],
                        ElementWiseBinaryOp(atan2, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                          return std::atan2(lhs_elem, rhs_elem);
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  Status HandleAtan2(HloInstruction* atan2) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[atan2],
        ElementWiseBinaryOp(atan2, [](ElementwiseT y, ElementwiseT x) {
          // atan2(y,x) = -i * log((x + i * y)/sqrt(x**2+y**2))
          auto i = ElementwiseT(0.0, 1.0);
          return (-i) * (std::log((x + i * y) / std::sqrt(x * x + y * y)));
        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<!std::is_floating_point_v<NativeT> &&
                                      !is_complex_v<NativeT>>* = nullptr>
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
    return OkStatus();
  }

  Status HandleMultiply(HloInstruction* multiply) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(
            multiply, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              return ElementwiseT(ToArithmeticSafeType(lhs_elem) *
                                  ToArithmeticSafeType(rhs_elem));
            }));
    return OkStatus();
  }

  Status HandleSubtract(HloInstruction* subtract) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(
            subtract, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              return ElementwiseT(ToArithmeticSafeType(lhs_elem) -
                                  ToArithmeticSafeType(rhs_elem));
            }));
    return OkStatus();
  }

  Status HandleAdd(HloInstruction* add) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[add],
                        ElementWiseBinaryOp(add, [](ElementwiseT lhs_elem,
                                                    ElementwiseT rhs_elem) {
                          return ElementwiseT(ToArithmeticSafeType(lhs_elem) +
                                              ToArithmeticSafeType(rhs_elem));
                        }));
    return OkStatus();
  }

  Status HandleDivide(HloInstruction* divide) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[divide],
        ElementWiseBinaryOp(
            divide,
            [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) -> ElementwiseT {
              if constexpr (std::is_integral_v<ElementwiseT>) {
                if constexpr (std::is_unsigned_v<ElementwiseT>) {
                  if (rhs_elem == 0) {
                    return std::numeric_limits<ElementwiseT>::max();
                  }
                }
                if constexpr (std::is_signed_v<ElementwiseT>) {
                  if (rhs_elem == 0) {
                    return static_cast<ElementwiseT>(-1);
                  }
                  if (rhs_elem == -1 &&
                      lhs_elem == std::numeric_limits<ElementwiseT>::min()) {
                    return lhs_elem;
                  }
                }
              }
              return lhs_elem / rhs_elem;
            }));
    return OkStatus();
  }

  Status HandleMaximum(HloInstruction* maximum) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[maximum],
          ElementWiseBinaryOp(maximum, [](ElementwiseT lhs, ElementwiseT rhs) {
            if constexpr (std::numeric_limits<ElementwiseT>::has_quiet_NaN) {
              if (std::isnan(lhs)) {
                return lhs;
              }
              if (std::isnan(rhs)) {
                return rhs;
              }
            }
            return std::max(lhs, rhs);
          }));
      return OkStatus();
    }
    return UnsupportedTypeError(maximum);
  }

  Status HandleMinimum(HloInstruction* minimum) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[minimum],
          ElementWiseBinaryOp(minimum, [](ElementwiseT lhs, ElementwiseT rhs) {
            if constexpr (std::numeric_limits<ElementwiseT>::has_quiet_NaN) {
              if (std::isnan(lhs)) {
                return lhs;
              }
              if (std::isnan(rhs)) {
                return rhs;
              }
            }
            return std::min(lhs, rhs);
          }));
      return OkStatus();
    }
    return UnsupportedTypeError(minimum);
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
    return OkStatus();
  }

  Status HandleSqrt(HloInstruction* sqrt) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sqrt],
                        ElementWiseUnaryOp(sqrt, [](ElementwiseT elem_operand) {
                          return std::sqrt(elem_operand);
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
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
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<!is_complex_v<NativeT>>* = nullptr>
  Status HandleCbrt(HloInstruction* cbrt) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[cbrt],
                        ElementWiseUnaryOp(cbrt, [](ElementwiseT elem_operand) {
                          return std::cbrt(elem_operand);
                        }));
    return OkStatus();
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
    return OkStatus();
  }

  Status HandleRemainder(HloInstruction* remainder) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[remainder],
          ElementWiseBinaryOp(
              remainder,
              [](ElementwiseT lhs_el, ElementwiseT rhs_el) -> ElementwiseT {
                if constexpr (std::is_integral_v<ElementwiseT>) {
                  if (rhs_el == 0) {
                    return lhs_el;
                  }
                  if constexpr (std::is_signed_v<ElementwiseT>) {
                    if (rhs_el == -1 &&
                        lhs_el == std::numeric_limits<ElementwiseT>::min()) {
                      return 0;
                    }
                  }
                  return lhs_el % rhs_el;
                }
                if constexpr (std::is_floating_point_v<ElementwiseT>) {
                  return std::fmod(lhs_el, rhs_el);
                }
              }));
      return OkStatus();
    }
    return UnsupportedTypeError(remainder);
  }

  Status HandleAnd(HloInstruction* and_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[and_inst],
          ElementWiseBinaryOp(and_inst,
                              [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                return lhs_el & rhs_el;
                              }));
      return OkStatus();
    }
    return UnsupportedTypeError(and_inst);
  }

  Status HandleOr(HloInstruction* or_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[or_inst],
                          ElementWiseBinaryOp(or_inst, [](ElementwiseT lhs_el,
                                                          ElementwiseT rhs_el) {
                            return lhs_el | rhs_el;
                          }));
      return OkStatus();
    }
    return UnsupportedTypeError(or_inst);
  }

  Status HandleXor(HloInstruction* xor_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[xor_inst],
          ElementWiseBinaryOp(xor_inst,
                              [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                return lhs_el ^ rhs_el;
                              }));
      return OkStatus();
    }
    return UnsupportedTypeError(xor_inst);
  }

  Status HandleShiftLeft(HloInstruction* shl) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[shl],
                          ElementWiseBinaryOp(shl, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                            return IsShiftOutOfBounds<ElementwiseT>(rhs_elem)
                                       ? 0
                                       : (lhs_elem << rhs_elem);
                          }));
      return OkStatus();
    }
    return UnsupportedTypeError(shl);
  }

  Status HandleShiftRightArithmetic(HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using SignedT = std::make_signed_t<ElementwiseT>;
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[shr],
          ElementWiseBinaryOp(
              shr, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                SignedT lhs_signed = static_cast<SignedT>(lhs_elem);
                if (IsShiftOutOfBounds<ElementwiseT>(rhs_elem)) {
                  return lhs_signed < 0 ? static_cast<SignedT>(-1) : 0;
                } else {
                  return lhs_signed >> rhs_elem;
                }
              }));
      return OkStatus();
    }
    return UnsupportedTypeError(shr);
  }

  Status HandleShiftRightLogical(HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using UnsignedT = std::make_unsigned_t<ElementwiseT>;
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[shr],
                          ElementWiseBinaryOp(shr, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                            // If shift amount is greater than the number of
                            // bits, then return 0.
                            if (IsShiftOutOfBounds<ElementwiseT>(rhs_elem)) {
                              return static_cast<ElementwiseT>(0);
                            }
                            return static_cast<ElementwiseT>(
                                static_cast<UnsignedT>(lhs_elem) >> rhs_elem);
                          }));
      return OkStatus();
    }
    return UnsupportedTypeError(shr);
  }

  Status HandleClamp(HloInstruction* clamp) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      auto clamp_op = [](ElementwiseT low, ElementwiseT value,
                         ElementwiseT high) {
        if constexpr (std::numeric_limits<ElementwiseT>::has_quiet_NaN) {
          if (std::isnan(low)) {
            return low;
          }
          if (std::isnan(value)) {
            return value;
          }
          if (std::isnan(high)) {
            return high;
          }
        }
        return std::min(high, std::max(value, low));
      };
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[clamp],
          ElementwiseTernaryOp(clamp,
                               std::move(ConvertTernaryFunction(clamp_op))));
      return OkStatus();
    }
    return UnsupportedTypeError(clamp);
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
    return OkStatus();
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
    return OkStatus();
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
                 batch_group_count](const absl::Span<const int64_t> out_index,
                                    int /*thread_id*/) {
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
    return OkStatus();
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

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_same_v<NativeT, float>>* = nullptr>
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
    return OkStatus();
  }

  template <typename NativeT, typename std::enable_if_t<
                                  !std::is_same_v<NativeT, float>>* = nullptr>
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
        [&](absl::Span<const int64_t> result_index, int /*thread_id*/) {
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
    return OkStatus();
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
    return OkStatus();
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

    return OkStatus();
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

    return OkStatus();
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
            arg_literals.push_back(
                LiteralUtil::GetScalarLiteral(arg_literal, multi_index));
          }

          Literal computed_result =
              embedded_evaluator.Evaluate(*computation, arg_literals).value();
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

    return OkStatus();
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
      std::optional<ReturnT> selected_val;
      std::optional<DimensionVector> selected_index;

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](absl::Span<const int64_t> operand_index) {
            auto curr_val = operand_literal.Get<ReturnT>(operand_index);
            if (!selected_val) {
              selected_val = curr_val;
              selected_index.emplace(operand_index.begin(),
                                     operand_index.end());
            }
            curr_val_literal.Set({}, curr_val);
            selected_val_literal.Set({}, *selected_val);
            Literal computed_result =
                embedded_evaluator
                    .Evaluate(*select,
                              {&selected_val_literal, &curr_val_literal})
                    .value();
            bool selected = !computed_result.Get<bool>({});
            if (selected) {
              selected_val = curr_val;
              selected_index.emplace(operand_index.begin(),
                                     operand_index.end());
            }
            embedded_evaluator.ResetVisitStates();
          });

      IterateThroughWindow(
          window_shape, window, operand_literal.shape(), source_index,
          [&](absl::Span<const int64_t> operand_index) {
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
                      .value();
              result.Set(operand_index, computed_result.Get<ReturnT>({}));
              // Clear visit states so that the we can use the evaluator again
              // on the same computation.
              embedded_evaluator.ResetVisitStates();
            }
          });
    } while (
        IndexUtil::BumpIndices(source->shape(), absl::MakeSpan(source_index)));

    parent_->evaluated_[select_and_scatter] = std::move(result);
    return OkStatus();
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

    const int num_threads = tensorflow::port::MaxParallelism() + 1;
    std::vector<std::unique_ptr<HloEvaluator>> embedded_evaluators;
    embedded_evaluators.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      embedded_evaluators.push_back(
          parent_->CreateEmbedded(parent_->max_loop_iterations_));
    }

    // For each resulting dimension, calculate and assign computed value.
    auto evaluate_impl = [&init_literal_vec, &window_shape, &window,
                          &input_literal_vec, &embedded_evaluators, function,
                          &inferred_return_shape](
                             absl::Span<const int64_t> output_index,
                             int thread_id) -> absl::InlinedVector<Literal, 2> {
      const int embedded_evaluator_index = thread_id + 1;
      CHECK_GE(embedded_evaluator_index, 0);
      CHECK_LT(embedded_evaluator_index, embedded_evaluators.size());
      HloEvaluator& embedded_evaluator =
          *embedded_evaluators[embedded_evaluator_index];
      absl::InlinedVector<Literal, 2> computed_result;
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
            absl::InlinedVector<Literal, 2> curr_val_literal_vec;
            curr_val_literal_vec.reserve(input_literal_vec.size());
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
            computed_result[0] =
                embedded_evaluator.Evaluate(*function, args).value();
            VLOG(2) << "Computed result:" << computed_result[0].ToString()
                    << "\n";
            // Clear visit states so that the we can use the evaluate again
            // on the same computation.
            embedded_evaluator.ResetVisitStates();
            if (inferred_return_shape.IsTuple()) {
              auto decomposed = computed_result[0].DecomposeTuple();
              computed_result.clear();
              computed_result.reserve(decomposed.size());
              for (int i = 0; i < decomposed.size(); ++i) {
                computed_result.push_back(std::move(decomposed[i]));
              }
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
      ShapeUtil::ForEachIndexParallel(
          inferred_return_shape.tuple_shapes(0),
          [&results, &evaluate_impl](absl::Span<const int64_t> output_index,
                                     int thread_id) -> bool {
            absl::InlinedVector<Literal, 2> computed_result_vec =
                evaluate_impl(output_index, thread_id);
            for (int i = 0; i < computed_result_vec.size(); ++i) {
              // We are reading from `computed_result_vec[i]` at the top-level
              // literal index and writing to `results[i]` at `output_index`.
              // This is thread-safe because:
              //  - `results[i]` is not changing size.
              //  - `computed_result_vec[i]` is thread-local.
              //  - There is exactly one write to `results[i]` for each
              //    `output_index`.
              TF_CHECK_OK(results[i].CopyElementFrom(computed_result_vec[i], {},
                                                     output_index));
            }
            return true;
          });
      result = Literal::MoveIntoTuple(absl::MakeSpan(results));
      VLOG(2) << "Final result is:" << result.ToString() << "\n";
    } else {
      TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
          [&evaluate_impl](absl::Span<const int64_t> output_index,
                           int thread_id) {
            return evaluate_impl(output_index, thread_id)[0]
                .template Get<ReturnT>({});
          }));
    }
    VLOG(2) << "Final result is:" << result.ToString() << "\n";
    parent_->evaluated_[reduce_window] = std::move(result);
    return OkStatus();
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
    return OkStatus();
  }

  // Enable CLZ only for int32_t, uint32_t, int64_t and uint64_t.
  template <typename NativeT,
            typename std::enable_if_t<!std::is_integral_v<NativeT> ||
                                      std::is_same_v<NativeT, bool>>* = nullptr>
  Status HandleClz(HloInstruction* clz) {
    return UnsupportedTypeError(clz);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_integral_v<NativeT> &&
                                  !std::is_same_v<NativeT, bool>>* = nullptr>
  Status HandleClz(HloInstruction* clz) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[clz],
        ElementWiseUnaryOp(clz, [](ElementwiseT elem_operand) {
          using UnsignedElementwiseT = std::make_unsigned_t<ElementwiseT>;
          return (std::numeric_limits<UnsignedElementwiseT>::digits - 1) -
                 Log2Floor<UnsignedElementwiseT>(elem_operand);
        }));
    return OkStatus();
  }

  Status HandleClz(HloInstruction* clz) override {
    return HandleClz<ElementwiseT>(clz);
  }

  template <typename NativeT,
            typename std::enable_if_t<!std::is_integral_v<NativeT> ||
                                      std::is_same_v<NativeT, bool>>* = nullptr>
  Status HandlePopulationCount(HloInstruction* popcnt) {
    return UnsupportedTypeError(popcnt);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_integral_v<NativeT> &&
                                  !std::is_same_v<NativeT, bool>>* = nullptr>
  Status HandlePopulationCount(HloInstruction* popcnt) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[popcnt],
        ElementWiseUnaryOp(popcnt, [](ElementwiseT elem_operand) {
          return std::bitset<CHAR_BIT * sizeof elem_operand>(elem_operand)
              .count();
        }));
    return OkStatus();
  }

  Status HandlePopulationCount(HloInstruction* popcnt) override {
    return HandlePopulationCount<ElementwiseT>(popcnt);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  Status HandleSin(HloInstruction* sin) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sin],
                        ElementWiseUnaryOp(sin, [](ElementwiseT elem_operand) {
                          return std::sin(elem_operand);
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_integral_v<NativeT> ||
                                      is_complex_v<NativeT>>* = nullptr>
  Status HandleSin(HloInstruction* sin) {
    return UnsupportedTypeError(sin);
  }

  Status HandleSin(HloInstruction* sin) override {
    return HandleSin<ElementwiseT>(sin);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  Status HandleCos(HloInstruction* cos) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[cos],
                        ElementWiseUnaryOp(cos, [](ElementwiseT elem_operand) {
                          return std::cos(elem_operand);
                        }));
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_integral_v<NativeT> ||
                                      is_complex_v<NativeT>>* = nullptr>
  Status HandleCos(HloInstruction* cos) {
    return UnsupportedTypeError(cos);
  }

  Status HandleCos(HloInstruction* cos) override {
    return HandleCos<ElementwiseT>(cos);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_same_v<NativeT, float> ||
                                  std::is_same_v<NativeT, double>>* = nullptr>
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

          using Uint = UintWithSizeType<sizeof(NativeT)>;
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
    return OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_integral_v<NativeT> ||
                                      is_complex_v<NativeT>>* = nullptr>
  Status HandleReducePrecision(HloInstruction* reduce_precision) {
    return UnsupportedTypeError(reduce_precision);
  }

  Status HandleReducePrecision(HloInstruction* reduce_precision) override {
    return HandleReducePrecision<ElementwiseT>(reduce_precision);
  }

  template <typename NativeT,
            typename std::enable_if_t<
                std::is_same_v<NativeT, bfloat16> ||
                std::is_same_v<NativeT, Eigen::half> ||
                std::is_integral_v<NativeT> || is_complex_v<NativeT> ||
                std::is_floating_point_v<NativeT>>* = nullptr>
  Status HandleIota(HloInstruction* instruction) {
    auto* iota = Cast<HloIotaInstruction>(instruction);

    Literal result(iota->shape());
    ShapeUtil::ForEachIndex(iota->shape(), [&](absl::Span<const int64_t> idx) {
      result.Set(idx, static_cast<NativeT>(idx[iota->iota_dimension()]));
      return true;
    });
    parent_->evaluated_[iota] = std::move(result);
    return OkStatus();
  }
  template <typename NativeT,
            typename std::enable_if_t<
                !(std::is_same_v<NativeT, bfloat16> ||
                  std::is_same_v<NativeT, Eigen::half> ||
                  std::is_integral_v<NativeT> || is_complex_v<NativeT> ||
                  std::is_floating_point_v<NativeT>)>* = nullptr>
  Status HandleIota(HloInstruction* iota) {
    return UnsupportedTypeError(iota);
  }
  Status HandleIota(HloInstruction* iota) override {
    return HandleIota<ReturnT>(iota);
  }

  template <typename NativeT,
            typename std::enable_if_t<!(std::is_integral_v<NativeT> ||
                                        std::is_floating_point_v<NativeT>)>* =
                nullptr>
  Status HandleRng(HloInstruction* random) {
    return UnsupportedTypeError(random);
  }
  template <
      typename NativeT,
      typename std::enable_if_t<(std::is_floating_point_v<NativeT>)>* = nullptr>
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
    return OkStatus();
  }
  template <typename NativeT,
            typename std::enable_if_t<(std::is_integral_v<NativeT>)>* = nullptr>
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
    return OkStatus();
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
      const std::function<void(absl::Span<const int64_t>)>& f) {
    const int64_t rank = base_shape.rank();
    DimensionVector window_index(rank);
    std::fill(window_index.begin(), window_index.end(), 0);
    do {
      DimensionVector base_index(rank);
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
    const auto& shape = instruction->shape();
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
    const auto& shape = instruction->shape();
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
    using UnsignedT = std::make_unsigned_t<NativeT>;
    UnsignedT lhs_bits_unsigned = std::numeric_limits<UnsignedT>::digits;
    UnsignedT rhs_unsigned = static_cast<UnsignedT>(rhs);
    return rhs_unsigned >= lhs_bits_unsigned;
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
