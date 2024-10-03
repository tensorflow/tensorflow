/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_HLO_EVALUATOR_HLO_EVALUATOR_TYPED_VISITOR_H_
#define XLA_HLO_EVALUATOR_HLO_EVALUATOR_TYPED_VISITOR_H_

#include <fenv.h>  // NOLINT

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/index_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/shape_inference.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

template <typename T>
T Nibble0(T t) {
  if constexpr (std::is_integral_v<T>) {
    constexpr auto shift = (8 * sizeof(T)) - 4;
    return (t << shift) >> shift;
  }
  return t;
}

template <typename T>
T Nibble1(T t) {
  if constexpr (std::is_integral_v<T>) {
    return t >> 4;
  }
  return t;
}

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
//
// NOTE: Prefer putting new implementation to HloEvalator rather than
// HloEvaluatorTypedVisitor whenever possible, because this class is templated
// for all primitive types and is an order of magnitude larger in code size as
// well as compile time. Only put op handling that involves compute using native
// C++ types here, such as elementwise ops with compute, convolution, dot, etc.
template <typename ReturnT, typename ElementwiseT = ReturnT>
class HloEvaluatorTypedVisitor : public ConstDfsHloVisitorWithDefault {
 private:
  ABSL_ATTRIBUTE_NOINLINE absl::Status UnsupportedTypeError(
      const HloInstruction* instruction) {
    return InvalidArgument(
        "Unsupported type for %s: %s", HloOpcodeString(instruction->opcode()),
        PrimitiveType_Name(instruction->shape().element_type()));
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

  absl::Status DefaultAction(const HloInstruction* hlo_instruction) override {
    return Unimplemented("unhandled HLO ops for HloEvaluator: %s.",
                         HloOpcodeString(hlo_instruction->opcode()));
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_unsigned_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return elem_operand;
                        }));
    return absl::OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_signed_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[abs],
                        ElementWiseUnaryOp(abs, [](NativeT elem_operand) {
                          return std::abs(elem_operand);
                        }));
    return absl::OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(abs->operand(0));
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[abs],
        (HloEvaluator::ElementWiseUnaryOpImpl<typename NativeT::value_type,
                                              NativeT>(
            abs, [](NativeT elem_operand) { return std::abs(elem_operand); },
            operand_literal)));

    return absl::OkStatus();
  }

  absl::Status HandleAbs(const HloInstruction* abs) override {
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

  absl::Status HandleRound(const HloInstruction* round) override {
    if constexpr (!is_complex_v<ReturnT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[round],
          ElementWiseUnaryOp(round, [](ElementwiseT elem_operand) {
            return std::round(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(round);
  }

  absl::Status HandleRoundNearestEven(const HloInstruction* round) override {
    if constexpr (!is_complex_v<ReturnT>) {
      // Verify the current rounding direction.
      TF_RET_CHECK(fegetround() == FE_TONEAREST);
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[round],
          ElementWiseUnaryOp(round, [](ElementwiseT elem_operand) {
            return std::nearbyint(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(round);
  }

  absl::Status HandleCeil(const HloInstruction* ceil) override {
    if constexpr (!is_complex_v<ReturnT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[ceil],
          ElementWiseUnaryOp(ceil, [](ElementwiseT elem_operand) {
            return std::ceil(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(ceil);
  }

  absl::Status HandleErf(const HloInstruction* erf) override {
    if constexpr (!is_complex_v<ReturnT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[erf],
          ElementWiseUnaryOp(erf, [](ElementwiseT elem_operand) {
            return std::erf(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(erf);
  }

  absl::Status HandleExp(const HloInstruction* exp) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[exp],
                        ElementWiseUnaryOp(exp, [](ElementwiseT elem_operand) {
                          return std::exp(elem_operand);
                        }));
    return absl::OkStatus();
  }

  absl::Status HandleExpm1(const HloInstruction* expm1) override {
    if constexpr (!is_complex_v<ReturnT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[expm1],
          ElementWiseUnaryOp(expm1, [](ElementwiseT elem_operand) {
            return std::expm1(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(expm1);
  }

  absl::Status HandleFloor(const HloInstruction* floor) override {
    if constexpr (!is_complex_v<ReturnT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[floor],
          ElementWiseUnaryOp(floor, [](ElementwiseT elem_operand) {
            return std::floor(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(floor);
  }

  absl::Status HandleLog(const HloInstruction* log) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[log],
                        ElementWiseUnaryOp(log, [](ElementwiseT elem_operand) {
                          return std::log(elem_operand);
                        }));
    return absl::OkStatus();
  }

  absl::Status HandleLog1p(const HloInstruction* log1p) override {
    if constexpr (!is_complex_v<ReturnT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[log1p],
          ElementWiseUnaryOp(log1p, [](ElementwiseT elem_operand) {
            return std::log1p(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(log1p);
  }

  absl::Status HandleNot(const HloInstruction* not_) override {
    if constexpr (std::is_arithmetic_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[not_],
          ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
            if constexpr (std::is_floating_point_v<ElementwiseT> ||
                          std::is_same_v<ElementwiseT, bool>) {
              return !elem_operand;
            } else {
              static_assert(std::is_integral_v<ElementwiseT>);
              return ~elem_operand;
            }
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(not_);
  }

  template <
      typename NativeT,
      typename std::enable_if_t<std::is_signed_v<NativeT> &&
                                !std::is_floating_point_v<NativeT>>* = nullptr>
  absl::Status HandleNegate(const HloInstruction* negate) {
    using type = std::make_unsigned_t<NativeT>;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(negate, [](ElementwiseT elem_operand) {
          return NativeT(-type(elem_operand));
        }));
    return absl::OkStatus();
  }

  template <typename NativeT, typename std::enable_if_t<
                                  !std::is_signed_v<NativeT> ||
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  absl::Status HandleNegate(const HloInstruction* negate) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[negate],
        ElementWiseUnaryOp(
            negate, [](ElementwiseT elem_operand) { return -elem_operand; }));
    return absl::OkStatus();
  }

  absl::Status HandleNegate(const HloInstruction* negate) override {
    return HandleNegate<ReturnT>(negate);
  }

  absl::Status HandleLogistic(const HloInstruction* logistic) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[logistic],
        ElementWiseUnaryOp(logistic, [](ElementwiseT elem_operand) {
          return static_cast<ElementwiseT>(1) /
                 (static_cast<ElementwiseT>(1) + std::exp(-elem_operand));
        }));
    return absl::OkStatus();
  }

  absl::Status HandleSign(const HloInstruction* sign) override {
    using NativeT = ElementwiseT;
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[sign],
        ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
          if constexpr (std::is_integral_v<NativeT>) {
            return (ElementwiseT(0) < elem_operand) -
                   (elem_operand < ElementwiseT(0));
          }
          if constexpr (std::is_floating_point_v<ElementwiseT>) {
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
    return absl::OkStatus();
  }

  absl::Status HandleAtan2(const HloInstruction* atan2) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[atan2],
                          ElementWiseBinaryOp(atan2, [](ElementwiseT lhs_elem,
                                                        ElementwiseT rhs_elem) {
                            return std::atan2(lhs_elem, rhs_elem);
                          }));
      return absl::OkStatus();
    }
    if constexpr (is_complex_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[atan2],
          ElementWiseBinaryOp(atan2, [](ElementwiseT y, ElementwiseT x) {
            // atan2(y,x) = -i * log((x + i * y)/sqrt(x**2+y**2))
            auto i = ElementwiseT(0.0, 1.0);
            return (-i) * (std::log((x + i * y) / std::sqrt(x * x + y * y)));
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(atan2);
  }

  absl::Status HandleTanh(const HloInstruction* tanh) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[tanh],
                        ElementWiseUnaryOp(tanh, [](ElementwiseT elem_operand) {
                          return std::tanh(elem_operand);
                        }));
    return absl::OkStatus();
  }

  absl::Status HandleMultiply(const HloInstruction* multiply) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[multiply],
        ElementWiseBinaryOp(
            multiply, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              return ElementwiseT(ToArithmeticSafeType(lhs_elem) *
                                  ToArithmeticSafeType(rhs_elem));
            }));
    return absl::OkStatus();
  }

  absl::Status HandleSubtract(const HloInstruction* subtract) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[subtract],
        ElementWiseBinaryOp(
            subtract, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
              return ElementwiseT(ToArithmeticSafeType(lhs_elem) -
                                  ToArithmeticSafeType(rhs_elem));
            }));
    return absl::OkStatus();
  }

  absl::Status HandleAdd(const HloInstruction* add) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[add],
                        ElementWiseBinaryOp(add, [](ElementwiseT lhs_elem,
                                                    ElementwiseT rhs_elem) {
                          return ElementwiseT(ToArithmeticSafeType(lhs_elem) +
                                              ToArithmeticSafeType(rhs_elem));
                        }));
    return absl::OkStatus();
  }

  absl::Status HandleDivide(const HloInstruction* divide) override {
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
    return absl::OkStatus();
  }

  absl::Status HandleMaximum(const HloInstruction* maximum) override {
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
      return absl::OkStatus();
    }
    return UnsupportedTypeError(maximum);
  }

  absl::Status HandleMinimum(const HloInstruction* minimum) override {
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
      return absl::OkStatus();
    }
    return UnsupportedTypeError(minimum);
  }

  absl::Status HandlePower(const HloInstruction* power) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[power],
        ElementWiseBinaryOp(
            power, [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
              // Case 0: 1^x = 1 and x^0 = 1, regardless of X, see
              // Branch Cuts for Complex Elementary Functions or Much Ado About
              // Nothing's Sign Bit, W. Kahan, Section 10.
              if (lhs_el == ElementwiseT(1) || rhs_el == ElementwiseT(0)) {
                return static_cast<ElementwiseT>(1);
              }
              // Case 1:
              // 1. inf^(a + 0i) = inf, if a > 0.
              // 2. inf^(a + 0i) = 0, if a < 0.
              if constexpr (is_complex_v<ElementwiseT>) {
                auto is_positive_infinity = [](ElementwiseT c) {
                  return c.imag() == 0 && c.real() > 0 && std::isinf(c.real());
                };
                auto is_positive_real = [](ElementwiseT c) {
                  return c.real() > 0 && c.imag() == 0;
                };
                auto is_negative_real = [](ElementwiseT c) {
                  return c.real() < 0 && c.imag() == 0;
                };
                if (is_positive_infinity(lhs_el) && is_positive_real(rhs_el)) {
                  return static_cast<ElementwiseT>(lhs_el);
                }
                if (is_positive_infinity(lhs_el) && is_negative_real(rhs_el)) {
                  return static_cast<ElementwiseT>(0);
                }
              }
              // Case 2:
              // Fallback to pow.
              if constexpr (std::is_same_v<ElementwiseT, bool>) {
                return lhs_el || !rhs_el;
              } else if constexpr (std::is_integral_v<ElementwiseT>) {
                if constexpr (std::is_signed_v<ElementwiseT>) {
                  if (rhs_el < static_cast<ElementwiseT>(0)) {
                    return static_cast<ElementwiseT>(
                        lhs_el == static_cast<ElementwiseT>(1) ? 1 : 0);
                  }
                }
                return static_cast<ElementwiseT>(
                    IPow<std::make_unsigned_t<ElementwiseT>>(lhs_el, rhs_el));
              } else {
                return static_cast<ElementwiseT>(std::pow(lhs_el, rhs_el));
              }
            }));
    return absl::OkStatus();
  }

  absl::Status HandleSqrt(const HloInstruction* sqrt) override {
    TF_ASSIGN_OR_RETURN(parent_->evaluated_[sqrt],
                        ElementWiseUnaryOp(sqrt, [](ElementwiseT elem_operand) {
                          return std::sqrt(elem_operand);
                        }));
    return absl::OkStatus();
  }

  absl::Status HandleCbrt(const HloInstruction* cbrt) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[cbrt],
          ElementWiseUnaryOp(cbrt, [](ElementwiseT elem_operand) {
            return std::cbrt(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(cbrt);
  }

  absl::Status HandleRsqrt(const HloInstruction* rsqrt) override {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[rsqrt],
        ElementWiseUnaryOp(rsqrt, [](ElementwiseT elem_operand) {
          return static_cast<ElementwiseT>(1) / std::sqrt(elem_operand);
        }));
    return absl::OkStatus();
  }

  absl::Status HandleRemainder(const HloInstruction* remainder) override {
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
      return absl::OkStatus();
    }
    return UnsupportedTypeError(remainder);
  }

  absl::Status HandleAnd(const HloInstruction* and_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[and_inst],
          ElementWiseBinaryOp(and_inst,
                              [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                return lhs_el & rhs_el;
                              }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(and_inst);
  }

  absl::Status HandleOr(const HloInstruction* or_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[or_inst],
                          ElementWiseBinaryOp(or_inst, [](ElementwiseT lhs_el,
                                                          ElementwiseT rhs_el) {
                            return lhs_el | rhs_el;
                          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(or_inst);
  }

  absl::Status HandleXor(const HloInstruction* xor_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[xor_inst],
          ElementWiseBinaryOp(xor_inst,
                              [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                return lhs_el ^ rhs_el;
                              }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(xor_inst);
  }

  absl::Status HandleShiftLeft(const HloInstruction* shl) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[shl],
                          ElementWiseBinaryOp(shl, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                            return IsShiftOutOfBounds<ElementwiseT>(rhs_elem)
                                       ? 0
                                       : (lhs_elem << rhs_elem);
                          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(shl);
  }

  absl::Status HandleShiftRightArithmetic(const HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using SignedT = make_specialized_signed_t<ReturnT>;
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[shr],
          ElementWiseBinaryOp(
              shr, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
                SignedT lhs_signed = static_cast<SignedT>(lhs_elem);
                if (IsShiftOutOfBounds<ReturnT>(rhs_elem)) {
                  return lhs_signed < 0 ? static_cast<ElementwiseT>(-1) : 0;
                } else {
                  return static_cast<ElementwiseT>(lhs_signed >> rhs_elem);
                }
              }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(shr);
  }

  absl::Status HandleShiftRightLogical(const HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using UnsignedT = make_specialized_unsigned_t<ReturnT>;
      TF_ASSIGN_OR_RETURN(parent_->evaluated_[shr],
                          ElementWiseBinaryOp(shr, [](ElementwiseT lhs_elem,
                                                      ElementwiseT rhs_elem) {
                            // If shift amount is greater than the number of
                            // bits, then return 0.
                            if (IsShiftOutOfBounds<ReturnT>(rhs_elem)) {
                              return static_cast<ElementwiseT>(0);
                            }
                            return static_cast<ElementwiseT>(
                                static_cast<UnsignedT>(lhs_elem) >> rhs_elem);
                          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(shr);
  }

  absl::Status HandleClamp(const HloInstruction* clamp) override {
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
      return absl::OkStatus();
    }
    return UnsupportedTypeError(clamp);
  }

  absl::Status HandleSelect(const HloInstruction* select) override {
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
    return absl::OkStatus();
  }

  absl::Status HandleConvolutionWithLiterals(const HloInstruction* conv,
                                             const Literal& lhs_literal,
                                             const Literal& rhs_literal) {
    const auto& window = conv->window();
    const Shape& result_shape = conv->shape();
    const Shape& lhs_shape = lhs_literal.shape();
    const Shape& rhs_shape = rhs_literal.shape();
    const auto packed_nibble_count =
        absl::c_count(conv->precision_config().operand_precision(),
                      PrecisionConfig::PACKED_NIBBLE);
    CHECK_NE(packed_nibble_count, 1);
    const bool is_packed_nibble = packed_nibble_count == 2;

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

    DimensionVector lhs_dim_multipliers =
        HloEvaluator::MakeDimMultipliers(lhs_shape);
    DimensionVector rhs_dim_multipliers =
        HloEvaluator::MakeDimMultipliers(rhs_shape);

    auto lhs_literal_data = lhs_literal.data<ReturnT>();
    auto rhs_literal_data = rhs_literal.data<ReturnT>();

    const int64_t feature_group_count = conv->feature_group_count();
    const int64_t batch_group_count = conv->batch_group_count();

    auto func = [&window_shape, &dnums, &lhs_shape, &rhs_shape, &window,
                 &lhs_dim_multipliers, &rhs_dim_multipliers, lhs_literal_data,
                 rhs_literal_data, feature_group_count, batch_group_count,
                 is_packed_nibble, result_shape,
                 this](const absl::Span<const int64_t> out_index,
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

      const int64_t depthwise_multiplier = output_z_size / batch_group_count;
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
          lhs_linear_index += (batch_group_index * batch_group_size) *
                              lhs_dim_multipliers[input_batch_dim];

          lhs_linear_index += iz * lhs_dim_multipliers[input_z_dim];
          int64_t rhs_linear_index = rhs_linear_spatial_index;

          rhs_linear_index += out_index[output_z_dim] *
                              rhs_dim_multipliers[kernel_output_z_dim];
          rhs_linear_index += rhs_iz * rhs_dim_multipliers[kernel_input_z_dim];
          auto lhs =
              static_cast<ElementwiseT>(lhs_literal_data[lhs_linear_index]);
          auto rhs =
              static_cast<ElementwiseT>(rhs_literal_data[rhs_linear_index]);
          if (is_packed_nibble) {
            auto lhs_n0 = ToArithmeticSafeType(Nibble0(lhs));
            auto lhs_n1 = ToArithmeticSafeType(Nibble1(lhs));
            auto rhs_n0 = ToArithmeticSafeType(Nibble0(rhs));
            auto rhs_n1 = ToArithmeticSafeType(Nibble1(rhs));
            result_val += (lhs_n0 * rhs_n0) + (lhs_n1 * rhs_n1);
          } else {
            result_val += ToArithmeticSafeType(lhs) * ToArithmeticSafeType(rhs);

            if (parent_->trace_mac_handler_ != nullptr) {
              const int64_t result_linear_index =
                  IndexUtil::MultidimensionalIndexToLinearIndex(result_shape,
                                                                out_index);

              parent_->trace_mac_handler_(result_linear_index, lhs_linear_index,
                                          rhs_linear_index);
            }
          }
        }
      cnt: {}
      } while (IndexUtil::BumpIndices(window_shape,
                                      absl::MakeSpan(rhs_spatial_index)));

      if constexpr (std::is_integral_v<ReturnT>) {
        auto l = static_cast<ElementwiseT>(std::numeric_limits<ReturnT>::min());
        auto h = static_cast<ElementwiseT>(std::numeric_limits<ReturnT>::max());
        result_val = std::max(l, std::min(h, result_val));
      }
      return static_cast<ReturnT>(result_val);
    };

    Literal result(result_shape);
    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(func));

    parent_->evaluated_[conv] = std::move(result);
    return absl::OkStatus();
  }

  absl::Status HandleConvolution(const HloInstruction* conv) override {
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
          conv, lhs_literal.Convert(result_shape.element_type()).value(),
          rhs_literal);
    }
    if (lhs_same) {
      return HandleConvolutionWithLiterals(
          conv, lhs_literal,
          rhs_literal.Convert(result_shape.element_type()).value());
    }
    return HandleConvolutionWithLiterals(
        conv, lhs_literal.Convert(result_shape.element_type()).value(),
        rhs_literal.Convert(result_shape.element_type()).value());
  }

  absl::Status HandleDot(const HloInstruction* dot) override {
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
  absl::Status HandleDot(const HloInstruction* dot) {
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
        parent_->GetEvaluatedLiteralFor(lhs).Convert(native_ty).value();
    Literal rhs_literal =
        parent_->GetEvaluatedLiteralFor(rhs).Convert(native_ty).value();
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
        std::move(result).Convert(dot->shape().element_type()).value();
    return absl::OkStatus();
  }

  template <typename NativeT, typename std::enable_if_t<
                                  !std::is_same_v<NativeT, float>>* = nullptr>
  absl::Status HandleDot(const HloInstruction* dot) {
    return HandleDotSlowPath(dot);
  }

  absl::Status HandleDotSlowPathWithLiterals(const HloInstruction* dot,
                                             const Literal& lhs_literal,
                                             const Literal& rhs_literal) {
    const auto& dnums = dot->dot_dimension_numbers();

    const auto lhs_rank = lhs_literal.shape().rank();
    const auto rhs_rank = rhs_literal.shape().rank();

    CHECK(ShapeUtil::SameElementType(lhs_literal.shape(), rhs_literal.shape()));
    CHECK(ShapeUtil::SameElementType(lhs_literal.shape(), dot->shape()));

    const auto packed_nibble_count =
        absl::c_count(dot->precision_config().operand_precision(),
                      PrecisionConfig::PACKED_NIBBLE);
    CHECK_NE(packed_nibble_count, 1);
    const bool is_packed_nibble = packed_nibble_count == 2;
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

    DimensionVector contracting_dim_sizes;
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
            const auto lhs =
                static_cast<ElementwiseT>(lhs_literal.Get<ReturnT>(lhs_index));
            const auto rhs =
                static_cast<ElementwiseT>(rhs_literal.Get<ReturnT>(rhs_index));
            if (is_packed_nibble) {
              auto lhs_n0 = ToArithmeticSafeType(Nibble0(lhs));
              auto lhs_n1 = ToArithmeticSafeType(Nibble1(lhs));
              auto rhs_n0 = ToArithmeticSafeType(Nibble0(rhs));
              auto rhs_n1 = ToArithmeticSafeType(Nibble1(rhs));
              result_val += (lhs_n0 * rhs_n0) + (lhs_n1 * rhs_n1);
            } else {
              result_val +=
                  ToArithmeticSafeType(lhs) * ToArithmeticSafeType(rhs);

              if (parent_->trace_mac_handler_ != nullptr) {
                const int64_t result_linear_index =
                    IndexUtil::MultidimensionalIndexToLinearIndex(dot->shape(),
                                                                  result_index);
                const int64_t lhs_linear_index =
                    IndexUtil::MultidimensionalIndexToLinearIndex(
                        lhs_literal.shape(), lhs_index);
                const int64_t rhs_linear_index =
                    IndexUtil::MultidimensionalIndexToLinearIndex(
                        rhs_literal.shape(), rhs_index);

                parent_->trace_mac_handler_(result_linear_index,
                                            lhs_linear_index, rhs_linear_index);
              }
            }

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
    return absl::OkStatus();
  }

  absl::Status HandleDotSlowPath(const HloInstruction* dot) {
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
          rhs_literal.Convert(dot->shape().element_type()).value());
    }
    if (rhs_same) {
      return HandleDotSlowPathWithLiterals(
          dot, lhs_literal.Convert(dot->shape().element_type()).value(),
          rhs_literal);
    }
    return HandleDotSlowPathWithLiterals(
        dot, lhs_literal.Convert(dot->shape().element_type()).value(),
        rhs_literal.Convert(dot->shape().element_type()).value());
  }

  absl::Status HandlePad(const HloInstruction* pad) override {
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
    // Try to convert the element type if the inferred type is not compatible.
    bool convert_element_type =
        pad->shape().element_type() != inferred_return_shape.element_type();
    if (convert_element_type) {
      inferred_return_shape.set_element_type(pad->shape().element_type());
    }
    CHECK(ShapeUtil::Compatible(pad->shape(), inferred_return_shape))
        << "return shape is set to: " << ShapeUtil::HumanString(pad->shape())
        << " but is inferred to be: "
        << ShapeUtil::HumanString(inferred_return_shape);
    ReturnT scalar;
    if (convert_element_type) {
      TF_ASSIGN_OR_RETURN(auto literal,
                          parent_->GetEvaluatedLiteralFor(pad->operand(1))
                              .Convert(inferred_return_shape.element_type()));
      scalar = literal.Get<ReturnT>({});
    } else {
      scalar =
          parent_->GetEvaluatedLiteralFor(pad->operand(1)).Get<ReturnT>({});
    }

    // Create new HLO of padded shape with padding value.
    Literal result(pad->shape());
    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&scalar](absl::Span<const int64_t> multi_index, int) {
          return scalar;
        }));

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

    ShapeUtil::ForEachIndexNoStatus(evaluated_operand.shape(), zero_base,
                                    evaluated_operand.shape().dimensions(),
                                    step, func);

    parent_->evaluated_[pad] = std::move(result);
    return absl::OkStatus();
  }

  absl::Status HandleClz(const HloInstruction* clz) override {
    // Enable CLZ only for integer types.
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[clz],
          ElementWiseUnaryOp(clz, [](ElementwiseT elem_operand) {
            int64_t unsigned_digits = std::numeric_limits<ReturnT>::digits +
                                      std::numeric_limits<ReturnT>::is_signed;
            return (unsigned_digits - 1) - Log2Floor<uint64_t>(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(clz);
  }

  absl::Status HandlePopulationCount(const HloInstruction* popcnt) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[popcnt],
          ElementWiseUnaryOp(popcnt, [](ElementwiseT elem_operand) {
            return std::bitset<CHAR_BIT * sizeof(ReturnT)>(elem_operand)
                .count();
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(popcnt);
  }

  absl::Status HandleSin(const HloInstruction* sin) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[sin],
          ElementWiseUnaryOp(sin, [](ElementwiseT elem_operand) {
            return std::sin(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(sin);
  }

  absl::Status HandleCos(const HloInstruction* cos) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[cos],
          ElementWiseUnaryOp(cos, [](ElementwiseT elem_operand) {
            return std::cos(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(cos);
  }

  absl::Status HandleTan(const HloInstruction* tan) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      TF_ASSIGN_OR_RETURN(
          parent_->evaluated_[tan],
          ElementWiseUnaryOp(tan, [](ElementwiseT elem_operand) {
            return std::tan(elem_operand);
          }));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(tan);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  absl::Status HandleReducePrecision(const HloInstruction* reduce_precision) {
    TF_ASSIGN_OR_RETURN(
        parent_->evaluated_[reduce_precision],
        ElementWiseUnaryOp(reduce_precision, [&](ElementwiseT elem) {
          const uint32_t src_mantissa_bits =
              std::numeric_limits<NativeT>::digits - 1;
          const uint32_t src_exponent_bits =
              8 * sizeof(NativeT) - src_mantissa_bits - 1;
          const uint32_t dest_mantissa_bits = reduce_precision->mantissa_bits();
          const uint32_t dest_exponent_bits = reduce_precision->exponent_bits();

          using Uint = UnsignedIntegerTypeForSizeType<sizeof(NativeT)>;
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
    return absl::OkStatus();
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_integral_v<NativeT> ||
                                      is_complex_v<NativeT>>* = nullptr>
  absl::Status HandleReducePrecision(const HloInstruction* reduce_precision) {
    return UnsupportedTypeError(reduce_precision);
  }

  absl::Status HandleReducePrecision(
      const HloInstruction* reduce_precision) override {
    return HandleReducePrecision<ElementwiseT>(reduce_precision);
  }

  absl::Status HandleIota(const HloInstruction* instruction) override {
    auto* iota = Cast<HloIotaInstruction>(instruction);
    if constexpr (std::is_integral_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT> ||
                  std::is_floating_point_v<ElementwiseT>) {
      Literal result(iota->shape());
      ShapeUtil::ForEachIndexNoStatus(
          iota->shape(), [&](absl::Span<const int64_t> idx) {
            result.Set(idx, static_cast<ReturnT>(idx[iota->iota_dimension()]));
            return true;
          });
      parent_->evaluated_[iota] = std::move(result);
      return absl::OkStatus();
    }
    return UnsupportedTypeError(iota);
  }

  absl::Status HandleRng(const HloInstruction* random) override {
    RandomDistribution distribution = random->random_distribution();
    const Shape& result_shape = random->shape();
    Literal result(result_shape);

    if constexpr (std::is_floating_point_v<ElementwiseT>) {
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
          const ReturnT low_val = low.Get<ReturnT>({});
          const ReturnT high_val = high.Get<ReturnT>({});
          std::uniform_real_distribution<ElementwiseT> generator(
              static_cast<ElementwiseT>(low_val),
              static_cast<ElementwiseT>(high_val));
          TF_RETURN_IF_ERROR(result.Populate<ReturnT>(
              [&](absl::Span<const int64_t> /*indexes*/) {
                while (true) {
                  const ReturnT v =
                      static_cast<ReturnT>(generator(parent_->engine_));
                  if (v >= low_val && v < high_val) {
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

          std::normal_distribution<ElementwiseT> generator(
              static_cast<ElementwiseT>(mean.Get<ReturnT>({})),
              static_cast<ElementwiseT>(stddev.Get<ReturnT>({})));

          TF_RETURN_IF_ERROR(result.Populate<ReturnT>(
              [&](absl::Span<const int64_t> /*indexes*/) {
                return static_cast<ReturnT>(generator(parent_->engine_));
              }));
          break;
        }
        default:
          return UnimplementedStrCat("The distribution ",
                                     RandomDistribution_Name(distribution),
                                     " is not implemented.");
      }
      parent_->evaluated_[random] = std::move(result);
      return absl::OkStatus();
    }
    if constexpr (std::is_integral_v<ElementwiseT>) {
      switch (distribution) {
        case RNG_UNIFORM: {
          const Literal& low =
              parent_->GetEvaluatedLiteralFor(random->operand(0));
          const Literal& high =
              parent_->GetEvaluatedLiteralFor(random->operand(1));

          // Note std::uniform_int_distribution assumes interval is closed,
          // i.e., [low, high], but we want [low, high) instead. Hence high-1 is
          // used as the upper range.
          std::uniform_int_distribution<int64_t> generator(
              static_cast<int64_t>(low.Get<ReturnT>({})),
              static_cast<int64_t>(high.Get<ReturnT>({})) - 1);

          TF_RETURN_IF_ERROR(result.Populate<ReturnT>(
              [&](absl::Span<const int64_t> /*indexes*/) {
                return static_cast<ReturnT>(generator(parent_->engine_));
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
      return absl::OkStatus();
    }
    return UnsupportedTypeError(random);
  }

 private:
  absl::StatusOr<Literal> ElementWiseUnaryOp(
      const HloInstruction* instruction,
      const std::function<ElementwiseT(ElementwiseT)>& unary_op) {
    const Literal& operand_literal =
        parent_->GetEvaluatedLiteralFor(instruction->operand(0));
    TF_ASSIGN_OR_RETURN(
        auto result_literal,
        (HloEvaluator::ElementWiseUnaryOpImpl<ReturnT, ReturnT>(
            instruction, ConvertUnaryFunction(unary_op), operand_literal)));

    return std::move(result_literal);
  }

  absl::StatusOr<Literal> ElementWiseBinaryOp(
      const HloInstruction* instruction,
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

    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&](absl::Span<const int64_t> multi_index, int) {
          return ConvertBinaryFunction(binary_op)(
              lhs_literal.Get<ReturnT>(multi_index),
              rhs_literal.Get<ReturnT>(multi_index));
        }));
    return std::move(result);
  }

  template <typename LhsType, typename RhsType, typename EhsType>
  absl::StatusOr<Literal> ElementwiseTernaryOp(
      const HloInstruction* instruction,
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

    TF_RETURN_IF_ERROR(result.PopulateParallel<ReturnT>(
        [&](absl::Span<const int64_t> multi_index, int) {
          return ternary_op(lhs_literal.Get<LhsType>(multi_index),
                            rhs_literal.Get<RhsType>(multi_index),
                            ehs_literal.Get<EhsType>(multi_index));
        }));

    return std::move(result);
  }

  template <typename NativeT>
  static bool IsShiftOutOfBounds(ElementwiseT rhs) {
    using UnsignedT = make_specialized_unsigned_t<NativeT>;
    UnsignedT lhs_bits_unsigned =
        static_cast<UnsignedT>(std::numeric_limits<UnsignedT>::digits);
    UnsignedT rhs_unsigned = static_cast<UnsignedT>(rhs);
    return rhs_unsigned >= lhs_bits_unsigned;
  }

  HloEvaluator* parent_;
};

// These extern templates prevent users of this class from implicitly
// instantiating it.  We explicitly instantiate this class in the various
// hlo_evaluator_typed_visitor*.cc files.
extern template class HloEvaluatorTypedVisitor<bool>;
extern template class HloEvaluatorTypedVisitor<u2, uint64_t>;
extern template class HloEvaluatorTypedVisitor<u4, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint8_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint16_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint32_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint64_t>;
extern template class HloEvaluatorTypedVisitor<s2, int64_t>;
extern template class HloEvaluatorTypedVisitor<s4, int64_t>;
extern template class HloEvaluatorTypedVisitor<int8_t, int64_t>;
extern template class HloEvaluatorTypedVisitor<int16_t, int64_t>;
extern template class HloEvaluatorTypedVisitor<int32_t, int64_t>;
extern template class HloEvaluatorTypedVisitor<int64_t>;
extern template class HloEvaluatorTypedVisitor<Eigen::half, float>;
extern template class HloEvaluatorTypedVisitor<float>;
extern template class HloEvaluatorTypedVisitor<double>;
extern template class HloEvaluatorTypedVisitor<complex64>;
extern template class HloEvaluatorTypedVisitor<complex128>;
extern template class HloEvaluatorTypedVisitor<bfloat16, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e5m2, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e4m3fn, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e4m3b11fnuz, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e5m2fnuz, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e4m3fnuz, float>;

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_TYPED_VISITOR_H_
