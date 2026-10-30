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
#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/casts.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

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
    return t;
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
  absl::Status UnsupportedTypeError(const HloInstruction* instruction) {
    return HloEvaluator::UnsupportedTypeError(instruction);
  }

  // Returns `shape`, if it has a layout, or a copy of `shape` with the default
  // layout if it doesn't. Some functions require shapes to have layouts, so we
  // simply always set one.
  Shape GetShapeWithLayout(const Shape& shape) {
    return HloEvaluator::GetShapeWithLayout(shape);
  }

 public:
  explicit HloEvaluatorTypedVisitor(HloEvaluator* p) : parent_(p) {}

  // Converts a UnaryOp from a unary function on ElementwiseT to a unary
  // function on ReturnT types.
  template <typename UnaryOp>
  auto ConvertUnaryFunction(const UnaryOp& unary_op) {
    return [&unary_op](ReturnT arg) {
      return static_cast<ReturnT>(unary_op(static_cast<ElementwiseT>(arg)));
    };
  }

  // Converts a BinaryOp from a binary function on ElementwiseT to a binary
  // function on ReturnT types.
  template <typename BinaryOp>
  auto ConvertBinaryFunction(const BinaryOp& binary_op) {
    return [&binary_op](ReturnT arg1, ReturnT arg2) {
      return static_cast<ReturnT>(binary_op(static_cast<ElementwiseT>(arg1),
                                            static_cast<ElementwiseT>(arg2)));
    };
  }

  // Converts a TernaryOp from a ternary function on ElementwiseT to a ternary
  // function on ReturnT types.
  template <typename TernaryOp>
  auto ConvertTernaryFunction(const TernaryOp& ternary_op) {
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
    return ElementWiseUnaryOp(
        abs, [](NativeT elem_operand) { return elem_operand; });
  }

  template <typename NativeT,
            typename std::enable_if_t<std::is_signed_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    return ElementWiseUnaryOp(
        abs, [](NativeT elem_operand) { return std::abs(elem_operand); });
  }

  template <typename NativeT,
            typename std::enable_if_t<is_complex_v<NativeT>>* = nullptr>
  absl::Status HandleAbs(const HloInstruction* abs) {
    return parent_
        ->ElementWiseUnaryOpImpl<typename NativeT::value_type, NativeT>(
            abs, [](NativeT elem_operand) { return std::abs(elem_operand); });
  }

  absl::Status HandleAbs(const HloInstruction* abs) override {
    // If the operand is of C64 type, the return type of abs will be F32.
    // However, ElementwiseT would still be the return type, F32, and thus
    // specifying the ElementwiseT explicitly as C64 is needed below.
    if constexpr (std::is_same_v<ReturnT, float>) {
      if (abs->operand(0)->shape().element_type() == C64) {
        return HandleAbs<complex64>(abs);
      }
    } else if constexpr (std::is_same_v<ReturnT, double>) {
      if (abs->operand(0)->shape().element_type() == C128) {
        return HandleAbs<complex128>(abs);
      }
    }
    return HandleAbs<ElementwiseT>(abs);
  }

  absl::Status HandleAcos(const HloInstruction* acos) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(acos, [](ElementwiseT elem_operand) {
        return std::acos(elem_operand);
      });
    }
    return UnsupportedTypeError(acos);
  }

  absl::Status HandleAcosh(const HloInstruction* acosh) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(acosh, [](ElementwiseT elem_operand) {
        return std::acosh(elem_operand);
      });
    }
    return UnsupportedTypeError(acosh);
  }

  absl::Status HandleAsin(const HloInstruction* asin) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(asin, [](ElementwiseT elem_operand) {
        return std::asin(elem_operand);
      });
    }
    return UnsupportedTypeError(asin);
  }

  absl::Status HandleRound(const HloInstruction* round) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseUnaryOp(round, [](ElementwiseT elem_operand) {
        return std::round(elem_operand);
      });
    }
    return UnsupportedTypeError(round);
  }

  absl::Status HandleRoundNearestEven(const HloInstruction* round) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      // Verify the current rounding direction.
      TF_RET_CHECK(fegetround() == FE_TONEAREST);
      return ElementWiseUnaryOp(round, [](ElementwiseT elem_operand) {
        return std::nearbyint(elem_operand);
      });
    }
    return UnsupportedTypeError(round);
  }

  absl::Status HandleCeil(const HloInstruction* ceil) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseUnaryOp(ceil, [](ElementwiseT elem_operand) {
        return std::ceil(elem_operand);
      });
    }
    return UnsupportedTypeError(ceil);
  }

  absl::Status HandleErf(const HloInstruction* erf) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseUnaryOp(erf, [](ElementwiseT elem_operand) {
        return std::erf(elem_operand);
      });
    }
    return UnsupportedTypeError(erf);
  }

  absl::Status HandleExp(const HloInstruction* exp) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(exp, [](ElementwiseT elem_operand) {
        return std::exp(elem_operand);
      });
    }
    return UnsupportedTypeError(exp);
  }

  absl::Status HandleExp2(const HloInstruction* exp2) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(exp2, [](ElementwiseT elem_operand) {
        if constexpr (is_complex_v<ReturnT>) {
          return std::exp(elem_operand * ElementwiseT(M_LN2));
        } else {
          return std::exp2(elem_operand);
        }
      });
    }
    return UnsupportedTypeError(exp2);
  }

  absl::Status HandleExpm1(const HloInstruction* expm1) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseUnaryOp(expm1, [](ElementwiseT elem_operand) {
        return std::expm1(elem_operand);
      });
    }
    return UnsupportedTypeError(expm1);
  }

  absl::Status HandleFloor(const HloInstruction* floor) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseUnaryOp(floor, [](ElementwiseT elem_operand) {
        return std::floor(elem_operand);
      });
    }
    return UnsupportedTypeError(floor);
  }

  absl::Status HandleLog(const HloInstruction* log) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(log, [](ElementwiseT elem_operand) {
        return std::log(elem_operand);
      });
    }
    return UnsupportedTypeError(log);
  }

  absl::Status HandleLog1p(const HloInstruction* log1p) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseUnaryOp(log1p, [](ElementwiseT elem_operand) {
        return std::log1p(elem_operand);
      });
    }
    return UnsupportedTypeError(log1p);
  }

  absl::Status HandleLog2(const HloInstruction* log2) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(log2, [](ElementwiseT elem_operand) {
        if constexpr (is_complex_v<ReturnT>) {
          return std::log(elem_operand) / ElementwiseT(M_LN2);
        } else {
          return std::log2(elem_operand);
        }
      });
    }
    return UnsupportedTypeError(log2);
  }

  absl::Status HandleNot(const HloInstruction* not_) override {
    if constexpr (std::is_arithmetic_v<ElementwiseT>) {
      return ElementWiseUnaryOp(not_, [](ElementwiseT elem_operand) {
        if constexpr (std::is_floating_point_v<ElementwiseT> ||
                      std::is_same_v<ElementwiseT, bool>) {
          return !elem_operand;
        } else {
          static_assert(std::is_integral_v<ElementwiseT>);
          return ~elem_operand;
        }
      });
    }
    return UnsupportedTypeError(not_);
  }

  template <
      typename NativeT,
      typename std::enable_if_t<std::is_signed_v<NativeT> &&
                                !std::is_floating_point_v<NativeT>>* = nullptr>
  absl::Status HandleNegate(const HloInstruction* negate) {
    using type = std::make_unsigned_t<NativeT>;
    return ElementWiseUnaryOp(negate, [](ElementwiseT elem_operand) {
      return NativeT(-type(elem_operand));
    });
  }

  template <typename NativeT, typename std::enable_if_t<
                                  !std::is_signed_v<NativeT> ||
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  absl::Status HandleNegate(const HloInstruction* negate) {
    return ElementWiseUnaryOp(
        negate, [](ElementwiseT elem_operand) { return -elem_operand; });
  }

  absl::Status HandleNegate(const HloInstruction* negate) override {
    return HandleNegate<ReturnT>(negate);
  }

  absl::Status HandleLogistic(const HloInstruction* logistic) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(logistic, [](ElementwiseT elem_operand) {
        return static_cast<ElementwiseT>(1) /
               (static_cast<ElementwiseT>(1) + std::exp(-elem_operand));
      });
    }
    return UnsupportedTypeError(logistic);
  }

  absl::Status HandleSign(const HloInstruction* sign) override {
    using NativeT = ElementwiseT;
    return ElementWiseUnaryOp(sign, [](ElementwiseT elem_operand) {
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
    });
  }

  absl::Status HandleAsinh(const HloInstruction* asinh) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(asinh, [](ElementwiseT elem_operand) {
        return std::asinh(elem_operand);
      });
    }
    return UnsupportedTypeError(asinh);
  }

  absl::Status HandleAtan2(const HloInstruction* atan2) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseBinaryOp(
          atan2, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
            return std::atan2(lhs_elem, rhs_elem);
          });
    }
    if constexpr (is_complex_v<ElementwiseT>) {
      return ElementWiseBinaryOp(atan2, [](ElementwiseT y, ElementwiseT x) {
        // atan2(y,x) = -i * log((x + i * y)/sqrt(x**2+y**2))
        auto i = ElementwiseT(0.0, 1.0);
        return (-i) * (std::log((x + i * y) / std::sqrt(x * x + y * y)));
      });
    }
    return UnsupportedTypeError(atan2);
  }

  absl::Status HandleAtanh(const HloInstruction* atanh) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(atanh, [](ElementwiseT elem_operand) {
        return std::atanh(elem_operand);
      });
    }
    return UnsupportedTypeError(atanh);
  }

  absl::Status HandleTanh(const HloInstruction* tanh) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(tanh, [](ElementwiseT elem_operand) {
        return std::tanh(elem_operand);
      });
    }
    return UnsupportedTypeError(tanh);
  }

  absl::Status HandleMultiply(const HloInstruction* multiply) override {
    return ElementWiseBinaryOp(
        multiply, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
          return ElementwiseT(ToArithmeticSafeType(lhs_elem) *
                              ToArithmeticSafeType(rhs_elem));
        });
  }

  absl::Status HandleMulhi(const HloInstruction* mulhi) override {
    if constexpr (std::is_integral_v<ReturnT> &&
                  !std::is_same_v<ReturnT, bool>) {
      return ElementWiseBinaryOp(
          mulhi, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
            constexpr int kBits = sizeof(ReturnT) * 8;
            using WideT = std::conditional_t<
                std::is_signed_v<ReturnT>,
                SignedIntegerTypeForSizeType<sizeof(ReturnT) * 2>,
                UnsignedIntegerTypeForSizeType<sizeof(ReturnT) * 2>>;
            return static_cast<ElementwiseT>(
                (static_cast<WideT>(lhs_elem) * static_cast<WideT>(rhs_elem)) >>
                kBits);
          });
    }
    return UnsupportedTypeError(mulhi);
  }

  absl::Status HandleSubtract(const HloInstruction* subtract) override {
    return ElementWiseBinaryOp(
        subtract, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
          return ElementwiseT(ToArithmeticSafeType(lhs_elem) -
                              ToArithmeticSafeType(rhs_elem));
        });
  }

  absl::Status HandleAdd(const HloInstruction* add) override {
    return ElementWiseBinaryOp(
        add, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
          return ElementwiseT(ToArithmeticSafeType(lhs_elem) +
                              ToArithmeticSafeType(rhs_elem));
        });
  }

  absl::Status HandleDivide(const HloInstruction* divide) override {
    return ElementWiseBinaryOp(
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
        });
  }

  absl::Status HandleMaximum(const HloInstruction* maximum) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      return ElementWiseBinaryOp(
          maximum, [](ElementwiseT lhs, ElementwiseT rhs) {
            if constexpr (std::numeric_limits<ElementwiseT>::has_quiet_NaN) {
              if (std::isnan(lhs)) {
                return lhs;
              }
              if (std::isnan(rhs)) {
                return rhs;
              }
            }
            return std::max(lhs, rhs);
          });
    }
    return UnsupportedTypeError(maximum);
  }

  absl::Status HandleMinimum(const HloInstruction* minimum) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      return ElementWiseBinaryOp(
          minimum, [](ElementwiseT lhs, ElementwiseT rhs) {
            if constexpr (std::numeric_limits<ElementwiseT>::has_quiet_NaN) {
              if (std::isnan(lhs)) {
                return lhs;
              }
              if (std::isnan(rhs)) {
                return rhs;
              }
            }
            return std::min(lhs, rhs);
          });
    }
    return UnsupportedTypeError(minimum);
  }

  absl::Status HandlePower(const HloInstruction* power) override {
    return ElementWiseBinaryOp(
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
                if (lhs_el == static_cast<ElementwiseT>(1)) {
                  return static_cast<ElementwiseT>(1);
                }
                if (lhs_el == static_cast<ElementwiseT>(-1)) {
                  return static_cast<ElementwiseT>(rhs_el % 2 == 0 ? 1 : -1);
                }
                return static_cast<ElementwiseT>(0);
              }
            }
            return static_cast<ElementwiseT>(
                IPow<std::make_unsigned_t<ElementwiseT>>(lhs_el, rhs_el));
          } else {
            return static_cast<ElementwiseT>(std::pow(lhs_el, rhs_el));
          }
        });
  }

  absl::Status HandleSqrt(const HloInstruction* sqrt) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(sqrt, [](ElementwiseT elem_operand) {
        return std::sqrt(elem_operand);
      });
    }
    return UnsupportedTypeError(sqrt);
  }

  absl::Status HandleCbrt(const HloInstruction* cbrt) override {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      return ElementWiseUnaryOp(cbrt, [](ElementwiseT elem_operand) {
        return std::cbrt(elem_operand);
      });
    }
    return UnsupportedTypeError(cbrt);
  }

  absl::Status HandleRsqrt(const HloInstruction* rsqrt) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(rsqrt, [](ElementwiseT elem_operand) {
        return static_cast<ElementwiseT>(1) / std::sqrt(elem_operand);
      });
    }
    return UnsupportedTypeError(rsqrt);
  }

  absl::Status HandleRemainder(const HloInstruction* remainder) override {
    if constexpr (!is_complex_v<ElementwiseT>) {
      return ElementWiseBinaryOp(
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
          });
    }
    return UnsupportedTypeError(remainder);
  }

  absl::Status HandleAnd(const HloInstruction* and_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      return ElementWiseBinaryOp(and_inst,
                                 [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                   return lhs_el & rhs_el;
                                 });
    }
    return UnsupportedTypeError(and_inst);
  }

  absl::Status HandleOr(const HloInstruction* or_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      return ElementWiseBinaryOp(or_inst,
                                 [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                   return lhs_el | rhs_el;
                                 });
    }
    return UnsupportedTypeError(or_inst);
  }

  absl::Status HandleXor(const HloInstruction* xor_inst) override {
    if constexpr (std::is_integral_v<ElementwiseT>) {
      return ElementWiseBinaryOp(xor_inst,
                                 [](ElementwiseT lhs_el, ElementwiseT rhs_el) {
                                   return lhs_el ^ rhs_el;
                                 });
    }
    return UnsupportedTypeError(xor_inst);
  }

  absl::Status HandleShiftLeft(const HloInstruction* shl) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      return ElementWiseBinaryOp(
          shl, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
            return IsShiftOutOfBounds<ElementwiseT>(rhs_elem)
                       ? 0
                       : (lhs_elem << rhs_elem);
          });
    }
    return UnsupportedTypeError(shl);
  }

  absl::Status HandleShiftRightArithmetic(const HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using SignedT = make_specialized_signed_t<ReturnT>;
      return ElementWiseBinaryOp(
          shr, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
            SignedT lhs_signed = static_cast<SignedT>(lhs_elem);
            if (IsShiftOutOfBounds<ReturnT>(rhs_elem)) {
              return lhs_signed < 0 ? static_cast<ElementwiseT>(-1) : 0;
            } else {
              return static_cast<ElementwiseT>(lhs_signed >> rhs_elem);
            }
          });
    }
    return UnsupportedTypeError(shr);
  }

  absl::Status HandleShiftRightLogical(const HloInstruction* shr) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      using UnsignedT = make_specialized_unsigned_t<ReturnT>;
      return ElementWiseBinaryOp(
          shr, [](ElementwiseT lhs_elem, ElementwiseT rhs_elem) {
            // If shift amount is greater than the number of
            // bits, then return 0.
            if (IsShiftOutOfBounds<ReturnT>(rhs_elem)) {
              return static_cast<ElementwiseT>(0);
            }
            return static_cast<ElementwiseT>(static_cast<UnsignedT>(lhs_elem) >>
                                             rhs_elem);
          });
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
      return ElementwiseTernaryOp(clamp, ConvertTernaryFunction(clamp_op));
    }
    return UnsupportedTypeError(clamp);
  }

  absl::Status HandleFma(const HloInstruction* fma) override {
    return EvaluateFma(fma);
  }

  absl::Status EvaluateFma(const HloInstruction* inst) {
    if constexpr (std::is_floating_point_v<ElementwiseT>) {
      auto fma_op = [](ReturnT a, ReturnT b, ReturnT c) -> ReturnT {
        if constexpr (std::is_same_v<ReturnT, float> ||
                      std::is_same_v<ReturnT, double>) {
          return std::fma(a, b, c);
        } else {
          // For narrow floats (f16, bf16, f8, f6, f4), the product a * b is
          // exact in double. Adding c in double with Round-To-Odd (via Knuth's
          // 2Sum error), then narrowing to float with Round-To-Odd before
          // converting to ReturnT (since Eigen::bfloat16/Eigen::half convert
          // from double via float), guarantees the exact single-rounded
          // IEEE-754 fusedMultiplyAdd result without double rounding.
          double p = static_cast<double>(a) * static_cast<double>(b);
          double dc = static_cast<double>(c);
          double s = p + dc;
          if (std::isfinite(s)) {
            double p_prime = s - dc;
            double err = (p - p_prime) + (dc - (s - p_prime));
            if (err != 0.0 && (absl::bit_cast<uint64_t>(s) & 1) == 0) {
              s = std::nextafter(
                  s, err > 0.0 ? std::numeric_limits<double>::infinity()
                               : -std::numeric_limits<double>::infinity());
            }
          }
          float f = static_cast<float>(s);
          if (std::isfinite(s) && std::isfinite(f)) {
            double fd = static_cast<double>(f);
            if (fd != s && (absl::bit_cast<uint32_t>(f) & 1) == 0) {
              f = std::nextafter(
                  f, s > fd ? std::numeric_limits<float>::infinity()
                            : -std::numeric_limits<float>::infinity());
            }
          }
          return static_cast<ReturnT>(f);
        }
      };
      return ElementwiseTernaryOp(inst, fma_op);
    }
    return UnsupportedTypeError(inst);
  }
  absl::Status HandleClz(const HloInstruction* clz) override {
    // Enable CLZ only for integer types.
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      return ElementWiseUnaryOp(clz, [](ElementwiseT elem_operand) {
        int64_t unsigned_digits = std::numeric_limits<ReturnT>::digits +
                                  std::numeric_limits<ReturnT>::is_signed;
        // Mask to ReturnT's width: ElementwiseT is 64 bits, so a negative
        // operand is sign extended and Log2Floor would count the sign bits.
        const uint64_t mask = unsigned_digits >= 64
                                  ? ~uint64_t{0}
                                  : (uint64_t{1} << unsigned_digits) - 1;
        const uint64_t bits = static_cast<uint64_t>(elem_operand) & mask;
        return (unsigned_digits - 1) - Log2Floor<uint64_t>(bits);
      });
    }
    return UnsupportedTypeError(clz);
  }

  absl::Status HandlePopulationCount(const HloInstruction* popcnt) override {
    if constexpr (std::is_integral_v<ElementwiseT> &&
                  !std::is_same_v<ElementwiseT, bool>) {
      return ElementWiseUnaryOp(popcnt, [](ElementwiseT elem_operand) {
        return std::bitset<CHAR_BIT * sizeof(ReturnT)>(elem_operand).count();
      });
    }
    return UnsupportedTypeError(popcnt);
  }

  absl::Status HandleSin(const HloInstruction* sin) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(sin, [](ElementwiseT elem_operand) {
        return std::sin(elem_operand);
      });
    }
    return UnsupportedTypeError(sin);
  }

  absl::Status HandleSinh(const HloInstruction* sinh) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(sinh, [](ElementwiseT elem_operand) {
        return std::sinh(elem_operand);
      });
    }
    return UnsupportedTypeError(sinh);
  }

  absl::Status HandleCos(const HloInstruction* cos) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(cos, [](ElementwiseT elem_operand) {
        return std::cos(elem_operand);
      });
    }
    return UnsupportedTypeError(cos);
  }

  absl::Status HandleCosh(const HloInstruction* cosh) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(cosh, [](ElementwiseT elem_operand) {
        return std::cosh(elem_operand);
      });
    }
    return UnsupportedTypeError(cosh);
  }

  absl::Status HandleTan(const HloInstruction* tan) override {
    if constexpr (std::is_floating_point_v<ElementwiseT> ||
                  is_complex_v<ElementwiseT>) {
      return ElementWiseUnaryOp(tan, [](ElementwiseT elem_operand) {
        return std::tan(elem_operand);
      });
    }
    return UnsupportedTypeError(tan);
  }

  template <typename NativeT, typename std::enable_if_t<
                                  std::is_floating_point_v<NativeT>>* = nullptr>
  absl::Status HandleReducePrecision(const HloInstruction* reduce_precision) {
    return ElementWiseUnaryOp(reduce_precision, [&](ElementwiseT elem) {
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
        const Uint x_rounding_bias = x_last_mantissa_bit + base_rounding_bias;

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
        const Uint reduced_exponent_bias = (1 << (dest_exponent_bits - 1)) - 1;
        const Uint reduced_max_exponent = exponent_bias + reduced_exponent_bias;
        const Uint reduced_min_exponent = exponent_bias - reduced_exponent_bias;

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
    });
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
      auto iota_shape = GetShapeWithLayout(iota->shape());
      Literal result(iota_shape);
      ShapeUtil::ForEachIndexNoStatus(
          iota_shape, [&](absl::Span<const int64_t> idx) {
            result.Set(idx, static_cast<ReturnT>(idx[iota->iota_dimension()]));
            return true;
          });
      parent_->SetEvaluatedLiteralFor(iota, std::move(result));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(iota);
  }

  absl::Status HandleRng(const HloInstruction* random) override {
    RandomDistribution distribution = random->random_distribution();
    Shape result_shape = GetShapeWithLayout(random->shape());
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
          ABSL_RETURN_IF_ERROR(result.Populate<ReturnT>(
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

          ABSL_RETURN_IF_ERROR(result.Populate<ReturnT>(
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
      parent_->SetEvaluatedLiteralFor(random, std::move(result));
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

          ABSL_RETURN_IF_ERROR(result.Populate<ReturnT>(
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
      parent_->SetEvaluatedLiteralFor(random, std::move(result));
      return absl::OkStatus();
    }
    return UnsupportedTypeError(random);
  }

 private:
  template <typename UnaryOp>
  absl::Status ElementWiseUnaryOp(const HloInstruction* instruction,
                                  UnaryOp&& unary_op) {
    static_assert(std::is_invocable_r_v<ElementwiseT, UnaryOp, ElementwiseT>,
                  "Invalid UnaryOp signature");
    return ElementWiseUnaryOpImpl(instruction, unary_op);
  }

  absl::Status ElementWiseUnaryOpImpl(
      const HloInstruction* instruction,
      absl::FunctionRef<ElementwiseT(ElementwiseT)> unary_op) {
    return parent_->ElementWiseUnaryOpImpl<ReturnT, ReturnT>(
        instruction, [unary_op](ReturnT arg) {
          return static_cast<ReturnT>(unary_op(static_cast<ElementwiseT>(arg)));
        });
  }

  template <typename BinaryOp>
  absl::Status ElementWiseBinaryOp(const HloInstruction* instruction,
                                   BinaryOp&& binary_op) {
    static_assert(std::is_invocable_r_v<ElementwiseT, BinaryOp, ElementwiseT,
                                        ElementwiseT>,
                  "Invalid BinaryOp signature");
    return ElementWiseBinaryOpImpl(instruction, binary_op);
  }

  absl::Status ElementWiseBinaryOpImpl(
      const HloInstruction* instruction,
      absl::FunctionRef<ElementwiseT(ElementwiseT, ElementwiseT)> binary_op) {
    return parent_->RunElementwiseBinaryOp(
        instruction,
        [&](Literal& result, const Literal& lhs_literal,
            const Literal& rhs_literal, bool same_layout) -> absl::Status {
          if (same_layout) {
            const ReturnT* lhs_data = lhs_literal.data<ReturnT>().data();
            const ReturnT* rhs_data = rhs_literal.data<ReturnT>().data();
            return result.PopulateLinearParallel<ReturnT>(
                [lhs_data, rhs_data, binary_op](int64_t linear_index, int) {
                  return static_cast<ReturnT>(binary_op(
                      static_cast<ElementwiseT>(lhs_data[linear_index]),
                      static_cast<ElementwiseT>(rhs_data[linear_index])));
                });
          }
          return result.PopulateParallel<ReturnT>(
              [&lhs_literal, &rhs_literal, binary_op](
                  absl::Span<const int64_t> multi_index, int) {
                return static_cast<ReturnT>(
                    binary_op(static_cast<ElementwiseT>(
                                  lhs_literal.Get<ReturnT>(multi_index)),
                              static_cast<ElementwiseT>(
                                  rhs_literal.Get<ReturnT>(multi_index))));
              });
        });
  }

  template <typename TernaryOp>
  absl::Status ElementwiseTernaryOp(const HloInstruction* instruction,
                                    TernaryOp&& ternary_op) {
    static_assert(
        std::is_invocable_r_v<ReturnT, TernaryOp, ReturnT, ReturnT, ReturnT>,
        "Invalid TernaryOp signature");
    return ElementwiseTernaryOpImpl(instruction, ternary_op);
  }

  absl::Status ElementwiseTernaryOpImpl(
      const HloInstruction* instruction,
      absl::FunctionRef<ReturnT(ReturnT, ReturnT, ReturnT)> ternary_op) {
    return parent_->RunElementwiseTernaryOp(
        instruction,
        [&](Literal& result, const Literal& lhs_literal,
            const Literal& rhs_literal, const Literal& ehs_literal,
            bool is_lhs_scalar, bool is_ehs_scalar,
            bool same_layout) -> absl::Status {
          if (same_layout) {
            const ReturnT* lhs_data = lhs_literal.data<ReturnT>().data();
            const ReturnT* rhs_data = rhs_literal.data<ReturnT>().data();
            const ReturnT* ehs_data = ehs_literal.data<ReturnT>().data();
            return result.PopulateLinearParallel<ReturnT>(
                [lhs_data, rhs_data, ehs_data, is_lhs_scalar, is_ehs_scalar,
                 ternary_op](int64_t linear_index, int) {
                  return ternary_op(lhs_data[is_lhs_scalar ? 0 : linear_index],
                                    rhs_data[linear_index],
                                    ehs_data[is_ehs_scalar ? 0 : linear_index]);
                });
          }
          return result.PopulateParallel<ReturnT>(
              [&lhs_literal, &rhs_literal, &ehs_literal, is_lhs_scalar,
               is_ehs_scalar,
               ternary_op](absl::Span<const int64_t> multi_index, int) {
                return ternary_op(
                    lhs_literal.Get<ReturnT>(is_lhs_scalar
                                                 ? absl::Span<const int64_t>{}
                                                 : multi_index),
                    rhs_literal.Get<ReturnT>(multi_index),
                    ehs_literal.Get<ReturnT>(is_ehs_scalar
                                                 ? absl::Span<const int64_t>{}
                                                 : multi_index));
              });
        });
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
extern template class HloEvaluatorTypedVisitor<u1, uint64_t>;
extern template class HloEvaluatorTypedVisitor<u2, uint64_t>;
extern template class HloEvaluatorTypedVisitor<u4, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint8_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint16_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint32_t, uint64_t>;
extern template class HloEvaluatorTypedVisitor<uint64_t>;
extern template class HloEvaluatorTypedVisitor<s1, int64_t>;
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
extern template class HloEvaluatorTypedVisitor<tsl::float4_e2m1fn, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e5m2, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e4m3, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e4m3fn, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e4m3b11fnuz, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e5m2fnuz, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e4m3fnuz, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e3m4, float>;
extern template class HloEvaluatorTypedVisitor<tsl::float8_e8m0fnu, float>;

}  // namespace xla

#endif  // XLA_HLO_EVALUATOR_HLO_EVALUATOR_TYPED_VISITOR_H_
