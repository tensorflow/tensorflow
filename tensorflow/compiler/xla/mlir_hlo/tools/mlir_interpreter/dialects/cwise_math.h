/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_CWISE_MATH_H_
#define MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_CWISE_MATH_H_

#include <complex>
#include <type_traits>

#include "tools/mlir_interpreter/framework/interpreter_value_util.h"

namespace mlir {
namespace interpreter {

struct ATan2 : CwiseReal {
  template <typename T>
  static T apply(T a, T b) {
    return std::atan2(a, b);
  }
};

struct Clz : CwiseInt {
  template <typename T>
  static T apply(T a) {
    if (!a) {
      // Return something well-defined for zeroes.
      return sizeof(T{}) * CHAR_BIT;
    }
    return __builtin_clzl(
               static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(a))) -
           (sizeof(uint64_t) - sizeof(T{})) * CHAR_BIT;
  }
};

struct Ctz : CwiseInt {
  template <typename T>
  static T apply(T a) {
    if (!a) {
      // Return something well-defined for zeroes.
      return sizeof(T{}) * CHAR_BIT;
    }
    return __builtin_ctzl(static_cast<uint64_t>(a));
  }
};

struct Complex : CwiseFloat {
  template <typename T>
  static std::complex<T> apply(T a, T b) {
    return {a, b};
  }
};

struct Max : CwiseReal {
  template <typename T>
  static T apply(T a, T b) {
    return std::max(a, b);
  }
};

struct Min : CwiseReal {
  template <typename T>
  static T apply(T a, T b) {
    return std::min(a, b);
  }
};

struct Power : CwiseArith {
  template <typename T>
  static T apply(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) {
        if (b < 0) {
          return a == 1 ? 1 : 0;
        }
      }
      T result = 1;
      while (b > 0) {
        if (b & 1) result *= a;
        b >>= 1;
        if (b) {
          a *= a;
        }
      }
      return result;
    } else {
      return std::pow(a, b);
    }
  }
};

struct Remainder : CwiseReal {
  template <typename T>
  static T apply(T a, T b) {
    if constexpr (std::is_integral_v<T>) {
      return a % b;
    } else {
      return std::fmod(a, b);
    }
  }
};

struct ShiftRightArith : CwiseInt {
  template <typename T>
  static T apply(T a, T b) {
    return b >= sizeof(T) * CHAR_BIT ? 0 : (a >> b);
  }
};

struct ShiftRightLogical : CwiseInt {
  template <typename T>
  static T apply(T a, T b) {
    return b >= sizeof(T) * CHAR_BIT
               ? 0
               : static_cast<std::make_unsigned_t<T>>(a) >> b;
  }
};

struct ShiftLeft : CwiseInt {
  template <typename T>
  static T apply(T a, T b) {
    return b >= sizeof(T) * CHAR_BIT ? 0 : (a << b);
  }
};

namespace detail {
template <template <typename T> class F, typename trait>
struct Wrap : trait {
  template <typename T>
  static T apply(T a, T b) {
    return F<T>{}(a, b);
  }
};
}  // namespace detail

using Plus = detail::Wrap<std::plus, CwiseArith>;
using Divide = detail::Wrap<std::divides, CwiseArith>;
using Multiply = detail::Wrap<std::multiplies, CwiseArith>;
using Minus = detail::Wrap<std::minus, CwiseAll>;
using BitAnd = detail::Wrap<std::bit_and, CwiseIntegral>;
using BitOr = detail::Wrap<std::bit_or, CwiseIntegral>;
using BitXor = detail::Wrap<std::bit_xor, CwiseIntegral>;

struct RSqrt : CwiseNonIntegral {
  template <typename T>
  static T apply(T a) {
    return static_cast<T>(T{1} / std::sqrt(a));
  }
};

#define DEFINE_WRAPPER(name, std_fun, trait) \
  struct name : trait {                      \
    template <typename T>                    \
    static auto apply(T a) {                 \
      return std_fun(a);                     \
    }                                        \
  };

DEFINE_WRAPPER(ATan, std::atan, CwiseNonIntegral);
DEFINE_WRAPPER(Abs, std::abs, CwiseSignedOrComplex);
DEFINE_WRAPPER(Cbrt, std::cbrt, CwiseFloat);
DEFINE_WRAPPER(Ceil, std::ceil, CwiseFloat);
DEFINE_WRAPPER(Cos, std::cos, CwiseNonIntegral);
DEFINE_WRAPPER(Erf, std::erf, CwiseFloat);
DEFINE_WRAPPER(Exp, std::exp, CwiseNonIntegral);
DEFINE_WRAPPER(Exp2, std::exp2, CwiseFloat);
DEFINE_WRAPPER(Floor, std::floor, CwiseFloat);
DEFINE_WRAPPER(Imag, std::imag, CwiseComplex);
DEFINE_WRAPPER(IsFinite, std::isfinite, CwiseFloat);
DEFINE_WRAPPER(Log, std::log, CwiseNonIntegral);
DEFINE_WRAPPER(Log10, std::log10, CwiseNonIntegral);
DEFINE_WRAPPER(Log2, std::log2, CwiseFloat);
DEFINE_WRAPPER(NearbyInt, std::nearbyint, CwiseFloat);
DEFINE_WRAPPER(Neg, std::negate<T>{}, CwiseSignedOrComplex);
DEFINE_WRAPPER(Real, std::real, CwiseComplex);
DEFINE_WRAPPER(Round, std::round, CwiseFloat);
DEFINE_WRAPPER(Sin, std::sin, CwiseNonIntegral);
DEFINE_WRAPPER(Sqrt, std::sqrt, CwiseNonIntegral);
DEFINE_WRAPPER(Tan, std::tan, CwiseNonIntegral);
DEFINE_WRAPPER(TanH, std::tanh, CwiseNonIntegral);
DEFINE_WRAPPER(Trunc, std::trunc, CwiseFloat);

#undef DEFINE_WRAPPER

struct ExpM1 : CwiseNonIntegral {
  template <typename T>
  static T apply(T a) {
    if constexpr (std::is_floating_point_v<T>) {
      return std::expm1(a);
    } else {
      auto r = std::real(a);
      auto i = std::imag(a);
      auto s = std::sin(i / 2);
      auto real = std::expm1(r) * std::cos(i) - 2 * s * s;
      auto imag = std::exp(r) * std::sin(i);
      return {real, imag};
    }
  }
};

struct Log1P : CwiseNonIntegral {
  template <typename T>
  static T apply(T a) {
    if constexpr (std::is_floating_point_v<T>) {
      return std::log1p(a);
    } else {
      auto r = std::real(a);
      auto i = std::imag(a);
      auto l = std::hypot(r + 1, i);
      auto real = std::log(l);
      auto imag = std::atan2(i, r + 1);
      return {real, imag};
    }
  }
};

struct PopCount : CwiseInt {
  template <typename T>
  static T apply(T a) {
    return __builtin_popcountl(
        static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(a)));
  }
};

}  // namespace interpreter
}  // namespace mlir

#endif  // MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_CWISE_MATH_H_
