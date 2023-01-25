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
using Minus = detail::Wrap<std::minus, CwiseArith>;
using BitAnd = detail::Wrap<std::bit_and, CwiseIntegral>;
using BitOr = detail::Wrap<std::bit_or, CwiseIntegral>;
using BitXor = detail::Wrap<std::bit_xor, CwiseIntegral>;

}  // namespace interpreter
}  // namespace mlir

#endif  // MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_CWISE_MATH_H_
