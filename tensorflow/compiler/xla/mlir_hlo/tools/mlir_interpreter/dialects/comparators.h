/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_COMPARATORS_H_
#define MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_COMPARATORS_H_

#include <complex>
#include <type_traits>

#include "llvm/Support/ErrorHandling.h"
#include "tools/mlir_interpreter/framework/interpreter_value_util.h"

namespace mlir {
namespace interpreter {

// Despite the name, this works on integers and complex too.
template <int64_t v, bool r, bool nan_result>
struct FloatCompare : CwiseArith {
  template <typename T>
  static bool apply(T a, T b) {
    if (isnan(a) || isnan(b)) return nan_result;
    if constexpr (v == 0) {
      // For complex eq/ne.
      return (a == b) == r;
    } else if constexpr (std::is_floating_point_v<T> || std::is_integral_v<T>) {
      auto cmp = a > b ? 1 : (a < b ? -1 : 0);
      return (cmp == v) == r;
    } else {
      llvm_unreachable("operation not supported for this type");
    }
  }

  template <typename T>
  static bool isnan(T a) {
    return std::isnan(a);
  }
  template <typename T>
  static bool isnan(std::complex<T> a) {
    return std::isnan(std::real(a)) || std::isnan(std::imag(a));
  }
};

using Foeq = FloatCompare<0, true, false>;
using Foge = FloatCompare<-1, false, false>;
using Fogt = FloatCompare<1, true, false>;
using Fole = FloatCompare<1, false, false>;
using Folt = FloatCompare<-1, true, false>;
using Fone = FloatCompare<0, false, false>;
using Ford = FloatCompare<99, false, false>;
using Fueq = FloatCompare<0, true, true>;
using Fuge = FloatCompare<-1, false, true>;
using Fugt = FloatCompare<1, true, true>;
using Fule = FloatCompare<1, false, true>;
using Fult = FloatCompare<-1, true, true>;
using Fune = FloatCompare<0, false, true>;
using Funo = FloatCompare<99, true, true>;

template <int64_t v, bool r>
struct UnsignedCompare : CwiseInt {
  template <typename T>
  static bool apply(T a, T b) {
    using U = std::make_unsigned_t<T>;
    auto aU = static_cast<U>(a);
    auto bU = static_cast<U>(b);
    auto cmp = aU > bU ? 1 : (aU < bU ? -1 : 0);
    return (cmp == v) == r;
  }
};

using Iuge = UnsignedCompare<-1, false>;
using Iule = UnsignedCompare<1, false>;
using Iugt = UnsignedCompare<1, true>;
using Iult = UnsignedCompare<-1, true>;

struct Iumax {
  template <typename T>
  static T apply(T a, T b) {
    return Iuge::apply(a, b) ? a : b;
  }
};

struct Iumin {
  template <typename T>
  static T apply(T a, T b) {
    return Iule::apply(a, b) ? a : b;
  }
};

}  // namespace interpreter
}  // namespace mlir

#endif  // MLIR_HLO_TOOLS_MLIR_INTERPRETER_DIALECTS_COMPARATORS_H_
