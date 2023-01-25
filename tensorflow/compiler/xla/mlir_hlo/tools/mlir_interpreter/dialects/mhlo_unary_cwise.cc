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

#include <complex>

#include "tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

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

struct Logistic : CwiseNonIntegral {
  template <typename T>
  static T apply(T a) {
    if (std::real(a) < 0) {
      T e = std::exp(a);
      return e / (e + T{1});
    }
    return T{1} / (std::exp(-a) + T{1});
  }
};

// Note: this can't be replaced with std::bit_not, which always returns true for
// bools.
struct Not : CwiseIntegral {
  static bool apply(bool a) { return !a; }

  template <typename T>
  static T apply(T a) {
    return ~a;
  }
};

struct PopCount : CwiseInt {
  template <typename T>
  static T apply(T a) {
    return __builtin_popcountl(
        static_cast<uint64_t>(static_cast<std::make_unsigned_t<T>>(a)));
  }
};

struct RSqrt : CwiseNonIntegral {
  template <typename T>
  static T apply(T a) {
    return static_cast<T>(T{1} / std::sqrt(a));
  }
};

struct Sign : CwiseSigned {
  template <typename T>
  static T apply(T a) {
    return std::copysign(T{1}, a);
  }
};

#define DEFINE_WRAPPER(name, std_fun, trait) \
  struct name : trait {                      \
    template <typename T>                    \
    static auto apply(T a) {                 \
      return std_fun(a);                     \
    }                                        \
  };

DEFINE_WRAPPER(Abs, std::abs, CwiseSignedOrComplex);
DEFINE_WRAPPER(Cbrt, std::cbrt, CwiseFloat);
DEFINE_WRAPPER(Ceil, std::ceil, CwiseFloat);
DEFINE_WRAPPER(Cos, std::cos, CwiseNonIntegral);
DEFINE_WRAPPER(Exp, std::exp, CwiseNonIntegral);
DEFINE_WRAPPER(Floor, std::floor, CwiseFloat);
DEFINE_WRAPPER(Imag, std::imag, CwiseComplex);
DEFINE_WRAPPER(IsFinite, std::isfinite, CwiseFloat);
DEFINE_WRAPPER(Log, std::log, CwiseNonIntegral);
DEFINE_WRAPPER(NearbyInt, std::nearbyint, CwiseFloat);
DEFINE_WRAPPER(Neg, std::negate<T>{}, CwiseSignedOrComplex);
DEFINE_WRAPPER(Real, std::real, CwiseComplex);
DEFINE_WRAPPER(Round, std::round, CwiseFloat);
DEFINE_WRAPPER(Sin, std::sin, CwiseNonIntegral);
DEFINE_WRAPPER(Sqrt, std::sqrt, CwiseNonIntegral);
DEFINE_WRAPPER(TanH, std::tanh, CwiseNonIntegral);

REGISTER_MLIR_INTERPRETER_OP("mhlo.abs", applyCwiseMap<Abs>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.cbrt", applyCwiseMap<Cbrt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.ceil", applyCwiseMap<Ceil>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.cosine", applyCwiseMap<Cos>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.count_leading_zeros", applyCwiseMap<Clz>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.exponential", applyCwiseMap<Exp>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.exponential_minus_one",
                             applyCwiseMap<ExpM1>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.floor", applyCwiseMap<Floor>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.imag", applyCwiseMap<Imag>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.is_finite", applyCwiseMap<IsFinite>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.log", applyCwiseMap<Log>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.log_plus_one", applyCwiseMap<Log1P>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.logistic", applyCwiseMap<Logistic>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.negate", applyCwiseMap<Neg>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.not", applyCwiseMap<Not>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.popcnt", applyCwiseMap<PopCount>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.real", applyCwiseMap<Real>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.round_nearest_afz", applyCwiseMap<Round>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.round_nearest_even",
                             applyCwiseMap<NearbyInt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.rsqrt", applyCwiseMap<RSqrt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.sign", applyCwiseMap<Sign>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.sine", applyCwiseMap<Sin>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.sqrt", applyCwiseMap<Sqrt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.tanh", applyCwiseMap<TanH>);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
