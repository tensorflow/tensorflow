/* Copyright 2023 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/mlir/tools/mlir_interpreter/dialects/cwise_math.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

struct Logistic : CwiseNonIntegral {
  template <typename T>
  static T Apply(T a) {
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
  static bool Apply(bool a) { return !a; }

  template <typename T>
  static T Apply(T a) {
    return ~a;
  }
};

struct Sign : CwiseSigned {
  template <typename T>
  static T Apply(T a) {
    return std::copysign(T{1}, a);
  }
};

REGISTER_MLIR_INTERPRETER_OP("mhlo.abs", ApplyCwiseMap<Abs>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.cbrt", ApplyCwiseMap<Cbrt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.ceil", ApplyCwiseMap<Ceil>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.cosine", ApplyCwiseMap<Cos>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.count_leading_zeros", ApplyCwiseMap<Clz>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.exponential", ApplyCwiseMap<Exp>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.exponential_minus_one",
                             ApplyCwiseMap<ExpM1>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.floor", ApplyCwiseMap<Floor>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.imag", ApplyCwiseMap<Imag>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.is_finite", ApplyCwiseMap<IsFinite>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.log", ApplyCwiseMap<Log>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.log_plus_one", ApplyCwiseMap<Log1P>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.logistic", ApplyCwiseMap<Logistic>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.negate", ApplyCwiseMap<Neg>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.not", ApplyCwiseMap<Not>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.popcnt", ApplyCwiseMap<PopCount>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.real", ApplyCwiseMap<Real>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.round_nearest_afz", ApplyCwiseMap<Round>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.round_nearest_even",
                             ApplyCwiseMap<NearbyInt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.rsqrt", ApplyCwiseMap<RSqrt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.sign", ApplyCwiseMap<Sign>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.sine", ApplyCwiseMap<Sin>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.sqrt", ApplyCwiseMap<Sqrt>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.tanh", ApplyCwiseMap<TanH>);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
