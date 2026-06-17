/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/mlir/tools/mlir_interpreter/dialects/cwise_math.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

struct CopySign : CwiseFloat {
  template <typename T>
  static T Apply(T a, T b) {
    return std::copysign(a, b);
  }
};

REGISTER_MLIR_INTERPRETER_OP("math.absf", ApplyCwiseMap<Abs>);
REGISTER_MLIR_INTERPRETER_OP("math.absi", ApplyCwiseMap<Abs>);
REGISTER_MLIR_INTERPRETER_OP("math.atan", ApplyCwiseMap<ATan>);
REGISTER_MLIR_INTERPRETER_OP("math.atan2", ApplyCwiseBinaryMap<ATan2>);
REGISTER_MLIR_INTERPRETER_OP("math.cbrt", ApplyCwiseMap<Cbrt>);
REGISTER_MLIR_INTERPRETER_OP("math.ceil", ApplyCwiseMap<Ceil>);
REGISTER_MLIR_INTERPRETER_OP("math.copysign", ApplyCwiseBinaryMap<CopySign>);
REGISTER_MLIR_INTERPRETER_OP("math.cos", ApplyCwiseMap<Cos>);
REGISTER_MLIR_INTERPRETER_OP("math.ctlz", ApplyCwiseMap<Clz>);
REGISTER_MLIR_INTERPRETER_OP("math.ctpop", ApplyCwiseMap<PopCount>);
REGISTER_MLIR_INTERPRETER_OP("math.cttz", ApplyCwiseMap<Ctz>);
REGISTER_MLIR_INTERPRETER_OP("math.erf", ApplyCwiseMap<Erf>);
REGISTER_MLIR_INTERPRETER_OP("math.exp", ApplyCwiseMap<Exp>);
REGISTER_MLIR_INTERPRETER_OP("math.exp2", ApplyCwiseMap<Exp2>);
REGISTER_MLIR_INTERPRETER_OP("math.expm1", ApplyCwiseMap<ExpM1>);
REGISTER_MLIR_INTERPRETER_OP("math.floor", ApplyCwiseMap<Floor>);
REGISTER_MLIR_INTERPRETER_OP("math.ipowi", ApplyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("math.log", ApplyCwiseMap<Log>);
REGISTER_MLIR_INTERPRETER_OP("math.log10", ApplyCwiseMap<Log10>);
REGISTER_MLIR_INTERPRETER_OP("math.log1p", ApplyCwiseMap<Log1P>);
REGISTER_MLIR_INTERPRETER_OP("math.log2", ApplyCwiseMap<Log2>);
REGISTER_MLIR_INTERPRETER_OP("math.powf", ApplyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("math.round", ApplyCwiseMap<Round>);
REGISTER_MLIR_INTERPRETER_OP("math.roundeven", ApplyCwiseMap<NearbyInt>);
REGISTER_MLIR_INTERPRETER_OP("math.rsqrt", ApplyCwiseMap<RSqrt>);
REGISTER_MLIR_INTERPRETER_OP("math.sin", ApplyCwiseMap<Sin>);
REGISTER_MLIR_INTERPRETER_OP("math.sqrt", ApplyCwiseMap<Sqrt>);
REGISTER_MLIR_INTERPRETER_OP("math.tan", ApplyCwiseMap<Tan>);
REGISTER_MLIR_INTERPRETER_OP("math.tanh", ApplyCwiseMap<TanH>);
REGISTER_MLIR_INTERPRETER_OP("math.trunc", ApplyCwiseMap<Trunc>);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
