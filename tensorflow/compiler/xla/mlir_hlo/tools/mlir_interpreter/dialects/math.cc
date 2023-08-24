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

#include "tools/mlir_interpreter/dialects/cwise_math.h"
#include "tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

struct CopySign : CwiseFloat {
  template <typename T>
  static T apply(T a, T b) {
    return std::copysign(a, b);
  }
};

REGISTER_MLIR_INTERPRETER_OP("math.absf", applyCwiseMap<Abs>);
REGISTER_MLIR_INTERPRETER_OP("math.absi", applyCwiseMap<Abs>);
REGISTER_MLIR_INTERPRETER_OP("math.atan", applyCwiseMap<ATan>);
REGISTER_MLIR_INTERPRETER_OP("math.atan2", applyCwiseBinaryMap<ATan2>);
REGISTER_MLIR_INTERPRETER_OP("math.cbrt", applyCwiseMap<Cbrt>);
REGISTER_MLIR_INTERPRETER_OP("math.ceil", applyCwiseMap<Ceil>);
REGISTER_MLIR_INTERPRETER_OP("math.copysign", applyCwiseBinaryMap<CopySign>);
REGISTER_MLIR_INTERPRETER_OP("math.cos", applyCwiseMap<Cos>);
REGISTER_MLIR_INTERPRETER_OP("math.ctlz", applyCwiseMap<Clz>);
REGISTER_MLIR_INTERPRETER_OP("math.ctpop", applyCwiseMap<PopCount>);
REGISTER_MLIR_INTERPRETER_OP("math.cttz", applyCwiseMap<Ctz>);
REGISTER_MLIR_INTERPRETER_OP("math.erf", applyCwiseMap<Erf>);
REGISTER_MLIR_INTERPRETER_OP("math.exp", applyCwiseMap<Exp>);
REGISTER_MLIR_INTERPRETER_OP("math.exp2", applyCwiseMap<Exp2>);
REGISTER_MLIR_INTERPRETER_OP("math.expm1", applyCwiseMap<ExpM1>);
REGISTER_MLIR_INTERPRETER_OP("math.floor", applyCwiseMap<Floor>);
REGISTER_MLIR_INTERPRETER_OP("math.ipowi", applyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("math.log", applyCwiseMap<Log>);
REGISTER_MLIR_INTERPRETER_OP("math.log10", applyCwiseMap<Log10>);
REGISTER_MLIR_INTERPRETER_OP("math.log1p", applyCwiseMap<Log1P>);
REGISTER_MLIR_INTERPRETER_OP("math.log2", applyCwiseMap<Log2>);
REGISTER_MLIR_INTERPRETER_OP("math.powf", applyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("math.round", applyCwiseMap<Round>);
REGISTER_MLIR_INTERPRETER_OP("math.roundeven", applyCwiseMap<NearbyInt>);
REGISTER_MLIR_INTERPRETER_OP("math.rsqrt", applyCwiseMap<RSqrt>);
REGISTER_MLIR_INTERPRETER_OP("math.sin", applyCwiseMap<Sin>);
REGISTER_MLIR_INTERPRETER_OP("math.sqrt", applyCwiseMap<Sqrt>);
REGISTER_MLIR_INTERPRETER_OP("math.tan", applyCwiseMap<Tan>);
REGISTER_MLIR_INTERPRETER_OP("math.tanh", applyCwiseMap<TanH>);
REGISTER_MLIR_INTERPRETER_OP("math.trunc", applyCwiseMap<Trunc>);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
