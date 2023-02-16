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

#include "tools/mlir_interpreter/dialects/cwise_math.h"
#include "tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

REGISTER_MLIR_INTERPRETER_OP("mhlo.add", applyCwiseBinaryMap<Plus>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.and", applyCwiseBinaryMap<BitAnd>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.atan2", applyCwiseBinaryMap<ATan2>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.complex", applyCwiseBinaryMap<Complex>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.divide", applyCwiseBinaryMap<Divide>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.maximum", applyCwiseBinaryMap<Max>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.minimum", applyCwiseBinaryMap<Min>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.multiply", applyCwiseBinaryMap<Multiply>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.or", applyCwiseBinaryMap<BitOr>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.power", applyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.remainder", applyCwiseBinaryMap<Remainder>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.shift_left", applyCwiseBinaryMap<ShiftLeft>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.shift_right_arithmetic",
                             applyCwiseBinaryMap<ShiftRightArith>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.shift_right_logical",
                             applyCwiseBinaryMap<ShiftRightLogical>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.subtract", applyCwiseBinaryMap<Minus>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.xor", applyCwiseBinaryMap<BitXor>);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
