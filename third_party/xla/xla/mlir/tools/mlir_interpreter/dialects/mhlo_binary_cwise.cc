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

#include "xla/mlir/tools/mlir_interpreter/dialects/cwise_math.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

REGISTER_MLIR_INTERPRETER_OP("mhlo.add", ApplyCwiseBinaryMap<Plus>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.and", ApplyCwiseBinaryMap<BitAnd>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.atan2", ApplyCwiseBinaryMap<ATan2>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.complex", ApplyCwiseBinaryMap<Complex>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.divide", ApplyCwiseBinaryMap<Divide>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.maximum", ApplyCwiseBinaryMap<Max>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.minimum", ApplyCwiseBinaryMap<Min>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.multiply", ApplyCwiseBinaryMap<Multiply>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.or", ApplyCwiseBinaryMap<BitOr>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.power", ApplyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.remainder", ApplyCwiseBinaryMap<Remainder>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.shift_left", ApplyCwiseBinaryMap<ShiftLeft>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.shift_right_arithmetic",
                             ApplyCwiseBinaryMap<ShiftRightArith>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.shift_right_logical",
                             ApplyCwiseBinaryMap<ShiftRightLogical>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.subtract", ApplyCwiseBinaryMap<Minus>);
REGISTER_MLIR_INTERPRETER_OP("mhlo.xor", ApplyCwiseBinaryMap<BitXor>);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
