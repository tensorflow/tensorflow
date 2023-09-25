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

#include "mlir/Dialect/Complex/IR/Complex.h"

#include "tools/mlir_interpreter/dialects/cwise_math.h"
#include "tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue constant(InterpreterState&, complex::ConstantOp constant) {
  auto ty = constant->getResultTypes()[0];
  return dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    if constexpr (is_complex_v<decltype(dummy)>) {
      using T = typename decltype(dummy)::value_type;
      auto values =
          llvm::to_vector(constant.getValue().getAsValueRange<FloatAttr>());
      return {decltype(dummy){static_cast<T>(values[0].convertToDouble()),
                              static_cast<T>(values[1].convertToDouble())}};
    } else {
      llvm_unreachable("invalid constant");
    }
  });
}

REGISTER_MLIR_INTERPRETER_OP("complex.abs", "math.absf");
REGISTER_MLIR_INTERPRETER_OP("complex.add", "arith.addf");
REGISTER_MLIR_INTERPRETER_OP("complex.cos", applyCwiseMap<Cos>);
REGISTER_MLIR_INTERPRETER_OP("complex.create", applyCwiseBinaryMap<Complex>);
REGISTER_MLIR_INTERPRETER_OP("complex.div", applyCwiseBinaryMap<Divide>);
REGISTER_MLIR_INTERPRETER_OP("complex.exp", applyCwiseMap<Exp>);
REGISTER_MLIR_INTERPRETER_OP("complex.expm1", applyCwiseMap<ExpM1>);
REGISTER_MLIR_INTERPRETER_OP("complex.im", applyCwiseMap<Imag>);
REGISTER_MLIR_INTERPRETER_OP("complex.log", applyCwiseMap<Log>);
REGISTER_MLIR_INTERPRETER_OP("complex.log1p", applyCwiseMap<Log1P>);
REGISTER_MLIR_INTERPRETER_OP("complex.mul", applyCwiseBinaryMap<Multiply>);
REGISTER_MLIR_INTERPRETER_OP("complex.neg", applyCwiseMap<Neg>);
REGISTER_MLIR_INTERPRETER_OP("complex.pow", applyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("complex.re", applyCwiseMap<Real>);
REGISTER_MLIR_INTERPRETER_OP("complex.rsqrt", applyCwiseMap<RSqrt>);
REGISTER_MLIR_INTERPRETER_OP("complex.sin", applyCwiseMap<Sin>);
REGISTER_MLIR_INTERPRETER_OP("complex.sqrt", applyCwiseMap<Sqrt>);
REGISTER_MLIR_INTERPRETER_OP("complex.tanh", applyCwiseMap<TanH>);
REGISTER_MLIR_INTERPRETER_OP(constant);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
