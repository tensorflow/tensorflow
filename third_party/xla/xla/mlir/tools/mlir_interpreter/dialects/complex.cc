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

#include "mlir/Dialect/Complex/IR/Complex.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/cwise_math.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue Constant(InterpreterState&, complex::ConstantOp constant) {
  auto ty = constant->getResultTypes()[0];
  return DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
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
REGISTER_MLIR_INTERPRETER_OP("complex.cos", ApplyCwiseMap<Cos>);
REGISTER_MLIR_INTERPRETER_OP("complex.create", ApplyCwiseBinaryMap<Complex>);
REGISTER_MLIR_INTERPRETER_OP("complex.div", ApplyCwiseBinaryMap<Divide>);
REGISTER_MLIR_INTERPRETER_OP("complex.exp", ApplyCwiseMap<Exp>);
REGISTER_MLIR_INTERPRETER_OP("complex.expm1", ApplyCwiseMap<ExpM1>);
REGISTER_MLIR_INTERPRETER_OP("complex.im", ApplyCwiseMap<Imag>);
REGISTER_MLIR_INTERPRETER_OP("complex.log", ApplyCwiseMap<Log>);
REGISTER_MLIR_INTERPRETER_OP("complex.log1p", ApplyCwiseMap<Log1P>);
REGISTER_MLIR_INTERPRETER_OP("complex.mul", ApplyCwiseBinaryMap<Multiply>);
REGISTER_MLIR_INTERPRETER_OP("complex.neg", ApplyCwiseMap<Neg>);
REGISTER_MLIR_INTERPRETER_OP("complex.pow", ApplyCwiseBinaryMap<Power>);
REGISTER_MLIR_INTERPRETER_OP("complex.re", ApplyCwiseMap<Real>);
REGISTER_MLIR_INTERPRETER_OP("complex.rsqrt", ApplyCwiseMap<RSqrt>);
REGISTER_MLIR_INTERPRETER_OP("complex.sin", ApplyCwiseMap<Sin>);
REGISTER_MLIR_INTERPRETER_OP("complex.sqrt", ApplyCwiseMap<Sqrt>);
REGISTER_MLIR_INTERPRETER_OP("complex.tanh", ApplyCwiseMap<TanH>);
REGISTER_MLIR_INTERPRETER_OP(Constant);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
