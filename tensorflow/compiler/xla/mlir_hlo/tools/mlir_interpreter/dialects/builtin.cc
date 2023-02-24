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

#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> unrealizedConversionCast(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState& state) {
  auto resultTy = op->getResultTypes()[0];
  auto operandTy = op->getOperandTypes()[0];
  if (resultTy == operandTy) {
    return {args[0]};
  }

  if (auto r = llvm::dyn_cast<ShapedType>(resultTy)) {
    if (auto o = llvm::dyn_cast<ShapedType>(operandTy)) {
      if (r.getElementType() == o.getElementType() &&
          r.getRank() == o.getRank()) {
        return {args[0]};
      }
    }
  }

  llvm::errs() << "Unimplemented cast: " << *op << "\n";
  llvm_unreachable("unimplemented cast");
}

REGISTER_MLIR_INTERPRETER_OP("builtin.unrealized_conversion_cast",
                             unrealizedConversionCast);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
