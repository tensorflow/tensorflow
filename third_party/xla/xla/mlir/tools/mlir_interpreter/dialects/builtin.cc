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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<InterpreterValue> UnrealizedConversionCast(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState&) {
  auto result_ty = op->getResultTypes()[0];
  auto operand_ty = op->getOperandTypes()[0];
  if (result_ty == operand_ty) {
    return {args[0]};
  }

  if (auto r = llvm::dyn_cast<ShapedType>(result_ty)) {
    if (auto o = llvm::dyn_cast<ShapedType>(operand_ty)) {
      if (verifyCompatibleShapes({o, r}).succeeded()) {
        return {DispatchScalarType(r, [&](auto dummy) -> InterpreterValue {
          TensorOrMemref<decltype(dummy)> result;
          result.view = args[0].View();
          result.buffer = args[0].GetBuffer();
          return {result};
        })};
      }
    }
  }

  llvm::errs() << "Unimplemented cast: " << *op << "\n";
  llvm_unreachable("unimplemented cast");
}

REGISTER_MLIR_INTERPRETER_OP("builtin.unrealized_conversion_cast",
                             UnrealizedConversionCast);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
