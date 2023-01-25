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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

#include <optional>  // NOLINT

#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue toTensor(InterpreterState&, bufferization::ToTensorOp,
                          const InterpreterValue& in) {
  return in.clone();
}

InterpreterValue toMemref(InterpreterState&, bufferization::ToMemrefOp,
                          const InterpreterValue& in) {
  return in;
}

InterpreterValue allocTensor(
    InterpreterState&, bufferization::AllocTensorOp alloc,
    ArrayRef<int64_t> dynamicSizes, std::optional<InterpreterValue> copy,
    const std::optional<InterpreterValue>& /*sizeHint*/) {
  auto ty = alloc->getResultTypes().front().cast<mlir::ShapedType>();
  auto shape = replaceDynamicVals(ty.getShape(), dynamicSizes);

  if (copy) {
    return copy->clone();
  }
  return InterpreterValue::makeTensor(ty.getElementType(), shape);
}

REGISTER_MLIR_INTERPRETER_OP(allocTensor);
REGISTER_MLIR_INTERPRETER_OP(toMemref);
REGISTER_MLIR_INTERPRETER_OP(toTensor);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
