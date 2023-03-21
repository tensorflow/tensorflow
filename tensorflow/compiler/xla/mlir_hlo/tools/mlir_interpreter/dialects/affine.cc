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

#include <algorithm>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "tools/mlir_interpreter/dialects/util.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<int64_t> apply(InterpreterState&, AffineApplyOp op,
                                 ArrayRef<int64_t> operands) {
  return evalAffineMap(op.getAffineMap(), operands);
}

int64_t min(InterpreterState&, AffineMinOp op, ArrayRef<int64_t> operands) {
  auto results = evalAffineMap(op.getAffineMap(), operands);
  return *std::min_element(results.begin(), results.end());
}

int64_t max(InterpreterState&, AffineMaxOp op, ArrayRef<int64_t> operands) {
  auto results = evalAffineMap(op.getAffineMap(), operands);
  return *std::max_element(results.begin(), results.end());
}

REGISTER_MLIR_INTERPRETER_OP(apply);
REGISTER_MLIR_INTERPRETER_OP(max);
REGISTER_MLIR_INTERPRETER_OP(min);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
