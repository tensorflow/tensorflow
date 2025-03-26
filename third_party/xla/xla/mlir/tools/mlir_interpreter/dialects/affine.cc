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

#include <algorithm>
#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

llvm::SmallVector<int64_t> Apply(InterpreterState&, affine::AffineApplyOp op,
                                 ArrayRef<int64_t> operands) {
  return EvalAffineMap(op.getAffineMap(), operands);
}

int64_t Min(InterpreterState&, affine::AffineMinOp op,
            ArrayRef<int64_t> operands) {
  auto results = EvalAffineMap(op.getAffineMap(), operands);
  return *std::min_element(results.begin(), results.end());
}

int64_t Max(InterpreterState&, affine::AffineMaxOp op,
            ArrayRef<int64_t> operands) {
  auto results = EvalAffineMap(op.getAffineMap(), operands);
  return *std::max_element(results.begin(), results.end());
}

REGISTER_MLIR_INTERPRETER_OP(Apply);
REGISTER_MLIR_INTERPRETER_OP(Max);
REGISTER_MLIR_INTERPRETER_OP(Min);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
