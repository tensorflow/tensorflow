/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/OperationSupport.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_interface.h.inc"
namespace TFL {
namespace {

// This pass verifies that the operands and results types are supported by
// TFLite runtime.
class RuntimeTypeVerifyPass : public mlir::FunctionPass<RuntimeTypeVerifyPass> {
 public:
  explicit RuntimeTypeVerifyPass() {}

 private:
  void runOnFunction() override;
};

void RuntimeTypeVerifyPass::runOnFunction() {
  getFunction().walk([&](TflRuntimeVerifyOpInterface op) {
    if (failed(op.VerifyTflRuntimeTypes(op.getOperation())))
      signalPassFailure();
  });
}
}  // namespace

// Verifies runtime supports types used.
std::unique_ptr<OpPassBase<FuncOp>> CreateRuntimeTypeVerifyPass() {
  return std::make_unique<RuntimeTypeVerifyPass>();
}

static PassRegistration<RuntimeTypeVerifyPass> pass(
    "tfl-runtime-verify", "TFLite runtime verification");

}  // namespace TFL
}  // namespace mlir
