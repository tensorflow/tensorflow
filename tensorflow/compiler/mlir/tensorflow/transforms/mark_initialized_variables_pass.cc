/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace tf_saved_model {
namespace {
// Checks if a function has only one block.
mlir::LogicalResult CheckSingleBlockFunction(FuncOp function) {
  if (!llvm::hasSingleElement(function)) {
    return function.emitError()
           << "expects function '" << function.getName()
           << "' to have 1 block, got " << function.getBlocks().size();
  }
  return success();
}

class MarkInitializedVariablesPass
    : public PassWrapper<MarkInitializedVariablesPass, FunctionPass> {
 public:
  explicit MarkInitializedVariablesPass(tensorflow::Session* session = nullptr)
      : session_(session) {}
  void runOnFunction() override;

 private:
  tensorflow::Session* session_;
};

void MarkInitializedVariablesPass::runOnFunction() {
  // We use session to check for variables value. If it is null then
  // nothing to do here.
  if (!session_) return;
  auto func_op = getFunction();
  if (failed(CheckSingleBlockFunction(func_op))) return signalPassFailure();

  if (failed(MarkInitializedVariablesInFunction(func_op, session_,
                                                &getContext()))) {
    return signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateMarkInitializedVariablesPass(
    tensorflow::Session* session) {
  return std::make_unique<MarkInitializedVariablesPass>(session);
}
}  // namespace tf_saved_model
}  // namespace mlir
