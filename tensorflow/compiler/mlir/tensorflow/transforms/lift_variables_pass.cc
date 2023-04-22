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

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/savedmodel_passes_detail.h"
#include "tensorflow/core/public/session.h"

namespace mlir {
namespace {

// This pass takes care of finding all variables from the function arguments and
// converting them to the corresponding global tensors, that will be located out
// of function. Also it converts resource arguments from function types to the
// corresponding saved model arguments accordingly.
class LiftVariablesPass
    : public tf_saved_model::SavedModelLiftVariablePassBase<LiftVariablesPass> {
 public:
  explicit LiftVariablesPass(tensorflow::Session* session)
      : session_(session) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(tf_saved_model::LiftVariables(module, session_)))
      signalPassFailure();
  }

 private:
  ::tensorflow::Session* session_;
};

}  // namespace

namespace tf_saved_model {

std::unique_ptr<OperationPass<ModuleOp>> CreateLiftVariablesPass(
    tensorflow::Session* session) {
  return std::make_unique<LiftVariablesPass>(session);
}

}  // namespace tf_saved_model
}  // namespace mlir
