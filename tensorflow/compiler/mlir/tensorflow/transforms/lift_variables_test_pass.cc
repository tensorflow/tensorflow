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

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/lift_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/fake_session.h"

namespace mlir {
namespace {
using ::tensorflow::Session;

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesTestPass
    : public PassWrapper<LiftVariablesTestPass, OperationPass<ModuleOp>> {
 public:
  LiftVariablesTestPass() { session_ = new TF::test_util::FakeSession(); }

  ~LiftVariablesTestPass() override { delete session_; }

  StringRef getArgument() const final {
    return "tf-saved-model-lift-variables-test";
  }

  StringRef getDescription() const final {
    return "Lift variables and save them as global tensors";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(tf_saved_model::LiftVariables(module, session_)))
      signalPassFailure();
  }

 private:
  Session* session_;
};

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesInvalidSessionTestPass
    : public PassWrapper<LiftVariablesInvalidSessionTestPass,
                         OperationPass<ModuleOp>> {
 public:
  StringRef getArgument() const final {
    return "tf-saved-model-lift-variables-invalid-session-test";
  }

  StringRef getDescription() const final {
    return "Lift variables and save them as global tensors with an invalid "
           "session";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Pass an invalid session argument, which is a nullptr.
    if (failed(tf_saved_model::LiftVariables(module, /*session=*/nullptr)))
      signalPassFailure();
  }
};

}  // namespace

namespace tf_saved_model {

static PassRegistration<LiftVariablesTestPass> lift_variables_test_pass;

static PassRegistration<LiftVariablesInvalidSessionTestPass>
    lift_variables_invalid_session_test_pass;

}  // namespace tf_saved_model
}  // namespace mlir
