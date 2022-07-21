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
#include "tensorflow/compiler/mlir/tensorflow/transforms/test_passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/fake_session.h"

namespace mlir {
namespace tf_saved_model {
namespace {
using ::tensorflow::Session;

// This pass is only available in the tf-opt binary for testing.
class LiftVariablesTestPass
    : public tf_test::LiftVariablesTestPassBase<LiftVariablesTestPass> {
 public:
  LiftVariablesTestPass() { session_ = new TF::test_util::FakeSession(); }

  ~LiftVariablesTestPass() override { delete session_; }

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
    : public tf_test::LiftVariablesInvalidSessionTestPassBase<
          LiftVariablesInvalidSessionTestPass> {
 public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Pass an invalid session argument, which is a nullptr.
    if (failed(tf_saved_model::LiftVariables(module, /*session=*/nullptr)))
      signalPassFailure();
  }
};

}  // namespace
}  // namespace tf_saved_model

namespace tf_test {

std::unique_ptr<OperationPass<ModuleOp>> CreateLiftVariablesTestPass() {
  return std::make_unique<tf_saved_model::LiftVariablesTestPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftVariablesInvalidSessionTestPass() {
  return std::make_unique<
      tf_saved_model::LiftVariablesInvalidSessionTestPass>();
}

}  // namespace tf_test
}  // namespace mlir
