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

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/mark_initialized_variables.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/fake_session.h"

namespace mlir {
namespace {

// This pass is only available in the tf-opt binary for testing.
class MarkInitializedVariablesTestPass
    : public PassWrapper<MarkInitializedVariablesTestPass, FunctionPass> {
 public:
  StringRef getArgument() const final {
    return "tf-saved-model-mark-initialized-variables-test";
  }

  StringRef getDescription() const final {
    return "Mark variables as initialized or not.";
  }

  void runOnFunction() override {
    TF::test_util::FakeSession session;
    if (failed(mlir::tf_saved_model::MarkInitializedVariablesInFunction(
            getFunction(), &session, &getContext())))
      return signalPassFailure();
  }
};

// This pass is only available in the tf-opt binary for testing.
class MarkInitializedVariablesInvalidSessionTestPass
    : public PassWrapper<MarkInitializedVariablesInvalidSessionTestPass,
                         FunctionPass> {
 public:
  StringRef getArgument() const final {
    return "tf-saved-model-mark-initialized-variables-invalid-session-test";
  }

  StringRef getDescription() const final {
    return "Mark variables as initialized or not, but with invalid session.";
  }

  void runOnFunction() override {
    // Pass an invalid session argument, which is a nullptr.
    if (failed(mlir::tf_saved_model::MarkInitializedVariablesInFunction(
            getFunction(), /*session=*/nullptr, &getContext())))
      return signalPassFailure();
  }
};

}  // namespace

namespace tf_saved_model {

static PassRegistration<MarkInitializedVariablesTestPass>
    mark_initialized_variables_test_pass;

static PassRegistration<MarkInitializedVariablesInvalidSessionTestPass>
    mark_initialized_variables_invalid_session_test_pass;

}  // namespace tf_saved_model
}  // namespace mlir
