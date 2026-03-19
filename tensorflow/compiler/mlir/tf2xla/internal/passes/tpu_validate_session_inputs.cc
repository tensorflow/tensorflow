/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>

#include "absl/log/log.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/tpu_validate_inputs_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

#define GEN_PASS_DEF_TPUVALIDATESESSIONINPUTSPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

using mlir::ModuleOp;

struct TPUValidateSessionInputsPass
    : public impl::TPUValidateSessionInputsPassBase<
          TPUValidateSessionInputsPass> {
  void runOnOperation() override;
};

void TPUValidateSessionInputsPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool success = true;

  module.walk([&](mlir::Operation* op) {
    if (IsPotentialUnsupportedOp(op)) {
      LOG(WARNING) << "Potential unsupported op: "
                   << op->getName().getStringRef().str()
                   << ". TF2XLA MLIR bridge does not guarantee to support it.";
    }
    if (!success) {
      signalPassFailure();
    }
  });

  module.walk([&](GraphOp graph) {
    if (HasV1ControlFlow(graph)) {
      LOG(WARNING) << "TF2XLA MLIR bridge does not support v1 control flow."
                   << " Use at your own risk.";
    }
    if (!success) {
      signalPassFailure();
    }
  });
}

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTPUValidateSessionInputsPass() {
  return std::make_unique<TPUValidateSessionInputsPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
