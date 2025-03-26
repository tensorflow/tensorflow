/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/internal/utils/dialect_detection_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {
using mlir::Operation;
using mlir::OperationPass;
using mlir::WalkResult;
using mlir::func::FuncOp;

#define GEN_PASS_DEF_VERIFYINPUTDIALECTTOEXECUTORPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/mlir_to_graph_passes.h.inc"

class VerifyInputDialectToExecutorPass
    : public impl::VerifyInputDialectToExecutorPassBase<
          VerifyInputDialectToExecutorPass> {
 public:
  void runOnOperation() override;
};

bool IsTfDeviceClusterFuncOp(Operation* op) {
  std::string kClusterFuncOpName = "tf_device.cluster_func";
  return op->getName().getStringRef().str() == kClusterFuncOpName;
}

void VerifyInputDialectToExecutorPass::runOnOperation() {
  Operation* func_op = getOperation();

  auto walk_result = func_op->walk([&](Operation* op) {
    if (!tensorflow::tf2xla::internal::IsInBridgeAcceptableDialects(op)) {
      std::string error = "op is in dialect " +
                          op->getDialect()->getNamespace().str() +
                          " which is not an accepted dialect";
      op->emitError() << error;
      return WalkResult::interrupt();
    }

    if (IsTfDeviceClusterFuncOp(op)) {
      std::string error =
          "failed TF functional to executor validation, op "
          "tf_device.cluster_func is not allowed";
      op->emitError() << error;
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateVerifyInputDialectToExecutorPass() {
  return std::make_unique<VerifyInputDialectToExecutorPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
