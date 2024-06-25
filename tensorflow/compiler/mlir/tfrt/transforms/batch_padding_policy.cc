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
#include <string>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

class BatchPaddingPolicyPass
    : public mlir::PassWrapper<BatchPaddingPolicyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit BatchPaddingPolicyPass(const std::string& batch_padding_policy)
      : mlir::PassWrapper<BatchPaddingPolicyPass,
                          mlir::OperationPass<mlir::ModuleOp>>(),
        batch_padding_policy_(batch_padding_policy) {}

  BatchPaddingPolicyPass() = default;
  BatchPaddingPolicyPass(const BatchPaddingPolicyPass&) = default;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchPaddingPolicyPass)

 private:
  llvm::StringRef getArgument() const final { return "batch-padding-policy"; }

  llvm::StringRef getDescription() const final {
    return "Sets the BatchFunction.batch_padding_policy attr.";
  }

  void runOnOperation() override {
    if (!batch_padding_policy_.empty()) {
      mlir::ModuleOp module = getOperation();
      module.walk([&](mlir::TF::BatchFunctionOp batch_op) {
        batch_op.setBatchPaddingPolicy(batch_padding_policy_);
      });
    }
  }

 protected:
  const std::string batch_padding_policy_;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateBatchPaddingPolicyPass(const std::string& batch_padding_policy) {
  return std::make_unique<BatchPaddingPolicyPass>(batch_padding_policy);
}

static mlir::PassRegistration<BatchPaddingPolicyPass> register_pass;

}  // namespace tfrt_compiler
}  // namespace tensorflow
