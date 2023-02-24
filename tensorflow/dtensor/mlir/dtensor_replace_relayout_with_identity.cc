/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {
namespace {

#define GEN_PASS_DEF_DTENSORREPLACERELAYOUTWITHIDENTITYPASS
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

class DTensorReplaceRelayoutWithIdentityPass
    : public impl::DTensorReplaceRelayoutWithIdentityPassBase<
          DTensorReplaceRelayoutWithIdentityPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp function = getOperation();
    function.walk([&](mlir::TF::RelayoutOp relayout_op) {
      mlir::OpBuilder builder(relayout_op);
      // Inserts an IdentityOp at the position of the relayout_op with the same
      // attributes as the relayout_op.
      auto new_identity = builder.create<mlir::TF::IdentityOp>(
          relayout_op->getLoc(), relayout_op.getType(), relayout_op.getInput(),
          relayout_op->getAttrs());
      relayout_op.getOutput().replaceAllUsesWith(new_identity.getOutput());
      relayout_op.erase();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorReplaceRelayoutWithIdentityPass() {
  return std::make_unique<DTensorReplaceRelayoutWithIdentityPass>();
}

}  // namespace dtensor
}  // namespace tensorflow
