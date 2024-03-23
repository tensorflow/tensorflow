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

#include "absl/container/flat_hash_set.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_TFRESTOREPRUNEIDENTITYPASS
#define GEN_PASS_DECL_TFRESTOREPRUNEIDENTITYPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class TfRestorePruneIdentityPass
    : public impl::TfRestorePruneIdentityPassBase<TfRestorePruneIdentityPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    absl::flat_hash_set<mlir::Block*> restoration_blocks;
    func.walk([&](mlir::TF::RestoreV2Op restore_op) {
      restoration_blocks.insert(restore_op->getBlock());
    });

    // Only remove identity ops from blocks containing restore op because
    // identity ops in inference functions cannot be optimized away uniformly.
    for (mlir::Block* block : restoration_blocks) {
      block->walk([](mlir::Operation* op) {
        if (llvm::isa<mlir::TF::IdentityOp, mlir::TF::IdentityNOp>(op)) {
          op->replaceAllUsesWith(op->getOperands());
          op->erase();
        }
      });
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfRestorePruneIdentityPass() {
  return std::make_unique<TfRestorePruneIdentityPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
