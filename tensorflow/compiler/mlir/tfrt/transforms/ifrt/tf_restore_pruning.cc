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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_TFRESTOREPRUNINGPASS
#define GEN_PASS_DECL_TFRESTOREPRUNINGPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

// Prune unused RestoreV2 Op.
class TfRestorePruningPass
    : public impl::TfRestorePruningPassBase<TfRestorePruningPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    func.walk([&](mlir::TF::RestoreV2Op restore) {
      if (restore.use_empty()) {
        restore.erase();
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTfRestorePruningPass() {
  return std::make_unique<TfRestorePruningPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
