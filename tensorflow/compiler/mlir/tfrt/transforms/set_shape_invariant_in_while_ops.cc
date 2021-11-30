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
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

class SetShapeInvariantInWhileOps
    : public mlir::PassWrapper<SetShapeInvariantInWhileOps,
                               mlir::OperationPass<mlir::FuncOp>> {
 public:
  void runOnOperation() override {
    mlir::FuncOp func_op = getOperation();

    auto shape_invariant = mlir::UnitAttr::get(&getContext());

    func_op.walk(
        [&](mlir::TF::WhileOp op) { op.shape_invariantAttr(shape_invariant); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateSetShapeInvariantInWhileOps() {
  return std::make_unique<SetShapeInvariantInWhileOps>();
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
