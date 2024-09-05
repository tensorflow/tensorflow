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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/FoldUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORCONSTANTFOLDING
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

constexpr int kMaxIteration = 10;

mlir::LogicalResult FoldConstantOp(mlir::OperationFolder& folder,
                                   mlir::TF::ConstOp op) {
  bool changed = false;
  int i = 0;
  // Iterate until convergence or until maxIterations. Deletion of the op as
  // a result of being dead or folded is convergence.
  do {
    changed = false;

    // If the operation is trivially dead - remove it.
    if (isOpTriviallyDead(op)) {
      op->erase();
      return mlir::success();
    }

    // Try to fold this op.
    bool inPlaceUpdate;
    if (succeeded(folder.tryToFold(op, &inPlaceUpdate))) {
      changed = true;
      if (!inPlaceUpdate) {
        return mlir::success();
      }
    }
  } while (changed && ++i < kMaxIteration);
  return mlir::success();
}

// MLIR pass that folds constants that can be removed or deduplicated away.
struct DTensorConstantFolding
    : public impl::DTensorConstantFoldingBase<DTensorConstantFolding> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OperationFolder helper(&context);

    // Collect and fold the operations within the function.
    llvm::SmallVector<mlir::TF::ConstOp, 8> const_ops;
    getOperation().walk([&](mlir::TF::ConstOp op) { const_ops.push_back(op); });

    // Attempt to fold the specified operation, including handling unused or
    // duplicated constants.
    for (mlir::TF::ConstOp op : llvm::reverse(const_ops))
      if (mlir::failed(FoldConstantOp(helper, op))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorConstantFolding() {
  return std::make_unique<DTensorConstantFolding>();
}

}  // namespace dtensor
}  // namespace tensorflow
