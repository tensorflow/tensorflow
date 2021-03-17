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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
// This transformation pass takes an operation with unknown op properties and
// wrap it by a TFL::CustomTfOp.
struct RaiseCustomOpsPass
    : public PassWrapper<RaiseCustomOpsPass, FunctionPass> {
  void runOnFunction() override;
};

void RaiseCustomOpsPass::runOnFunction() {
  auto fn = getFunction();
  OpBuilder builder(fn.getContext());

  llvm::SmallVector<Operation *, 4> custom_ops;
  for (Operation &op : fn.getOps()) {
    // Skips the ops with known op property.
    if (op.getAbstractOperation()) continue;
    // Skips already imported ops that are imported as CustomTfOp.
    if (op.getParentOfType<CustomTfOp>()) continue;
    if (llvm::isa<TFL::CustomTfOp>(op) || llvm::isa<TFL::CustomOp>(op))
      continue;
    custom_ops.push_back(&op);
  }

  for (auto *op : custom_ops) {
    builder.setInsertionPoint(op);
    auto custom_op = builder.create<CustomTfOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands());
    Region region;
    Block *new_block = new Block;
    region.push_back(new_block);

    builder.setInsertionPointToEnd(&region.front());
    Operation *inner_op = builder.clone(*op);

    new_block->addArguments(op->getOperandTypes());
    for (auto idx_args : llvm::enumerate(new_block->getArguments())) {
      inner_op->setOperand(idx_args.index(), idx_args.value());
    }
    custom_op->setAttrs(inner_op->getAttrs());
    builder.create<YieldOp>(op->getLoc(), inner_op->getResults());
    custom_op.body().takeBody(region);

    op->replaceAllUsesWith(custom_op);
    op->erase();
  }
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect raise custom op pass.
std::unique_ptr<OperationPass<FuncOp>> CreateRaiseCustomOpsPass() {
  return std::make_unique<RaiseCustomOpsPass>();
}

static PassRegistration<RaiseCustomOpsPass> pass(
    "tfl-raise-custom-ops", "Raise custom ops into tflite dialect.");

}  // namespace TFL
}  // namespace mlir
