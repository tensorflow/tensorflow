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

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_RAISECUSTOMOPSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// This transformation pass takes an operation with unknown op properties and
// wrap it by a TFL::CustomTfOp.
struct RaiseCustomOpsPass
    : public impl::RaiseCustomOpsPassBase<RaiseCustomOpsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RaiseCustomOpsPass)

  explicit RaiseCustomOpsPass() {}
  explicit RaiseCustomOpsPass(const std::vector<std::string> &target_ops) {
    this->target_ops_ = target_ops;
  }

  explicit RaiseCustomOpsPass(const RaiseCustomOpsPassOptions &options) {
    this->target_ops_ = options.target_ops_;
  }

  void runOnOperation() override;
};

void RaiseCustomOpsPass::runOnOperation() {
  auto fn = getOperation();
  OpBuilder builder(fn.getContext());

  absl::flat_hash_set<std::string> target_op_names(target_ops_.begin(),
                                                   target_ops_.end());

  llvm::SmallVector<Operation *, 4> custom_ops;
  fn.walk([&](Operation *op) {
    // Skips already imported ops that are imported as CustomTfOp.
    if (op->getParentOfType<CustomTfOp>()) return;
    if (llvm::isa<TFL::CustomTfOp>(op) || llvm::isa<TFL::CustomOp>(op)) return;

    std::string op_name = op->getName().getIdentifier().str();
    // Wrap the operation, if
    // - the op is targeted explicitly, or
    // - the op isn't registered when there are no target list.
    if (target_op_names.contains(op_name) ||
        (target_op_names.empty() && !op->isRegistered())) {
      custom_ops.push_back(op);
    }
  });

  for (auto *op : custom_ops) {
    builder.setInsertionPoint(op);
    Location loc = op->getLoc();
    auto custom_op = builder.create<CustomTfOp>(loc, op->getResultTypes(),
                                                op->getOperands());
    Region region;
    Block *new_block = new Block;
    region.push_back(new_block);

    builder.setInsertionPointToEnd(&region.front());
    Operation *inner_op = builder.clone(*op);

    new_block->addArguments(op->getOperandTypes(),
                            SmallVector<Location>(op->getNumOperands(), loc));
    for (const auto &idx_args : llvm::enumerate(new_block->getArguments())) {
      inner_op->setOperand(idx_args.index(), idx_args.value());
    }
    custom_op->setAttrs(inner_op->getAttrs());
    builder.create<YieldOp>(loc, inner_op->getResults());
    custom_op.getBody().takeBody(region);

    op->replaceAllUsesWith(custom_op);
    op->erase();
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseCustomOpsPass() {
  return std::make_unique<RaiseCustomOpsPass>();
}

// Creates an instance of the TensorFlow Lite dialect raise custom op pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseCustomOpsPass(
    const std::vector<std::string> &target_ops) {
  return std::make_unique<RaiseCustomOpsPass>(target_ops);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateRaiseCustomOpsPass(
    const RaiseCustomOpsPassOptions &options) {
  return std::make_unique<RaiseCustomOpsPass>(options);
}

static PassRegistration<RaiseCustomOpsPass> pass;

}  // namespace TFL
}  // namespace mlir
