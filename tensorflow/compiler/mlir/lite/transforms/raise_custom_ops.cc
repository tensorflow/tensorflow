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

#include "absl/container/flat_hash_set.h"
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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// NOLINTNEXTLINE
static llvm::cl::list<std::string> target_ops(
    "tfl-test-raise-tf-targets", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated list of target op names to be wrapped. Only"
                   " used in tests"),
    llvm::cl::CommaSeparated);

namespace mlir {
namespace TFL {
namespace {
// This transformation pass takes an operation with unknown op properties and
// wrap it by a TFL::CustomTfOp.
struct RaiseCustomOpsPass
    : public PassWrapper<RaiseCustomOpsPass, FunctionPass> {
 public:
  explicit RaiseCustomOpsPass()
      : target_op_names(target_ops.begin(), target_ops.end()) {}
  explicit RaiseCustomOpsPass(const std::vector<std::string> &target_ops)
      : target_op_names(target_ops.begin(), target_ops.end()) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-raise-custom-ops";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Raise custom ops into tflite dialect.";
  }

  void runOnFunction() override;

 private:
  // If this set is empty, then all the qualified ops will be wrapped.
  const absl::flat_hash_set<std::string> target_op_names;
};

void RaiseCustomOpsPass::runOnFunction() {
  auto fn = getFunction();
  OpBuilder builder(fn.getContext());

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
        (target_op_names.empty() && !op->getAbstractOperation())) {
      custom_ops.push_back(op);
    }
  });

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
std::unique_ptr<OperationPass<FuncOp>> CreateRaiseCustomOpsPass(
    const std::vector<std::string> &target_ops) {
  return std::make_unique<RaiseCustomOpsPass>(target_ops);
}

static PassRegistration<RaiseCustomOpsPass> pass;

}  // namespace TFL
}  // namespace mlir
