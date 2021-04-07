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

namespace mlir {
namespace TFL {
namespace {
// This transformation pass takes an operation wrapped by a TFL::CustomTfOp and
// inline it.
struct LowerCustomOpsPass
    : public PassWrapper<LowerCustomOpsPass, FunctionPass> {
 public:
  explicit LowerCustomOpsPass() {}

  void runOnFunction() override;
};

void LowerCustomOpsPass::runOnFunction() {
  auto fn = getFunction();
  OpBuilder builder(fn.getContext());

  llvm::SmallVector<Operation*, 4> wrapped_ops;
  fn.walk([&](TFL::CustomTfOp custom_op) {
    auto* real_op = &custom_op.body().front().front();
    wrapped_ops.push_back(real_op);
  });

  for (auto* op : wrapped_ops) {
    auto parent_op = op->getParentOfType<TFL::CustomTfOp>();
    if (!parent_op) continue;
    builder.setInsertionPoint(parent_op);

    BlockAndValueMapping mapping;
    for (auto vals : llvm::zip(op->getOperands(), parent_op->getOperands())) {
      mapping.map(std::get<0>(vals), std::get<1>(vals));
    }
    Operation* inlined = builder.clone(*op, mapping);

    parent_op->replaceAllUsesWith(inlined);
    parent_op->erase();
  }
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect lower custom op pass.
std::unique_ptr<OperationPass<FuncOp>> CreateLowerCustomOpsPass() {
  return std::make_unique<LowerCustomOpsPass>();
}

static PassRegistration<LowerCustomOpsPass> pass(
    "tfl-lower-custom-ops", "Lower custom ops from tflite dialect.");

}  // namespace TFL
}  // namespace mlir
