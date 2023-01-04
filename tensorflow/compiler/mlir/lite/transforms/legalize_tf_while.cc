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

// Converts TF While to TFL While with single call in body and cond.

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_LEGALIZEWHILEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Legalize TF While to TFL While with calls to the original functions from the
// cond and body regions.
struct LegalizeWhilePass
    : public impl::LegalizeWhilePassBase<LegalizeWhilePass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeWhilePass)
  void RunOnFunction(func::FuncOp func);

  void runOnOperation() override {
    for (auto op : getOperation().getOps<func::FuncOp>()) RunOnFunction(op);
  }
};

}  // namespace

// Inserts call to the given function into the 'region'.
void CreateRegionWithCall(func::FuncOp func, Region& region, Location loc) {
  OpBuilder builder(region);
  auto block = builder.createBlock(&region);
  SmallVector<Value, 4> new_operands;
  for (Type t : func.getFunctionType().getInputs())
    new_operands.push_back(block->addArgument(t, loc));
  auto call = builder.create<func::CallOp>(loc, func, new_operands);
  builder.create<YieldOp>(loc, call.getResults());
  // Mark old function as private so that it can be DCE'd if not called.
  func.setPrivate();
}

void RunOnWhile(TF::WhileOp while_op) {
  Operation* op = while_op.getOperation();
  // Create new TFL While op that will be used to replace TF While op.
  auto new_op = OpBuilder(op).create<TFL::WhileOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands(),
      while_op.getIsStateless());
  Location loc = while_op->getLoc();
  CreateRegionWithCall(while_op.cond_function(), new_op.getCond(), loc);
  CreateRegionWithCall(while_op.body_function(), new_op.getBody(), loc);

  op->replaceAllUsesWith(new_op.getResults());
  op->erase();
}

void LegalizeWhilePass::RunOnFunction(func::FuncOp func) {
  // Convert all TF WhileOps inside the function body to TFL While ops.
  func.getBody().walk([](TF::WhileOp while_op) { RunOnWhile(while_op); });
}

// Creates an instance of the TensorFlow While to TFLite While pass.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFWhilePass() {
  return std::make_unique<LegalizeWhilePass>();
}

static PassRegistration<LegalizeWhilePass> pass;

}  // namespace TFL
}  // namespace mlir
