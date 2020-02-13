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

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {

struct ConvertTFWhileOp : public OpRewritePattern<TF::WhileOp> {
  using OpRewritePattern<TF::WhileOp>::OpRewritePattern;
  PatternMatchResult matchAndRewrite(TF::WhileOp while_op,
                                     PatternRewriter& rewriter) const override;
};

PatternMatchResult ConvertTFWhileOp::matchAndRewrite(
    TF::WhileOp while_op, PatternRewriter& rewriter) const {
  Operation* op = while_op.getOperation();
  auto new_op =
      rewriter.create<TFL::WhileOp>(op->getLoc(), op->getResultTypes(),
                                    op->getOperands(), while_op.is_stateless());
  auto build_region = [&](Region& region, FlatSymbolRefAttr func) {
    OpBuilder b(region);
    auto block = b.createBlock(&region);
    SmallVector<Value, 4> new_operands;
    auto fn = while_op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
        func.getValue());
    auto types = fn.getType().getResults();
    for (Type t : fn.getType().getInputs())
      new_operands.push_back(block->addArgument(t));
    auto call = b.create<CallOp>(while_op.getLoc(), func, types, new_operands);
    b.create<YieldOp>(while_op.getLoc(), call.getResults());
  };
  build_region(new_op.cond(), while_op.condAttr());
  build_region(new_op.body(), while_op.bodyAttr());

  rewriter.replaceOp(op, new_op.getResults());
  return matchSuccess();
}

// Legalize operations in functions.
struct LegalizeWhile : public FunctionPass<LegalizeWhile> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto* ctx = &getContext();
    auto func = getFunction();

    patterns.insert<ConvertTFWhileOp>(ctx);
    applyPatternsGreedily(func, patterns);
  }
};

}  // namespace

// Creates an instance of the TensorFlow While to TFLite While pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateLegalizeTFWhilePass() {
  return std::make_unique<LegalizeWhile>();
}

static PassRegistration<LegalizeWhile> pass(
    "tfl-legalize-tf-while",
    "Legalize from TensorFlow While to TensorFlow Lite While");

}  // namespace TFL
}  // namespace mlir
