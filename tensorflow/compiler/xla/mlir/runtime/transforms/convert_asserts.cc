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
#include <utility>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_CONVERTASSERTS
#include "tensorflow/compiler/xla/mlir/runtime/transforms/passes.h.inc"

class ConvertAssertsPass : public impl::ConvertAssertsBase<ConvertAssertsPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------====/

class AssertOpLowering : public OpRewritePattern<cf::AssertOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cf::AssertOp op,
                                PatternRewriter& rewriter) const override {
    // Check if assertion is inside the exported runtime function.
    auto exported = dyn_cast<func::FuncOp>(op->getParentOp());
    if (!exported || !exported->hasAttr(kExportedAttrName))
      return rewriter.notifyMatchFailure(
          op, "assertion is not inside the exported runtime function");

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value exec_ctx = exported.getArgument(0);

    // Split the block at the assert operation.
    Block* block = op->getBlock();
    Block* ok = rewriter.splitBlock(block, op->getIterator());

    // Set up block for returning error.
    Block* err = rewriter.createBlock(&exported.getBody());
    b.setInsertionPointToStart(err);
    b.create<SetErrorOp>(exec_ctx, op.getMsg());
    b.create<func::ReturnOp>();

    // Branch into the error block if assertion failed.
    b.setInsertionPointToEnd(block);
    b.create<cf::CondBranchOp>(op.getArg(), ok, err);

    // Erase the original assert operation.
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------====/

void ConvertAssertsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<AssertOpLowering>(&getContext());

  ModuleOp op = getOperation();
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> CreateConvertAssertsPass() {
  return std::make_unique<ConvertAssertsPass>();
}

}  // namespace runtime
}  // namespace xla
