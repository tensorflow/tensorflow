/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <memory>
#include <utility>

#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_REWRITEREALLOCTOALLOCPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class RewriteReallocToAllocPass
    : public impl::RewriteReallocToAllocPassBase<RewriteReallocToAllocPass> {
  void runOnOperation() override;
};

class ReallocToAllocRewriter : public OpRewritePattern<memref::ReallocOp> {
  using OpRewritePattern::OpRewritePattern;
  // Rewrites a Realloc to alloc + copy
  LogicalResult matchAndRewrite(memref::ReallocOp op,
                                PatternRewriter& rewriter) const override {
    Value alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(), op.getType(), op.getOperands().drop_front(1),
        op.getAlignmentAttr());
    rewriter.create<memref::CopyOp>(op.getLoc(), op.getSource(), alloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void RewriteReallocToAllocPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  patterns.insert<ReallocToAllocRewriter>(ctx);

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRewriteReallocToAllocPass() {
  return std::make_unique<RewriteReallocToAllocPass>();
}

}  // namespace cpu
}  // namespace xla
