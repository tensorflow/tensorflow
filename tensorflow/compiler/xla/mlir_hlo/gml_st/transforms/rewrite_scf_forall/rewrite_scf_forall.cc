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

#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

// Rewrites `scf.forall` to an `scf.for` loop nest.
LogicalResult rewriteScfForallToScfFor(scf::ForallOp op,
                                       PatternRewriter &rewriter) {
  if (op.getRank() == 0) return failure();
  // Do not convert to scf.for if scf.forall is mapped to threads.
  if (op.getMapping().has_value()) return failure();
  Location loc = op.getLoc();

  rewriter.setInsertionPoint(op);
  SmallVector<scf::ForOp> forOps;
  SmallVector<Value> ivs;
  ValueRange iterArgs = op.getOutputs();
  for (auto [lower, upper, step] :
       llvm::zip(op.getLowerBound(rewriter), op.getUpperBound(rewriter),
                 op.getStep(rewriter))) {
    auto forOp = forOps.emplace_back(
        rewriter.create<scf::ForOp>(loc, lower, upper, step, iterArgs));
    iterArgs = forOp.getRegionIterArgs();
    forOp->setAttrs(op->getAttrs());
    ivs.push_back(forOp.getInductionVar());

    if (forOps.size() > 1) {
      rewriter.create<scf::YieldOp>(loc, forOp.getResults());
    }
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  rewriter.replaceAllUsesWith(op.getInductionVars().drop_back(),
                              ValueRange{ivs}.drop_back(1));
  op.getBody()->eraseArguments(0, forOps.size() - 1);

  forOps.back().getRegion().takeBody(op.getRegion());
  rewriter.replaceOp(op, forOps.front().getResults());

  return success();
}

// Rewrites `in_parallel { parallel_insert_slice* }` to
// `insert_slice*; scf.yield`.
LogicalResult rewriteScfInParallel(scf::InParallelOp inParallelOp,
                                   PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(inParallelOp);
  SmallVector<Value> results;
  for (auto &op :
       llvm::make_early_inc_range(inParallelOp.getRegion().getOps())) {
    auto parallelInsertSlice =
        llvm::dyn_cast<tensor::ParallelInsertSliceOp>(op);
    if (!parallelInsertSlice) return failure();

    results.push_back(rewriter.create<tensor::InsertSliceOp>(
        op.getLoc(), op.getOperands(), op.getAttrs()));
  }
  rewriter.create<scf::YieldOp>(inParallelOp.getLoc(), results);
  rewriter.eraseOp(inParallelOp);

  return success();
}

#define GEN_PASS_DEF_REWRITEFORALLOPPASS
#include "gml_st/transforms/passes.h.inc"

class RewriteForallOpPass
    : public impl::RewriteForallOpPassBase<RewriteForallOpPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    patterns.add(rewriteScfForallToScfFor);
    patterns.add(rewriteScfInParallel);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRewriteForallOpPass() {
  return std::make_unique<RewriteForallOpPass>();
}

}  // namespace gml_st
}  // namespace mlir
