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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

// Rewrites `scf.forall` to an `scf.for` loop nest.
LogicalResult rewriteScfForallToScfFor(scf::ForallOp forallOp,
                                       PatternRewriter &rewriter) {
  if (forallOp.getRank() == 0) return failure();
  // Do not convert to scf.for if scf.forall is mapped to threads.
  if (forallOp.getMapping().has_value()) return failure();

  Location loc = forallOp.getLoc();
  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, forallOp.getLowerBound(rewriter),
      forallOp.getUpperBound(rewriter), forallOp.getStep(rewriter),
      forallOp.getOutputs(),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs,
          ValueRange iterArgs) -> scf::ValueVector {
        IRMapping map;
        map.map(forallOp.getInductionVars(), ivs);
        map.map(forallOp.getOutputBlockArguments(), iterArgs);

        for (auto &op : forallOp.getBody()->without_terminator())
          nestedBuilder.clone(op, map);

        auto inParallelOp = forallOp.getTerminator();
        scf::ValueVector results;
        for (auto &op : inParallelOp.getYieldingOps()) {
          auto mappedOperands =
              llvm::to_vector(llvm::map_range(op.getOperands(), [&](Value val) {
                return map.lookupOrDefault(val);
              }));
          results.push_back(rewriter.create<tensor::InsertSliceOp>(
              nestedLoc, mappedOperands, op.getAttrs()));
        }
        rewriter.eraseOp(forallOp.getTerminator());
        return results;
      });

  // Copy attributes from `scf.forall` to the output
  SmallVector<StringAttr> elidedAttrs{forallOp.getOperandSegmentSizesAttrName(),
                                      forallOp.getStaticLowerBoundAttrName(),
                                      forallOp.getStaticUpperBoundAttrName(),
                                      forallOp.getStaticStepAttrName()};
  SmallVector<NamedAttribute> attrs = llvm::to_vector(llvm::make_filter_range(
      forallOp->getAttrs(), [&](const NamedAttribute &attr) {
        return !llvm::is_contained(elidedAttrs, attr.getName());
      }));

  for (scf::ForOp loop : loopNest.loops) {
    rewriter.updateRootInPlace(loop, [&]() {
      for (const auto &attr : attrs)
        loop->setAttr(attr.getName(), attr.getValue());
    });
  }
  rewriter.replaceOp(forallOp, loopNest.results);
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
    patterns.add(rewriteScfForallToScfFor);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRewriteForallOpPass() {
  return std::make_unique<RewriteForallOpPass>();
}

}  // namespace mlir::gml_st
