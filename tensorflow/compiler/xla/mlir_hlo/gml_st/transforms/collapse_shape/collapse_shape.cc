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
#include <string>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/linalg_utils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_COLLAPSESHAPEPASS
#include "gml_st/transforms/passes.h.inc"

// Creates reassociation indices for `shape_collapse` and `shape_expand` ops.
// Given `rank`(N) and `retainTrailingDims`(M), returns the following
// reassociation:
//     [[0, 1, ..., N-M-1], [N-M], [N-M+1], ..., [N-1]]
//                         |--- retainTrailingDims ---|
//     |-------------------- rank --------------------|
SmallVector<ReassociationIndices> getCollapsingReassociationIndices(
    int64_t rank, int64_t retainTrailingDims) {
  SmallVector<ReassociationIndices> reassociation;
  reassociation.reserve(retainTrailingDims + 1);
  if (rank > retainTrailingDims) {
    auto seq = llvm::seq<int64_t>(0, rank - retainTrailingDims);
    reassociation.emplace_back(seq.begin(), seq.end());
  }
  for (int64_t i = rank - retainTrailingDims; i < rank; ++i)
    reassociation.push_back({i});
  return reassociation;
}

struct CollapseBcastPattern : OpRewritePattern<linalg::BroadcastOp> {
  using OpRewritePattern<linalg::BroadcastOp>::OpRewritePattern;

  CollapseBcastPattern(MLIRContext* ctx, int64_t retainTrailingDims)
      : OpRewritePattern<linalg::BroadcastOp>(ctx),
        retainTrailingDims(retainTrailingDims) {}

  LogicalResult matchAndRewrite(linalg::BroadcastOp op,
                                PatternRewriter& rewriter) const override {
    Value init = op.getInit();
    auto initTy = init.getType().cast<RankedTensorType>();
    int64_t initRank = initTy.getRank();
    int64_t numCollapsedDims = initRank - retainTrailingDims;

    if (numCollapsedDims < 2) {
      return rewriter.notifyMatchFailure(op, "no dimension to collapse");
    }

    // Dimensions to be collapsed must either be all broadcasted or not
    // broadcasted.
    llvm::ArrayRef<int64_t> nonBroadcastedDims = op.getDimensions();

    bool firstDimsBroadcasted = true;
    if (!nonBroadcastedDims.empty()) {
      int64_t i = 0;
      while (i < (int64_t)nonBroadcastedDims.size() &&
             nonBroadcastedDims[i] == i && i < numCollapsedDims) {
        ++i;
      }
      if (i >= numCollapsedDims) {
        firstDimsBroadcasted = false;
      } else if (llvm::any_of(nonBroadcastedDims,
                              [numCollapsedDims](unsigned dim) {
                                return dim < numCollapsedDims;
                              })) {
        return rewriter.notifyMatchFailure(
            op, "collapsed dims are not broadcasted in order");
      }
    }

    Value operand = op.getInput();
    auto operandTy = operand.getType().cast<RankedTensorType>();
    int64_t operandRank = operandTy.getRank();
    llvm::DenseSet<int64_t> nonBroadcastedDimsSet(nonBroadcastedDims.begin(),
                                                  nonBroadcastedDims.end());
    llvm::SmallVector<int64_t> collapsedNonBroadcastedDims;
    collapsedNonBroadcastedDims.reserve(numCollapsedDims +
                                        (firstDimsBroadcasted ? 1 : 0));
    for (int64_t dim = numCollapsedDims; dim < initRank; ++dim) {
      if (nonBroadcastedDimsSet.contains(dim)) {
        collapsedNonBroadcastedDims.push_back(dim - numCollapsedDims + 1);
      }
    }
    int64_t operandRetainTrailingDims =
        retainTrailingDims - collapsedNonBroadcastedDims.size();

    // Collapse operand and init tensor.
    // For bcasts, this retains the last `retainTrailingDims` dimensions of the
    // *result* and collapses all others.
    Location loc = op.getLoc();
    Value collapsedOperand = operand;
    if (operandRank > operandRetainTrailingDims + 1) {
      SmallVector<ReassociationIndices> operandReassociation =
          getCollapsingReassociationIndices(operandRank,
                                            operandRetainTrailingDims);
      collapsedOperand = rewriter.createOrFold<tensor::CollapseShapeOp>(
          loc, operand, operandReassociation);
    }
    SmallVector<ReassociationIndices> initReassociation =
        getCollapsingReassociationIndices(initRank, retainTrailingDims);
    Value collapsedInit =
        rewriter.create<tensor::CollapseShapeOp>(loc, init, initReassociation);

    // Create collapsed bcast op.
    if (!firstDimsBroadcasted) {
      collapsedNonBroadcastedDims.push_back(0);
    }
    Value collapsedBcastOp =
        rewriter
            .create<linalg::BroadcastOp>(
                loc, collapsedOperand, collapsedInit,
                ArrayRef<int64_t>(collapsedNonBroadcastedDims))
            .getResult()
            .front();

    // Re-expand broadcast op and replace the original.
    auto reexpandedBcastOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, initTy, collapsedBcastOp, initReassociation);
    rewriter.replaceOp(op, reexpandedBcastOp.getResult());
    return success();
  }

 private:
  int64_t retainTrailingDims;
};

struct CollapseReductionPattern : OpRewritePattern<linalg::ReduceOp> {
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  CollapseReductionPattern(MLIRContext* ctx, int64_t retainTrailingDims)
      : OpRewritePattern<linalg::ReduceOp>(ctx),
        retainTrailingDims(retainTrailingDims) {}

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumDpsInits() != 1 || op.getDimensions().empty())
      return failure();
    int64_t reductionDim = op.getDimensions()[0];

    Value operand = op.getInputs().front();
    auto operandTy = operand.getType().cast<RankedTensorType>();
    int64_t operandRank = operandTy.getRank();

    if (operandRank <= retainTrailingDims + 1) {
      return rewriter.notifyMatchFailure(op, "no dimension to collapse");
    }

    if (operandRank - 1 - reductionDim >= retainTrailingDims) {
      return rewriter.notifyMatchFailure(
          op, "reduction dimension must be retained");
    }

    Value init = op.getInits().front();
    auto initTy = init.getType().cast<RankedTensorType>();
    int64_t initRank = initTy.getRank();

    // Collapse operand and init tensor.
    // For reductions, this retains the last `retainTrailingDims` dimensions of
    // the *operand* and collapses all others.
    Location loc = op.getLoc();
    SmallVector<ReassociationIndices> operandReassociation =
        getCollapsingReassociationIndices(operandRank, retainTrailingDims);
    Value collapsedOperand = rewriter.create<tensor::CollapseShapeOp>(
        loc, operand, operandReassociation);
    SmallVector<ReassociationIndices> initReassociation =
        getCollapsingReassociationIndices(initRank, retainTrailingDims - 1);
    Value collapsedInit =
        rewriter.create<tensor::CollapseShapeOp>(loc, init, initReassociation);

    auto collapsedOperandTy =
        collapsedOperand.getType().cast<RankedTensorType>();
    int64_t collapsedOperandRank = collapsedOperandTy.getRank();
    auto collapsedInitTy = collapsedInit.getType().cast<RankedTensorType>();

    // Create collapsed reduction op.
    int64_t collapsedReductionDim =
        reductionDim - operandRank + collapsedOperandRank;
    SmallVector<utils::IteratorType> collapsedIteratorTypes(
        collapsedOperandRank, utils::IteratorType::parallel);
    collapsedIteratorTypes[collapsedReductionDim] =
        utils::IteratorType::reduction;
    auto collapsedReductionOp = rewriter.create<linalg::ReduceOp>(
        loc, collapsedInitTy, collapsedOperand, collapsedInit,
        ArrayRef<int64_t>({collapsedReductionDim}));
    collapsedReductionOp.getRegion().takeBody(op.getBodyRegion());

    // Re-expand reduction op and replace the original.
    auto reexpandedReductionOp = rewriter.create<tensor::ExpandShapeOp>(
        loc, initTy, collapsedReductionOp.getResults().front(),
        initReassociation);
    rewriter.replaceOp(op, reexpandedReductionOp.getResult());
    return success();
  }

 private:
  int64_t retainTrailingDims;
};

linalg::MapOp createCollapsedMapOp(
    linalg::MapOp mapOp, PatternRewriter& rewriter,
    const SmallVector<ReassociationIndices>& reassociation) {
  // Collapsed operands and init tensor.
  Location loc = mapOp.getLoc();
  SmallVector<Value> collapsedOperands = llvm::to_vector(
      llvm::map_range(mapOp.getInputs(), [&](Value it) -> Value {
        return rewriter.create<tensor::CollapseShapeOp>(loc, it, reassociation);
      }));
  Value init = mapOp.getInit();
  Value collapsedInit =
      rewriter.create<tensor::CollapseShapeOp>(loc, init, reassociation);

  // Create collapsed map op.
  auto collapsedInitTy = collapsedInit.getType().cast<RankedTensorType>();
  auto collapsedMapOp = rewriter.create<linalg::MapOp>(
      loc, collapsedInitTy, collapsedOperands, collapsedInit);
  IRMapping bvm;
  mapOp.getBodyRegion().cloneInto(&collapsedMapOp.getRegion(), bvm);
  return collapsedMapOp;
}

struct CollapseMapPattern : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  CollapseMapPattern(MLIRContext* ctx, int64_t retainTrailingDims)
      : OpRewritePattern<linalg::MapOp>(ctx),
        retainTrailingDims(retainTrailingDims) {}

  LogicalResult matchAndRewrite(linalg::MapOp op,
                                PatternRewriter& rewriter) const override {
    Value init = op.getInit();
    auto initTy = init.getType().cast<RankedTensorType>();
    int64_t rank = initTy.getRank();

    if (rank <= retainTrailingDims + 1) {
      return rewriter.notifyMatchFailure(op, "no dimension to collapse");
    }

    SmallVector<ReassociationIndices> reassociation =
        getCollapsingReassociationIndices(rank, retainTrailingDims);
    auto collapsedMapOp = createCollapsedMapOp(op, rewriter, reassociation);

    // Re-expand map op and replace the original.
    auto reexpandedMapOp = rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), initTy, collapsedMapOp.getResult().front(), reassociation);
    rewriter.replaceOp(op, reexpandedMapOp.getResult());
    return success();
  }

 private:
  int64_t retainTrailingDims;
};

struct MoveCollapseBeforeMapPattern
    : OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;

  explicit MoveCollapseBeforeMapPattern(MLIRContext* ctx)
      : OpRewritePattern<tensor::CollapseShapeOp>(ctx) {}

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp op,
                                PatternRewriter& rewriter) const override {
    auto mapOp = op.getSrc().getDefiningOp<linalg::MapOp>();
    if (!mapOp) return failure();
    auto collapsedMapOp =
        createCollapsedMapOp(mapOp, rewriter, op.getReassociationIndices());
    rewriter.replaceOp(op, collapsedMapOp.getResult());
    return success();
  }
};

struct CollapseShapePass
    : public impl::CollapseShapePassBase<CollapseShapePass> {
  using CollapseShapePassBase<CollapseShapePass>::CollapseShapePassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    CollapseShapePassBase<CollapseShapePass>::getDependentDialects(registry);

    // TODO(frgossen): Move these iface implementations into the tensor dialect.
    // Some of its canonicalizations depend on it. Until then, we have to
    // register them explicitly.
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* ctx = &getContext();

    // Populate shape-collapsing patterns for cwise ops, reductions, and bcasts.
    RewritePatternSet patterns(ctx);
    patterns.add<CollapseBcastPattern, CollapseMapPattern,
                 CollapseReductionPattern>(ctx, retainTrailingDims);
    // By moving CollapseShapeOp before MapOp, we can potentially remove it if
    // it cancels out with an ExpandShapeOp.
    patterns.add<MoveCollapseBeforeMapPattern>(ctx);

    // Collect some related canonicalization patterns.
    linalg::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::MapOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::ReduceOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorEmptyPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createCollapseShapePass() {
  return std::make_unique<CollapseShapePass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createCollapseShapePass(
    const CollapseShapePassOptions& options) {
  return std::make_unique<CollapseShapePass>(options);
}

}  // namespace gml_st
}  // namespace mlir
