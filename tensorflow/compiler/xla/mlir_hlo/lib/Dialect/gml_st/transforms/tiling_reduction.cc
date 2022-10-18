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

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define GEN_PASS_DEF_TILINGREDUCTIONPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

using namespace mlir;

namespace {
struct TilingReductionPattern : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter& rewriter) const override;
};

struct TilingReductionPass
    : public ::impl::TilingReductionPassBase<TilingReductionPass> {
  void runOnOperation() override;
};
}  // namespace

LogicalResult TilingReductionPattern::matchAndRewrite(
    linalg::GenericOp genericOp, PatternRewriter& rewriter) const {
  // Match only if it's a linalg.generic tensor<1x?xf32> -> tensor<1xf32> with
  // iterator_types = ["parallel", "reduction"].
  auto itTypes = llvm::to_vector(
      genericOp.getIteratorTypes().getAsValueRange<StringAttr>());
  if (itTypes.size() != 2 || itTypes[0] != getParallelIteratorTypeName() ||
      itTypes[1] != getReductionIteratorTypeName()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Expected ['parallel', 'reduction']");
  }
  if (genericOp.getNumInputs() != 1 || genericOp.getNumOutputs() != 1) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Expected single input and output");
  }
  auto input = genericOp.getInputs().front();
  auto output = genericOp.getOutputs().front();
  auto inType = input.getType().dyn_cast<TensorType>();
  auto outType = output.getType().dyn_cast<TensorType>();
  constexpr int kWarpSize = 32;
  if (!inType || !inType.hasRank() || inType.getRank() != 2 ||
      inType.getShape().front() != 1 || inType.getShape().back() == kWarpSize) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Expected 2D row tensor input");
  }
  if (!outType || !outType.hasStaticShape() || outType.getRank() != 1 ||
      outType.getNumElements() != 1) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "Expected 1D 1-element output");
  }

  Location loc = genericOp->getLoc();
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value cWarpSize = rewriter.create<arith::ConstantIndexOp>(loc, kWarpSize);
  Value reductionDim = rewriter.create<tensor::DimOp>(loc, input, c1);
  OpFoldResult oneAttr = rewriter.getIndexAttr(1);
  auto warpDist = rewriter.getStringAttr("warp");

  // Create warp-sized partial reduction result tensor.
  Value partial = rewriter.create<tensor::EmptyOp>(loc, kWarpSize,
                                                   outType.getElementType());
  Value outPoint = rewriter.create<gml_st::SpaceOp>(loc, oneAttr);
  Value partSpace = rewriter.create<gml_st::SpaceOp>(
      loc, OpFoldResult(rewriter.getIndexAttr(kWarpSize)));

  auto getResult = [](Operation* op) { return op->getResult(0); };

  // Create gml_st.parallel initializing the partial result.
  partial = getResult(rewriter.create<gml_st::ParallelOp>(
      loc, partial.getType(), c0, cWarpSize, c1, warpDist,
      [&](OpBuilder& builder, Location loc, ValueRange ivs) {
        OpFoldResult laneIdx = ivs.front();
        Value partPoint = builder.create<gml_st::TileOp>(
            loc, partSpace, laneIdx, oneAttr, oneAttr);
        builder.create<gml_st::SetYieldOp>(loc, output, partial, partPoint);
      }));

  // Create gml_st.parallel finalizing the partial result.
  partial = getResult(rewriter.create<gml_st::ParallelOp>(
      loc, partial.getType(), c0, cWarpSize, c1, warpDist,
      [&](OpBuilder& builder, Location loc, ValueRange ivs) {
        Value laneIdx = ivs.front();
        Value partPoint = builder.create<gml_st::TileOp>(
            loc, partSpace, OpFoldResult(laneIdx), oneAttr, oneAttr);
        Value initVal =
            builder.create<gml_st::MaterializeOp>(loc, partial, partPoint);
        // Create gml_st.for sequentially reducing parts of the row.
        auto forOp = builder.create<gml_st::ForOp>(
            loc, outType, laneIdx, reductionDim, cWarpSize, initVal,
            [&](OpBuilder& builder, Location loc, ValueRange ivs,
                ValueRange outputs) {
              Value colIdx = ivs.front();
              Value partElement = outputs.front();
              using OFRs = ArrayRef<OpFoldResult>;
              Value inSpace = builder.create<gml_st::SpaceOp>(
                  loc, OFRs{oneAttr, reductionDim});
              Value inPoint = builder.create<gml_st::TileOp>(
                  loc, inSpace, OFRs{rewriter.getIndexAttr(0), colIdx},
                  OFRs{oneAttr, oneAttr}, OFRs{oneAttr, oneAttr});
              Value inElement =
                  builder.create<gml_st::MaterializeOp>(loc, input, inPoint);
              // Clone linalg.generic op reducing two 1-element tensors.
              Operation* clonedOp = builder.clone(*genericOp.getOperation());
              clonedOp->setOperands({inElement, partElement});
              builder.create<gml_st::SetYieldOp>(loc, clonedOp->getResult(0),
                                                 partElement, outPoint);
            });
        builder.create<gml_st::SetYieldOp>(loc, forOp.getResults(), partial,
                                           partPoint);
      }));

  // Change existing linalg.generic to warp-reduce the partial result.
  partial = rewriter.create<tensor::ExpandShapeOp>(
      loc, outType.clone({1, 32}), partial, ReassociationIndices{0, 1});
  rewriter.updateRootInPlace(genericOp,
                             [&] { genericOp->setOperand(0, partial); });

  return success();
}

void TilingReductionPass::runOnOperation() {
  FrozenRewritePatternSet patterns = RewritePatternSet(
      &getContext(), std::make_unique<TilingReductionPattern>(&getContext()));
  auto walkResult = getOperation()->walk([&](linalg::GenericOp genericOp) {
    if (failed(applyOpPatternsAndFold(genericOp, patterns)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) return signalPassFailure();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::gml_st::createTilingReductionPass() {
  return std::make_unique<TilingReductionPass>();
}
