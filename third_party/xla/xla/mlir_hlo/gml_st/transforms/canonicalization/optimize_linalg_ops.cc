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
#include <optional>
#include <string>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_OPTIMIZELINALGOPSPASS
#include "gml_st/transforms/passes.h.inc"

std::optional<Value> getSplatValue(PatternRewriter& rewriter, Location loc,
                                   Value value) {
  auto* definingOp = value.getDefiningOp();
  if (!definingOp) return std::nullopt;

  if (auto constantOp = dyn_cast_or_null<arith::ConstantOp>(definingOp)) {
    auto denseElementsAttr =
        constantOp.getValue().dyn_cast<DenseElementsAttr>();

    if (!denseElementsAttr.isSplat()) return std::nullopt;

    auto splatAttr = denseElementsAttr.getSplatValue<Attribute>();
    auto splatType = denseElementsAttr.getElementType();

    if (complex::ConstantOp::isBuildableWith(splatAttr, splatType))
      return rewriter.create<complex::ConstantOp>(loc, splatType,
                                                  splatAttr.cast<ArrayAttr>());

    return rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(splatAttr));
  }

  if (auto fillOp = dyn_cast_or_null<linalg::FillOp>(definingOp))
    return fillOp.getInputs()[0];
  return std::nullopt;
}

LogicalResult foldConstantOperandsIntoMap(linalg::MapOp op,
                                          PatternRewriter& rewriter) {
  auto loc = op->getLoc();
  SmallVector<Value> newInputs;
  IRMapping mapping;

  for (auto [operand, bbArg] :
       llvm::zip(op.getDpsInputOperands(), op.getBody()->getArguments())) {
    auto constantValue = getSplatValue(rewriter, loc, operand->get());
    if (constantValue.has_value()) {
      mapping.map(bbArg, *constantValue);
    } else {
      newInputs.push_back(operand->get());
    }
  }

  // No constant operands found.
  if (newInputs.size() == op.getInputs().size()) return failure();

  auto newMapOp = rewriter.create<linalg::MapOp>(loc, op.getResultTypes(),
                                                 /*inputs=*/newInputs,
                                                 /*init=*/op.getInit());
  rewriter.cloneRegionBefore(op.getRegion(), newMapOp.getRegion(),
                             newMapOp.getRegion().begin(), mapping);
  rewriter.replaceOp(op, newMapOp.getResults());

  return success();
}

// Replace linalg.map with no inputs with an linalg.fill.
LogicalResult replaceConstantMapWithFill(linalg::MapOp op,
                                         PatternRewriter& rewriter) {
  // Only replace linalg.map that has no inputs.
  if (!op.getInputs().empty()) return failure();

  // linalg.index indicates that region result is not constant.
  if (!op.getBody()->getOps<linalg::IndexOp>().empty()) return failure();

  // Move all ops outside of the region. It's safe, because this linalg.map has
  // only implicit arguments.
  for (Operation& regionOp :
       llvm::make_early_inc_range(op.getBody()->without_terminator())) {
    regionOp.moveBefore(op);
  }

  // Get fill value from gml_st.yield operand.
  auto yieldValue = op.getBody()->getTerminator()->getOperand(0);

  rewriter.replaceOpWithNewOp<linalg::FillOp>(op, yieldValue, op.getInit());
  return success();
}

// Replace linalg.broadcast(single_element_tensor) with linalg.fill.
LogicalResult replaceBroadcastWithFill(linalg::BroadcastOp op,
                                       PatternRewriter& rewriter) {
  Value input = op.getInput();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) return failure();

  Location loc = op.getLoc();
  Value scalar;
  if (auto splatValue = getSplatValue(rewriter, loc, input)) {
    scalar = *splatValue;
  } else if (hasSingleElement(inputType)) {
    SmallVector<Value> indicesInput(
        inputType.getRank(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
    scalar = rewriter.create<tensor::ExtractOp>(loc, input, indicesInput);
  }
  if (!scalar) return failure();
  rewriter.replaceOpWithNewOp<linalg::FillOp>(op, scalar, op.getInit());
  return success();
}

// Rewrite `tensor.extract_slice(op(arg1, ...))` into
// `op(tensor.extract_slice(arg1, ...))`.
LogicalResult rewriteExtractSliceOfTileableOp(Operation* op,
                                              PatternRewriter& rewriter) {
  auto tileableOp = dyn_cast<TilingInterface>(op);
  if (!tileableOp) return failure();

  // Support only ops with a single result for now.
  if (op->getNumResults() != 1) return failure();
  auto result = op->getResult(0);

  // If the op has several uses, then it is not always beneficial to rewrite.
  if (!result.hasOneUse()) return failure();
  auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(*result.getUsers().begin());
  // Check if the defining op and the slice op are located in the same block.
  // Cases when they are not are covered by fusion.
  if (!sliceOp || sliceOp->getBlock() != op->getBlock()) return failure();

  rewriter.setInsertionPointAfter(sliceOp);
  FailureOr<TilingResult> tilingResult =
      tensor::replaceExtractSliceWithTiledProducer(rewriter, sliceOp, result);

  if (failed(tilingResult)) return failure();
  rewriter.replaceOp(sliceOp, tilingResult->tiledValues);

  return success();
}

LogicalResult rewriteExtractSliceOfReverseOp(thlo::ReverseOp reverseOp,
                                             PatternRewriter& rewriter) {
  return rewriteExtractSliceOfTileableOp(reverseOp, rewriter);
}

struct RewriteExtractSliceOfLinalgOpPattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter& rewriter) const override {
    return rewriteExtractSliceOfTileableOp(linalgOp, rewriter);
  }
};

struct OptimizeLinalgOpsPass
    : public impl::OptimizeLinalgOpsPassBase<OptimizeLinalgOpsPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* ctx = &getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    patterns.add<RewriteExtractSliceOfLinalgOpPattern>(ctx);
    patterns.add(foldConstantOperandsIntoMap);
    patterns.add(replaceBroadcastWithFill);
    patterns.add(replaceConstantMapWithFill);
    patterns.add(rewriteExtractSliceOfReverseOp);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createOptimizeLinalgOpsPass() {
  return std::make_unique<gml_st::OptimizeLinalgOpsPass>();
}

}  // namespace mlir::gml_st
