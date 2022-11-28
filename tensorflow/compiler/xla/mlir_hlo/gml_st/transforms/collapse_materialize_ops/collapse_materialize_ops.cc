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

#include <iterator>
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_COLLAPSEMATERIALIZEOPSPASS
#include "gml_st/transforms/passes.h.inc"
OpFoldResult multiplyOperandsOrIntegers(PatternRewriter& rewriter, Location loc,
                                        OpFoldResult lhs, OpFoldResult rhs) {
  // Both operands are static.
  if (lhs.is<Attribute>() && rhs.is<Attribute>()) {
    return rewriter.getI64IntegerAttr(
        lhs.get<Attribute>().cast<IntegerAttr>().getInt() *
        rhs.get<Attribute>().cast<IntegerAttr>().getInt());
  }

  // Exploit commutativity and move static operand to the left (if any).
  if (rhs.is<Attribute>()) std::swap(lhs, rhs);

  // Create constant if needed.
  if (lhs.is<Attribute>()) {
    int64_t lhsInt = lhs.get<Attribute>().cast<IntegerAttr>().getInt();

    // Exploit static operand if possible.
    if (lhsInt == 0) return lhs;
    if (lhsInt == 1) return rhs;

    lhs = rewriter.create<arith::ConstantIndexOp>(loc, lhsInt).getResult();
  }

  // Multiply.
  return rewriter.create<arith::MulIOp>(loc, lhs.get<Value>(), rhs.get<Value>())
      .getResult();
}

OpFoldResult addOperandsOrIntegers(PatternRewriter& rewriter, Location loc,
                                   OpFoldResult lhs, OpFoldResult rhs) {
  // Both operands are static.
  if (lhs.is<Attribute>() && rhs.is<Attribute>()) {
    return rewriter.getI64IntegerAttr(
        lhs.get<Attribute>().cast<IntegerAttr>().getInt() +
        rhs.get<Attribute>().cast<IntegerAttr>().getInt());
  }

  // Exploit commutativity and move static operand to the left (if any).
  if (rhs.is<Attribute>()) std::swap(lhs, rhs);

  // Create constant if needed.
  if (lhs.is<Attribute>()) {
    int64_t lhsInt = lhs.get<Attribute>().cast<IntegerAttr>().getInt();

    // Exploit static operand if possible.
    if (lhsInt == 0) return rhs;

    lhs = rewriter.create<arith::ConstantIndexOp>(loc, lhsInt).getResult();
  }

  // Add.
  return rewriter.create<arith::AddIOp>(loc, lhs.get<Value>(), rhs.get<Value>())
      .getResult();
}

// Compose offsets with newOffset = supersetOffset + supersetStride * offset.
SmallVector<OpFoldResult> composeOffsets(
    const llvm::SmallVectorImpl<OpFoldResult>& supersetOffsets,
    const llvm::SmallVectorImpl<OpFoldResult>& supersetStrides,
    const llvm::SmallVectorImpl<OpFoldResult>& offsets, Location loc,
    PatternRewriter& rewriter) {
  SmallVector<OpFoldResult> composedOffsets;
  for (auto it : llvm::zip(supersetOffsets, supersetStrides, offsets)) {
    composedOffsets.push_back(addOperandsOrIntegers(
        rewriter, loc, std::get<0>(it),
        multiplyOperandsOrIntegers(rewriter, loc, std::get<1>(it),
                                   std::get<2>(it))));
  }
  return composedOffsets;
}

// Compose strides with newStride = supersetStride * stride.
SmallVector<OpFoldResult> composeStrides(
    PatternRewriter& rewriter, Location loc,
    const llvm::SmallVectorImpl<OpFoldResult>& supersetStrides,
    const llvm::SmallVectorImpl<OpFoldResult>& strides) {
  SmallVector<OpFoldResult> composedStrides;
  for (auto it : llvm::zip(supersetStrides, strides)) {
    composedStrides.push_back(multiplyOperandsOrIntegers(
        rewriter, loc, std::get<0>(it), std::get<1>(it)));
  }
  return composedStrides;
}

// Collapse materialize operations with nested tile chains t1, t2, ..., tn, and
// u1, u2, ..., un. A materialize op of the form ...
//   `materialize(materialize(tensor2, t2), t1)
// ... is collapsed as ...
//   `materialize(t2, composed_tile(t1, t2))
struct CollapseMaterializeOpPattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter& rewriter) const override {
    auto tileOp = op.getSet().getDefiningOp<TileOp>();
    if (!tileOp) return failure();

    auto producerMaterializeOp = op.getSource().getDefiningOp<MaterializeOp>();
    if (!producerMaterializeOp) return failure();

    auto producerTileOp =
        producerMaterializeOp.getSet().getDefiningOp<TileOp>();
    if (!producerTileOp) return failure();

    // Compose tileOp and producerTileOp.
    auto loc = op.getLoc();
    auto producerStrides = producerTileOp.getMixedStrides();
    auto composedOffsets =
        composeOffsets(producerTileOp.getMixedOffsets(), producerStrides,
                       tileOp.getMixedOffsets(), loc, rewriter);
    auto composedStrides = composeStrides(rewriter, loc, producerStrides,
                                          tileOp.getMixedStrides());
    auto composedTileOp = rewriter.create<TileOp>(
        loc, composedOffsets, tileOp.getMixedSizes(), composedStrides);

    rewriter.replaceOpWithNewOp<MaterializeOp>(
        op, producerMaterializeOp.getSource(), composedTileOp);
    return success();
  }
};

struct CollapseMaterializeOpsPass
    : public impl::CollapseMaterializeOpsPassBase<CollapseMaterializeOpsPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateCollapseMaterializeOpsPatterns(ctx, &patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateCollapseMaterializeOpsPatterns(MLIRContext* ctx,
                                            RewritePatternSet* patterns) {
  patterns->add<CollapseMaterializeOpPattern>(ctx);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createCollapseMaterializeOpsPass() {
  return std::make_unique<CollapseMaterializeOpsPass>();
}

}  // namespace gml_st
}  // namespace mlir
