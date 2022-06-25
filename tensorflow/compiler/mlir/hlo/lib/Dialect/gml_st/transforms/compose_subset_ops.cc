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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

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

// Compose offsets with newOffset = argOffset + argStride * offset.
SmallVector<OpFoldResult> composeOffsets(
    const llvm::SmallVectorImpl<OpFoldResult>& argOffsets,
    const llvm::SmallVectorImpl<OpFoldResult>& argStrides,
    const llvm::SmallVectorImpl<OpFoldResult>& offsets, Location loc,
    PatternRewriter& rewriter) {
  SmallVector<OpFoldResult> composedOffsets;
  for (auto it : llvm::zip(argOffsets, argStrides, offsets)) {
    composedOffsets.push_back(addOperandsOrIntegers(
        rewriter, loc, std::get<0>(it),
        multiplyOperandsOrIntegers(rewriter, loc, std::get<1>(it),
                                   std::get<2>(it))));
  }
  return composedOffsets;
}

// Compose strides with newStride = argStride * stride.
SmallVector<OpFoldResult> composeStrides(
    PatternRewriter& rewriter, Location loc,
    const llvm::SmallVectorImpl<OpFoldResult>& argStrides,
    const llvm::SmallVectorImpl<OpFoldResult>& strides) {
  SmallVector<OpFoldResult> composedStrides;
  for (auto it : llvm::zip(argStrides, strides)) {
    composedStrides.push_back(multiplyOperandsOrIntegers(
        rewriter, loc, std::get<0>(it), std::get<1>(it)));
  }
  return composedStrides;
}

struct ComposeTilesPattern : public OpRewritePattern<TileOp> {
  using OpRewritePattern<TileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileOp op,
                                PatternRewriter& rewriter) const override {
    auto argOp = llvm::dyn_cast_or_null<TileOp>(op.subset().getDefiningOp());
    if (!argOp) return failure();

    // Compose offsets with newOffset = argOffset + argStride * offset.
    auto loc = op.getLoc();
    auto composedOffsets = decomposeMixedStridesOrOffsets(
        rewriter,
        composeOffsets(argOp.getMixedOffsets(), argOp.getMixedStrides(),
                       op.getMixedOffsets(), loc, rewriter));

    // Compose strides with newStride = argStride * stride.
    auto composedStrides = decomposeMixedStridesOrOffsets(
        rewriter, composeStrides(rewriter, loc, argOp.getMixedStrides(),
                                 op.getMixedStrides()));

    // Build the composed tile op.
    rewriter.replaceOpWithNewOp<TileOp>(
        op, argOp.subset(), composedOffsets.second, op.sizes(),
        composedStrides.second, composedOffsets.first, op.static_sizes(),
        composedStrides.first);
    return success();
  }
};

class ComposeSubsetOpsPass
    : public ComposeSubsetOpsPassBase<ComposeSubsetOpsPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithmeticDialect, GmlStDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ComposeTilesPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createComposeSubsetOpsPass() {
  return std::make_unique<ComposeSubsetOpsPass>();
}

}  // namespace gml_st
}  // namespace mlir
