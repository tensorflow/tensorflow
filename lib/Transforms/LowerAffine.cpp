//===- LowerAffine.cpp - Lower affine constructs to primitives ------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file lowers affine constructs (If and For statements, AffineApply
// operations) within a function into their standard If and For equivalent ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LowerAffine.h"
#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
// Visit affine expressions recursively and build the sequence of operations
// that correspond to it.  Visitation functions return an Value of the
// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value *> {
public:
  // This internal class expects arguments to be non-null, checks must be
  // performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ArrayRef<Value *> dimValues,
                      ArrayRef<Value *> symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy> Value *buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op.getResult();
  }

  Value *visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<AddIOp>(expr);
  }

  Value *visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<MulIOp>(expr);
  }

  // Euclidean modulo operation: negative RHS is not allowed.
  // Remainder of the euclidean integer division is always non-negative.
  //
  // Implemented as
  //
  //     a mod b =
  //         let remainder = srem a, b;
  //             negative = a < 0 in
  //         select negative, remainder + b, remainder.
  Value *visitModExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (modulo by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "modulo by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value *remainder = builder.create<RemISOp>(loc, lhs, rhs);
    Value *zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value *isRemainderNegative =
        builder.create<CmpIOp>(loc, CmpIPredicate::SLT, remainder, zeroCst);
    Value *correctedRemainder = builder.create<AddIOp>(loc, remainder, rhs);
    Value *result = builder.create<SelectOp>(loc, isRemainderNegative,
                                             correctedRemainder, remainder);
    return result;
  }

  // Floor division operation (rounds towards negative infinity).
  //
  // For positive divisors, it can be implemented without branching and with a
  // single division operation as
  //
  //        a floordiv b =
  //            let negative = a < 0 in
  //            let absolute = negative ? -a - 1 : a in
  //            let quotient = absolute / b in
  //                negative ? -quotient - 1 : quotient
  Value *visitFloorDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value *zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value *noneCst = builder.create<ConstantIndexOp>(loc, -1);
    Value *negative =
        builder.create<CmpIOp>(loc, CmpIPredicate::SLT, lhs, zeroCst);
    Value *negatedDecremented = builder.create<SubIOp>(loc, noneCst, lhs);
    Value *dividend =
        builder.create<SelectOp>(loc, negative, negatedDecremented, lhs);
    Value *quotient = builder.create<DivISOp>(loc, dividend, rhs);
    Value *correctedQuotient = builder.create<SubIOp>(loc, noneCst, quotient);
    Value *result =
        builder.create<SelectOp>(loc, negative, correctedQuotient, quotient);
    return result;
  }

  // Ceiling division operation (rounds towards positive infinity).
  //
  // For positive divisors, it can be implemented without branching and with a
  // single division operation as
  //
  //     a ceildiv b =
  //         let negative = a <= 0 in
  //         let absolute = negative ? -a : a - 1 in
  //         let quotient = absolute / b in
  //             negative ? -quotient : quotient + 1
  Value *visitCeilDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(loc) << "semi-affine expressions (division by non-const) are "
                        "not supported";
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value *zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value *oneCst = builder.create<ConstantIndexOp>(loc, 1);
    Value *nonPositive =
        builder.create<CmpIOp>(loc, CmpIPredicate::SLE, lhs, zeroCst);
    Value *negated = builder.create<SubIOp>(loc, zeroCst, lhs);
    Value *decremented = builder.create<SubIOp>(loc, lhs, oneCst);
    Value *dividend =
        builder.create<SelectOp>(loc, nonPositive, negated, decremented);
    Value *quotient = builder.create<DivISOp>(loc, dividend, rhs);
    Value *negatedQuotient = builder.create<SubIOp>(loc, zeroCst, quotient);
    Value *incrementedQuotient = builder.create<AddIOp>(loc, quotient, oneCst);
    Value *result = builder.create<SelectOp>(loc, nonPositive, negatedQuotient,
                                             incrementedQuotient);
    return result;
  }

  Value *visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op =
        builder.create<ConstantOp>(loc, builder.getIndexType(), valueAttr);
    return op.getResult();
  }

  Value *visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value *visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  ArrayRef<Value *> dimValues;
  ArrayRef<Value *> symbolValues;

  Location loc;
};
} // namespace

// Create a sequence of operations that implement the `expr` applied to the
// given dimension and symbol values.
mlir::Value *mlir::expandAffineExpr(OpBuilder &builder, Location loc,
                                    AffineExpr expr,
                                    ArrayRef<Value *> dimValues,
                                    ArrayRef<Value *> symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

// Create a sequence of operations that implement the `affineMap` applied to
// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value *, 8>> static expandAffineMap(
    OpBuilder &builder, Location loc, AffineMap affineMap,
    ArrayRef<Value *> operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = functional::map(
      [numDims, &builder, loc, operands](AffineExpr expr) {
        return expandAffineExpr(builder, loc, expr,
                                operands.take_front(numDims),
                                operands.drop_front(numDims));
      },
      affineMap.getResults());
  if (llvm::all_of(expanded, [](Value *v) { return v; }))
    return expanded;
  return None;
}

// Given a range of values, emit the code that reduces them with "min" or "max"
// depending on the provided comparison predicate.  The predicate defines which
// comparison to perform, "lt" for "min", "gt" for "max" and is used for the
// `cmpi` operation followed by the `select` operation:
//
//   %cond   = cmpi "predicate" %v0, %v1
//   %result = select %cond, %v0, %v1
//
// Multiple values are scanned in a linear sequence.  This creates a data
// dependences that wouldn't exist in a tree reduction, but is easier to
// recognize as a reduction by the subsequent passes.
static Value *buildMinMaxReductionSeq(Location loc, CmpIPredicate predicate,
                                      ArrayRef<Value *> values,
                                      OpBuilder &builder) {
  assert(!llvm::empty(values) && "empty min/max chain");

  auto valueIt = values.begin();
  Value *value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<SelectOp>(loc, cmpOp.getResult(), value, *valueIt);
  }

  return value;
}

// Emit instructions that correspond to the affine map in the lower bound
// applied to the respective operands, and compute the maximum value across
// the results.
Value *mlir::lowerAffineLowerBound(AffineForOp op, OpBuilder &builder) {
  SmallVector<Value *, 8> boundOperands(op.getLowerBoundOperands());
  auto lbValues = expandAffineMap(builder, op.getLoc(), op.getLowerBoundMap(),
                                  boundOperands);
  if (!lbValues)
    return nullptr;
  return buildMinMaxReductionSeq(op.getLoc(), CmpIPredicate::SGT, *lbValues,
                                 builder);
}

// Emit instructions that correspond to the affine map in the upper bound
// applied to the respective operands, and compute the minimum value across
// the results.
Value *mlir::lowerAffineUpperBound(AffineForOp op, OpBuilder &builder) {
  SmallVector<Value *, 8> boundOperands(op.getUpperBoundOperands());
  auto ubValues = expandAffineMap(builder, op.getLoc(), op.getUpperBoundMap(),
                                  boundOperands);
  if (!ubValues)
    return nullptr;
  return buildMinMaxReductionSeq(op.getLoc(), CmpIPredicate::SLT, *ubValues,
                                 builder);
}

namespace {
// Affine terminators are removed.
class AffineTerminatorLowering : public ConversionPattern {
public:
  AffineTerminatorLowering(MLIRContext *ctx)
      : ConversionPattern(AffineTerminatorOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TerminatorOp>(op);
    return matchSuccess();
  }
};

class AffineForLowering : public ConversionPattern {
public:
  AffineForLowering(MLIRContext *ctx)
      : ConversionPattern(AffineForOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto affineForOp = cast<AffineForOp>(op);
    Location loc = op->getLoc();
    Value *lowerBound = lowerAffineLowerBound(affineForOp, rewriter);
    Value *upperBound = lowerAffineUpperBound(affineForOp, rewriter);
    Value *step = rewriter.create<ConstantIndexOp>(loc, affineForOp.getStep());
    auto f = rewriter.create<ForOp>(loc, lowerBound, upperBound, step);
    f.region().getBlocks().clear();
    rewriter.inlineRegionBefore(affineForOp.getRegion(), f.region(),
                                f.region().end());
    rewriter.replaceOp(op, {});
    return matchSuccess();
  }
};

class AffineIfLowering : public ConversionPattern {
public:
  AffineIfLowering(MLIRContext *ctx)
      : ConversionPattern(AffineIfOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto affineIfOp = cast<AffineIfOp>(op);
    auto loc = op->getLoc();

    // Now we just have to handle the condition logic.
    auto integerSet = affineIfOp.getIntegerSet();
    Value *zeroConstant = rewriter.create<ConstantIndexOp>(loc, 0);

    // Calculate cond as a conjunction without short-circuiting.
    Value *cond = nullptr;
    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value *affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                          operands.take_front(numDims),
                                          operands.drop_front(numDims));
      if (!affResult)
        return matchFailure();
      auto pred = isEquality ? CmpIPredicate::EQ : CmpIPredicate::SGE;
      Value *cmpVal =
          rewriter.create<CmpIOp>(loc, pred, affResult, zeroConstant);
      cond =
          cond ? rewriter.create<AndOp>(loc, cond, cmpVal).getResult() : cmpVal;
    }
    cond = cond ? cond
                : rewriter.create<ConstantIntOp>(loc, /*value=*/1, /*width=*/1);

    bool hasElseRegion = !affineIfOp.getElseBlocks().empty();
    auto ifOp = rewriter.create<IfOp>(loc, cond, hasElseRegion);
    rewriter.inlineRegionBefore(affineIfOp.getThenBlocks(),
                                &ifOp.thenRegion().back());
    ifOp.thenRegion().back().erase();
    if (hasElseRegion) {
      rewriter.inlineRegionBefore(affineIfOp.getElseBlocks(),
                                  &ifOp.elseRegion().back());
      ifOp.elseRegion().back().erase();
    }

    // Ok, we're done!
    rewriter.replaceOp(op, {});
    return matchSuccess();
  }
};

// Convert an "affine.apply" operation into a sequence of arithmetic
// operations using the StandardOps dialect.
class AffineApplyLowering : public ConversionPattern {
public:
  AffineApplyLowering(MLIRContext *ctx)
      : ConversionPattern(AffineApplyOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto affineApplyOp = cast<AffineApplyOp>(op);
    auto maybeExpandedMap = expandAffineMap(
        rewriter, op->getLoc(), affineApplyOp.getAffineMap(), operands);
    if (!maybeExpandedMap)
      return matchFailure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return matchSuccess();
  }
};

// Apply the affine map from an 'affine.load' operation to its operands, and
// feed the results to a newly created 'std.load' operation (which replaces the
// original 'affine.load').
class AffineLoadLowering : public ConversionPattern {
public:
  AffineLoadLowering(MLIRContext *ctx)
      : ConversionPattern(AffineLoadOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto affineLoadOp = cast<AffineLoadOp>(op);
    // Expand affine map from 'affineLoadOp'.
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op->getLoc(), affineLoadOp.getAffineMap(),
                        operands.drop_front());
    if (!maybeExpandedMap)
      return matchFailure();
    // Build std.load memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<LoadOp>(op, operands[0], *maybeExpandedMap);
    return matchSuccess();
  }
};

// Apply the affine map from an 'affine.store' operation to its operands, and
// feed the results to a newly created 'std.store' operation (which replaces the
// original 'affine.store').
class AffineStoreLowering : public ConversionPattern {
public:
  AffineStoreLowering(MLIRContext *ctx)
      : ConversionPattern(AffineStoreOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto affineStoreOp = cast<AffineStoreOp>(op);
    // Expand affine map from 'affineStoreOp'.
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op->getLoc(), affineStoreOp.getAffineMap(),
                        operands.drop_front(2));
    if (!maybeExpandedMap)
      return matchFailure();
    // Build std.store valutToStore, memref[expandedMap.results].
    rewriter.replaceOpWithNewOp<StoreOp>(op, operands[0], operands[1],
                                         *maybeExpandedMap);
    return matchSuccess();
  }
};

// Apply the affine maps from an 'affine.dma_start' operation to each of their
// respective map operands, and feed the results to a newly created
// 'std.dma_start' operation (which replaces the original 'affine.dma_start').
class AffineDmaStartLowering : public ConversionPattern {
public:
  AffineDmaStartLowering(MLIRContext *ctx)
      : ConversionPattern(AffineDmaStartOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto affineDmaStartOp = cast<AffineDmaStartOp>(op);
    // Expand affine map for DMA source memref.
    auto maybeExpandedSrcMap = expandAffineMap(
        rewriter, op->getLoc(), affineDmaStartOp.getSrcMap(),
        operands.drop_front(affineDmaStartOp.getSrcMemRefOperandIndex() + 1));
    if (!maybeExpandedSrcMap)
      return matchFailure();
    // Expand affine map for DMA destination memref.
    auto maybeExpandedDstMap = expandAffineMap(
        rewriter, op->getLoc(), affineDmaStartOp.getDstMap(),
        operands.drop_front(affineDmaStartOp.getDstMemRefOperandIndex() + 1));
    if (!maybeExpandedDstMap)
      return matchFailure();
    // Expand affine map for DMA tag memref.
    auto maybeExpandedTagMap = expandAffineMap(
        rewriter, op->getLoc(), affineDmaStartOp.getTagMap(),
        operands.drop_front(affineDmaStartOp.getTagMemRefOperandIndex() + 1));
    if (!maybeExpandedTagMap)
      return matchFailure();

    // Build std.dma_start operation with affine map results.
    auto *srcMemRef = operands[affineDmaStartOp.getSrcMemRefOperandIndex()];
    auto *dstMemRef = operands[affineDmaStartOp.getDstMemRefOperandIndex()];
    auto *tagMemRef = operands[affineDmaStartOp.getTagMemRefOperandIndex()];
    unsigned numElementsIndex = affineDmaStartOp.getTagMemRefOperandIndex() +
                                1 + affineDmaStartOp.getTagMap().getNumInputs();
    auto *numElements = operands[numElementsIndex];
    auto *stride =
        affineDmaStartOp.isStrided() ? operands[numElementsIndex + 1] : nullptr;
    auto *eltsPerStride =
        affineDmaStartOp.isStrided() ? operands[numElementsIndex + 2] : nullptr;

    rewriter.replaceOpWithNewOp<DmaStartOp>(
        op, srcMemRef, *maybeExpandedSrcMap, dstMemRef, *maybeExpandedDstMap,
        numElements, tagMemRef, *maybeExpandedTagMap, stride, eltsPerStride);
    return matchSuccess();
  }
};

// Apply the affine map from an 'affine.dma_wait' operation tag memref,
// and feed the results to a newly created 'std.dma_wait' operation (which
// replaces the original 'affine.dma_wait').
class AffineDmaWaitLowering : public ConversionPattern {
public:
  AffineDmaWaitLowering(MLIRContext *ctx)
      : ConversionPattern(AffineDmaWaitOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto affineDmaWaitOp = cast<AffineDmaWaitOp>(op);
    // Expand affine map for DMA tag memref.
    auto maybeExpandedTagMap =
        expandAffineMap(rewriter, op->getLoc(), affineDmaWaitOp.getTagMap(),
                        operands.drop_front());
    if (!maybeExpandedTagMap)
      return matchFailure();

    // Build std.dma_wait operation with affine map results.
    unsigned numElementsIndex = 1 + affineDmaWaitOp.getTagMap().getNumInputs();
    rewriter.replaceOpWithNewOp<DmaWaitOp>(
        op, operands[0], *maybeExpandedTagMap, operands[numElementsIndex]);
    return matchSuccess();
  }
};

} // end namespace

LogicalResult mlir::lowerAffineConstructs(FuncOp function) {
  OwningRewritePatternList patterns;
  RewriteListBuilder<AffineApplyLowering, AffineDmaStartLowering,
                     AffineDmaWaitLowering, AffineLoadLowering,
                     AffineStoreLowering, AffineForLowering, AffineIfLowering,
                     AffineTerminatorLowering>::build(patterns,
                                                      function.getContext());
  ConversionTarget target(*function.getContext());
  target.addLegalDialect<StandardOpsDialect>();
  return applyConversionPatterns(function, target, std::move(patterns));
}

namespace {
class LowerAffinePass : public FunctionPass<LowerAffinePass> {
  void runOnFunction() override { lowerAffineConstructs(getFunction()); }
};
} // namespace

/// Lowers If and For operations within a function into their lower level CFG
/// equivalent blocks.
FunctionPassBase *mlir::createLowerAffinePass() {
  return new LowerAffinePass();
}

static PassRegistration<LowerAffinePass>
    pass("lower-affine",
         "Lower If, For, AffineApply operations to primitive equivalents");
