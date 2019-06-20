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
// operations) within a function into their lower level CFG equivalent blocks.
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
      builder.getContext()->emitError(
          loc,
          "semi-affine expressions (modulo by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      builder.getContext()->emitError(
          loc, "modulo by non-positive value is not supported");
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
      builder.getContext()->emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      builder.getContext()->emitError(
          loc, "division by non-positive value is not supported");
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
      builder.getContext()->emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      builder.getContext()->emitError(
          loc, "division by non-positive value is not supported");
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
    rewriter.replaceOp(op, {});
    return matchSuccess();
  }
};

// Create a CFG subgraph for the loop around its body blocks (if the body
// contained other loops, they have been already lowered to a flow of blocks).
// Maintain the invariants that a CFG subgraph created for any loop has a single
// entry and a single exit, and that the entry/exit blocks are respectively
// first/last blocks in the parent region.  The original loop operation is
// replaced by the initialization operations that set up the initial value of
// the loop induction variable (%iv) and computes the loop bounds that are loop-
// invariant for affine loops.  The operations following the original affine.for
// are split out into a separate continuation (exit) block. A condition block is
// created before the continuation block. It checks the exit condition of the
// loop and branches either to the continuation block, or to the first block of
// the body. Induction variable modification is appended to the last block of
// the body (which is the exit block from the body subgraph thanks to the
// invariant we maintain) along with a branch that loops back to the condition
// block.
//
// NOTE: this relies on the DialectConversion infrastructure knowing how to undo
// the creation of operations if the conversion fails.  In particular, lowering
// of the affine maps may insert operations and then fail on a semi-affine map.
//
//      +---------------------------------+
//      |   <code before the AffineForOp> |
//      |   <compute initial %iv value>   |
//      |   br cond(%iv)                  |
//      +---------------------------------+
//             |
//  -------|   |
//  |      v   v
//  |   +--------------------------------+
//  |   | cond(%iv):                     |
//  |   |   <compare %iv to upper bound> |
//  |   |   cond_br %r, body, end        |
//  |   +--------------------------------+
//  |          |               |
//  |          |               -------------|
//  |          v                            |
//  |   +--------------------------------+  |
//  |   | body-first:                    |  |
//  |   |   <body contents>              |  |
//  |   +--------------------------------+  |
//  |                   |                   |
//  |                  ...                  |
//  |                   |                   |
//  |   +--------------------------------+  |
//  |   | body-last:                     |  |
//  |   |   <body contents>              |  |
//  |   |   %new_iv =<add step to %iv>   |  |
//  |   |   br cond(%new_iv)             |  |
//  |   +--------------------------------+  |
//  |          |                            |
//  |-----------        |--------------------
//                      v
//      +--------------------------------+
//      | end:                           |
//      |   <code after the AffineForOp> |
//      +--------------------------------+
//
class AffineForLowering : public ConversionPattern {
public:
  AffineForLowering(MLIRContext *ctx)
      : ConversionPattern(AffineForOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto forOp = cast<AffineForOp>(op);
    Location loc = op->getLoc();

    // Start by splitting the block containing the 'affine.for' into two parts.
    // The part before will get the init code, the part after will be the end
    // point.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

    // Use the first block of the loop body as the condition block since it is
    // the block that has the induction variable as its argument.  Split out
    // all operations from the first block into a new block.  Move all body
    // blocks from the loop body region to the region containing the loop.
    auto *conditionBlock = &forOp.getRegion().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock = &forOp.getRegion().back();
    rewriter.inlineRegionBefore(forOp.getRegion(), Region::iterator(endBlock));
    auto *iv = conditionBlock->getArgument(0);

    // Append the induction variable stepping logic to the last body block and
    // branch back to the condition block.  Construct an affine expression f :
    // (x -> x+step) and apply this expression to the induction variable.
    rewriter.setInsertionPointToEnd(lastBodyBlock);
    auto affStep = rewriter.getAffineConstantExpr(forOp.getStep());
    auto affDim = rewriter.getAffineDimExpr(0);
    auto stepped = expandAffineExpr(rewriter, loc, affDim + affStep, iv, {});
    if (!stepped)
      return matchFailure();
    rewriter.create<BranchOp>(loc, conditionBlock, stepped);

    // Compute loop bounds before branching to the condition.
    rewriter.setInsertionPointToEnd(initBlock);
    Value *lowerBound = lowerAffineLowerBound(forOp, rewriter);
    Value *upperBound = lowerAffineUpperBound(forOp, rewriter);
    if (!lowerBound || !upperBound)
      return matchFailure();
    rewriter.create<BranchOp>(loc, conditionBlock, lowerBound);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comparison =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::SLT, iv, upperBound);
    rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock,
                                  ArrayRef<Value *>(), endBlock,
                                  ArrayRef<Value *>());
    // Ok, we're done!
    rewriter.replaceOp(op, {});
    return matchSuccess();
  }
};

// Create a CFG subgraph for the affine.if operation (including its "then" and
// optional "else" operation blocks).  We maintain the invariants that the
// subgraph has a single entry and a single exit point, and that the entry/exit
// blocks are respectively the first/last block of the enclosing region. The
// operations following the affine.if are split into a continuation (subgraph
// exit) block. The condition is lowered to a chain of blocks that implement the
// short-circuit scheme.  Condition blocks are created by splitting out an empty
// block from the block that contains the affine.if operation.  They
// conditionally branch to either the first block of the "then" region, or to
// the first block of the "else" region.  If the latter is absent, they branch
// to the continuation block instead.  The last blocks of "then" and "else"
// regions (which are known to be exit blocks thanks to the invariant we
// maintain).
//
// NOTE: this relies on the DialectConversion infrastructure knowing how to undo
// the creation of operations if the conversion fails.  In particular, lowering
// of the affine maps may insert operations and then fail on a semi-affine map.
//
//      +--------------------------------+
//      | <code before the AffineIfOp>   |
//      | %zero = constant 0 : index     |
//      | %v = affine.apply #expr1(%ops) |
//      | %c = cmpi "sge" %v, %zero      |
//      | cond_br %c, %next, %else       |
//      +--------------------------------+
//             |              |
//             |              --------------|
//             v                            |
//      +--------------------------------+  |
//      | next:                          |  |
//      |   <repeat the check for expr2> |  |
//      |   cond_br %c, %next2, %else    |  |
//      +--------------------------------+  |
//             |              |             |
//            ...             --------------|
//             |   <Per-expression checks>  |
//             v                            |
//      +--------------------------------+  |
//      | last:                          |  |
//      |   <repeat the check for exprN> |  |
//      |   cond_br %c, %then, %else     |  |
//      +--------------------------------+  |
//             |              |             |
//             |              --------------|
//             v                            |
//      +--------------------------------+  |
//      | then:                          |  |
//      |   <then contents>              |  |
//      |   br continue                  |  |
//      +--------------------------------+  |
//             |                            |
//   |----------               |-------------
//   |                         V
//   |  +--------------------------------+
//   |  | else:                          |
//   |  |   <else contents>              |
//   |  |   br continue                  |
//   |  +--------------------------------+
//   |         |
//   ------|   |
//         v   v
//      +--------------------------------+
//      | continue:                      |
//      |   <code after the AffineIfOp>  |
//      +--------------------------------+
//
class AffineIfLowering : public ConversionPattern {
public:
  AffineIfLowering(MLIRContext *ctx)
      : ConversionPattern(AffineIfOp::getOperationName(), 1, ctx) {}

  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  PatternRewriter &rewriter) const override {
    auto ifOp = cast<AffineIfOp>(op);
    auto loc = op->getLoc();

    // Start by splitting the block containing the 'affine.if' into two parts.
    // The part before will contain the condition, the part after will be the
    // continuation point.
    auto *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *continueBlock = rewriter.splitBlock(condBlock, opPosition);

    // Move blocks from the "then" region to the region containing 'affine.if',
    // place it before the continuation block, and branch to it.
    auto *thenBlock = &ifOp.getThenBlocks().front();
    rewriter.setInsertionPointToEnd(&ifOp.getThenBlocks().back());
    rewriter.create<BranchOp>(loc, continueBlock);
    rewriter.inlineRegionBefore(ifOp.getThenBlocks(),
                                Region::iterator(continueBlock));

    // Move blocks from the "else" region (if present) to the region containing
    // 'affine.if', place it before the continuation block and branch to it.  It
    // will be placed after the "then" regions.
    auto *elseBlock = continueBlock;
    if (!ifOp.getElseBlocks().empty()) {
      elseBlock = &ifOp.getElseBlocks().front();
      rewriter.setInsertionPointToEnd(&ifOp.getElseBlocks().back());
      rewriter.create<BranchOp>(loc, continueBlock);
      rewriter.inlineRegionBefore(ifOp.getElseBlocks(),
                                  Region::iterator(continueBlock));
    }

    // Now we just have to handle the condition logic.
    auto integerSet = ifOp.getIntegerSet();

    // Implement short-circuit logic.  For each affine expression in the
    // 'affine.if' condition, convert it into an affine map and call
    // `affine.apply` to obtain the resulting value.  Perform the equality or
    // the greater-than-or-equality test between this value and zero depending
    // on the equality flag of the condition.  If the test fails, jump
    // immediately to the false branch, which may be the else block if it is
    // present or the continuation block otherwise. If the test succeeds, jump
    // to the next block testing the next conjunct of the condition in the
    // similar way.  When all conjuncts have been handled, jump to the 'then'
    // block instead.
    rewriter.setInsertionPointToEnd(condBlock);
    Value *zeroConstant = rewriter.create<ConstantIndexOp>(loc, 0);

    for (unsigned i = 0, e = integerSet.getNumConstraints(); i < e; ++i) {
      AffineExpr constraintExpr = integerSet.getConstraint(i);
      bool isEquality = integerSet.isEq(i);

      // Create the fall-through block for the next condition, if present, by
      // splitting an empty block out of an existing block.  Otherwise treat the
      // first "then" block as the block we should branch to if the (last)
      // condition is true.
      auto *nextBlock = (i == e - 1)
                            ? thenBlock
                            : rewriter.splitBlock(condBlock, condBlock->end());

      // Build and apply an affine expression
      auto numDims = integerSet.getNumDims();
      Value *affResult = expandAffineExpr(rewriter, loc, constraintExpr,
                                          operands.take_front(numDims),
                                          operands.drop_front(numDims));
      if (!affResult)
        return matchFailure();
      auto comparisonOp = rewriter.create<CmpIOp>(
          loc, isEquality ? CmpIPredicate::EQ : CmpIPredicate::SGE, affResult,
          zeroConstant);
      rewriter.create<CondBranchOp>(loc, comparisonOp.getResult(), nextBlock,
                                    /*trueArgs=*/ArrayRef<Value *>(), elseBlock,
                                    /*falseArgs=*/ArrayRef<Value *>());
      rewriter.setInsertionPointToEnd(nextBlock);
      condBlock = nextBlock;
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
} // end namespace

LogicalResult mlir::lowerAffineConstructs(Function &function) {
  OwningRewritePatternList patterns;
  RewriteListBuilder<AffineApplyLowering, AffineForLowering, AffineIfLowering,
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
