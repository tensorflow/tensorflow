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

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/Passes.h"
using namespace mlir;

namespace {
// Visit affine expressions recursively and build the sequence of instructions
// that correspond to it.  Visitation functions return an Value of the
// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value *> {
public:
  // This internal class expects arguments to be non-null, checks must be
  // performed at the call site.
  AffineApplyExpander(FuncBuilder *builder, ArrayRef<Value *> dimValues,
                      ArrayRef<Value *> symbolValues, Location loc)
      : builder(*builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy> Value *buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op->getResult();
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
  // single division instruction as
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
  // single division instruction as
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
    return op->getResult();
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
  FuncBuilder &builder;
  ArrayRef<Value *> dimValues;
  ArrayRef<Value *> symbolValues;

  Location loc;
};
} // namespace

// Create a sequence of instructions that implement the `expr` applied to the
// given dimension and symbol values.
static mlir::Value *expandAffineExpr(FuncBuilder *builder, Location loc,
                                     AffineExpr expr,
                                     ArrayRef<Value *> dimValues,
                                     ArrayRef<Value *> symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

// Create a sequence of instructions that implement the `affineMap` applied to
// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value *, 8>> static expandAffineMap(
    FuncBuilder *builder, Location loc, AffineMap affineMap,
    ArrayRef<Value *> operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = functional::map(
      [numDims, builder, loc, operands](AffineExpr expr) {
        return expandAffineExpr(builder, loc, expr,
                                operands.take_front(numDims),
                                operands.drop_front(numDims));
      },
      affineMap.getResults());
  if (llvm::all_of(expanded, [](Value *v) { return v; }))
    return expanded;
  return None;
}

namespace {
class LowerAffinePass : public FunctionPass {
public:
  LowerAffinePass() : FunctionPass(&passID) {}
  PassResult runOnFunction(Function *function) override;

  bool lowerAffineFor(OpPointer<AffineForOp> forOp);
  bool lowerAffineIf(AffineIfOp *ifOp);
  bool lowerAffineApply(AffineApplyOp *op);

  static char passID;
};
} // end anonymous namespace

char LowerAffinePass::passID = 0;

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
                                      FuncBuilder &builder) {
  assert(!llvm::empty(values) && "empty min/max chain");

  auto valueIt = values.begin();
  Value *value = *valueIt++;
  for (; valueIt != values.end(); ++valueIt) {
    auto cmpOp = builder.create<CmpIOp>(loc, predicate, value, *valueIt);
    value = builder.create<SelectOp>(loc, cmpOp->getResult(), value, *valueIt);
  }

  return value;
}

// Convert a "for" loop to a flow of blocks.  Return `false` on success.
//
// Create an SESE region for the loop (including its body) and append it to the
// end of the current region.  The loop region consists of the initialization
// block that sets up the initial value of the loop induction variable (%iv) and
// computes the loop bounds that are loop-invariant in functions; the condition
// block that checks the exit condition of the loop; the body SESE region; and
// the end block that post-dominates the loop.  The end block of the loop
// becomes the new end of the current SESE region.  The body of the loop is
// constructed recursively after starting a new region (it may be, for example,
// a nested loop).  Induction variable modification is appended to the body SESE
// region that always loops back to the condition block.
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
//  |   | body:                          |  |
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
bool LowerAffinePass::lowerAffineFor(OpPointer<AffineForOp> forOp) {
  auto loc = forOp->getLoc();
  auto *forInst = forOp->getInstruction();

  // Start by splitting the block containing the 'for' into two parts.  The part
  // before will get the init code, the part after will be the end point.
  auto *initBlock = forInst->getBlock();
  auto *endBlock = initBlock->splitBlock(forInst);

  // Create the condition block, with its argument for the loop induction
  // variable.  We set it up below.
  auto *conditionBlock = new Block();
  conditionBlock->insertBefore(endBlock);
  auto *iv = conditionBlock->addArgument(IndexType::get(forInst->getContext()));

  // Create the body block, moving the body of the forOp over to it.
  auto *bodyBlock = new Block();
  bodyBlock->insertBefore(endBlock);

  auto *oldBody = forOp->getBody();
  bodyBlock->getInstructions().splice(bodyBlock->begin(),
                                      oldBody->getInstructions(),
                                      oldBody->begin(), oldBody->end());

  // The code in the body of the forOp now uses 'iv' as its indvar.
  forOp->getInductionVar()->replaceAllUsesWith(iv);

  // Append the induction variable stepping logic and branch back to the exit
  // condition block.  Construct an affine expression f : (x -> x+step) and
  // apply this expression to the induction variable.
  FuncBuilder builder(bodyBlock);
  auto affStep = builder.getAffineConstantExpr(forOp->getStep());
  auto affDim = builder.getAffineDimExpr(0);
  auto stepped = expandAffineExpr(&builder, loc, affDim + affStep, iv, {});
  if (!stepped)
    return true;
  // We know we applied a one-dimensional map.
  builder.create<BranchOp>(loc, conditionBlock, stepped);

  // Now that the body block done, fill in the code to compute the bounds of the
  // induction variable in the init block.
  builder.setInsertionPointToEnd(initBlock);

  // Compute loop bounds.
  SmallVector<Value *, 8> operands(forOp->getLowerBoundOperands());
  auto lbValues = expandAffineMap(&builder, forInst->getLoc(),
                                  forOp->getLowerBoundMap(), operands);
  if (!lbValues)
    return true;
  Value *lowerBound =
      buildMinMaxReductionSeq(loc, CmpIPredicate::SGT, *lbValues, builder);

  operands.assign(forOp->getUpperBoundOperands().begin(),
                  forOp->getUpperBoundOperands().end());
  auto ubValues = expandAffineMap(&builder, forInst->getLoc(),
                                  forOp->getUpperBoundMap(), operands);
  if (!ubValues)
    return true;
  Value *upperBound =
      buildMinMaxReductionSeq(loc, CmpIPredicate::SLT, *ubValues, builder);
  builder.create<BranchOp>(loc, conditionBlock, lowerBound);

  // With the body block done, we can fill in the condition block.
  builder.setInsertionPointToEnd(conditionBlock);
  auto comparison =
      builder.create<CmpIOp>(loc, CmpIPredicate::SLT, iv, upperBound);
  builder.create<CondBranchOp>(loc, comparison, bodyBlock, ArrayRef<Value *>(),
                               endBlock, ArrayRef<Value *>());

  // Ok, we're done!
  forOp->erase();
  return false;
}

// Convert an "if" instruction into a flow of basic blocks.
//
// Create an SESE region for the if instruction (including its "then" and
// optional "else" instruction blocks) and append it to the end of the current
// region.  The conditional region consists of a sequence of condition-checking
// blocks that implement the short-circuit scheme, followed by a "then" SESE
// region and an "else" SESE region, and the continuation block that
// post-dominates all blocks of the "if" instruction.  The flow of blocks that
// correspond to the "then" and "else" clauses are constructed recursively,
// enabling easy nesting of "if" instructions and if-then-else-if chains.
//
//      +--------------------------------+
//      | <code before the AffineIfOp>       |
//      | %zero = constant 0 : index     |
//      | %v = affine_apply #expr1(%ops) |
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
//      |   <code after the AffineIfOp>      |
//      +--------------------------------+
//
bool LowerAffinePass::lowerAffineIf(AffineIfOp *ifOp) {
  auto *ifInst = ifOp->getInstruction();
  auto loc = ifInst->getLoc();

  // Start by splitting the block containing the 'if' into two parts.  The part
  // before will contain the condition, the part after will be the continuation
  // point.
  auto *condBlock = ifInst->getBlock();
  auto *continueBlock = condBlock->splitBlock(ifInst);

  // Create a block for the 'then' code, inserting it between the cond and
  // continue blocks.  Move the instructions over from the AffineIfOp and add a
  // branch to the continuation point.
  Block *thenBlock = new Block();
  thenBlock->insertBefore(continueBlock);

  // If the 'then' block is not empty, then splice the instructions.
  auto &oldThenBlocks = ifOp->getThenBlocks();
  if (!oldThenBlocks.empty()) {
    // We currently only handle one 'then' block.
    if (std::next(oldThenBlocks.begin()) != oldThenBlocks.end())
      return true;

    Block *oldThen = &oldThenBlocks.front();

    thenBlock->getInstructions().splice(thenBlock->begin(),
                                        oldThen->getInstructions(),
                                        oldThen->begin(), oldThen->end());
  }

  FuncBuilder builder(thenBlock);
  builder.create<BranchOp>(loc, continueBlock);

  // Handle the 'else' block the same way, but we skip it if we have no else
  // code.
  Block *elseBlock = continueBlock;
  auto &oldElseBlocks = ifOp->getElseBlocks();
  if (!oldElseBlocks.empty()) {
    // We currently only handle one 'else' block.
    if (std::next(oldElseBlocks.begin()) != oldElseBlocks.end())
      return true;

    auto *oldElse = &oldElseBlocks.front();
    elseBlock = new Block();
    elseBlock->insertBefore(continueBlock);

    elseBlock->getInstructions().splice(elseBlock->begin(),
                                        oldElse->getInstructions(),
                                        oldElse->begin(), oldElse->end());
    builder.setInsertionPointToEnd(elseBlock);
    builder.create<BranchOp>(loc, continueBlock);
  }

  // Ok, now we just have to handle the condition logic.
  auto integerSet = ifOp->getIntegerSet();

  // Implement short-circuit logic.  For each affine expression in the 'if'
  // condition, convert it into an affine map and call `affine_apply` to obtain
  // the resulting value.  Perform the equality or the greater-than-or-equality
  // test between this value and zero depending on the equality flag of the
  // condition.  If the test fails, jump immediately to the false branch, which
  // may be the else block if it is present or the continuation block otherwise.
  // If the test succeeds, jump to the next block testing the next conjunct of
  // the condition in the similar way.  When all conjuncts have been handled,
  // jump to the 'then' block instead.
  builder.setInsertionPointToEnd(condBlock);
  Value *zeroConstant = builder.create<ConstantIndexOp>(loc, 0);

  for (auto tuple :
       llvm::zip(integerSet.getConstraints(), integerSet.getEqFlags())) {
    AffineExpr constraintExpr = std::get<0>(tuple);
    bool isEquality = std::get<1>(tuple);

    // Create the fall-through block for the next condition right before the
    // 'thenBlock'.
    auto *nextBlock = new Block();
    nextBlock->insertBefore(thenBlock);

    // Build and apply an affine expression
    SmallVector<Value *, 8> operands(ifInst->getOperands());
    auto operandsRef = ArrayRef<Value *>(operands);
    auto numDims = integerSet.getNumDims();
    Value *affResult = expandAffineExpr(&builder, loc, constraintExpr,
                                        operandsRef.take_front(numDims),
                                        operandsRef.drop_front(numDims));
    if (!affResult)
      return true;

    // Compare the result of the apply and branch.
    auto comparisonOp = builder.create<CmpIOp>(
        loc, isEquality ? CmpIPredicate::EQ : CmpIPredicate::SGE, affResult,
        zeroConstant);
    builder.create<CondBranchOp>(loc, comparisonOp->getResult(), nextBlock,
                                 /*trueArgs*/ ArrayRef<Value *>(), elseBlock,
                                 /*falseArgs*/ ArrayRef<Value *>());
    builder.setInsertionPointToEnd(nextBlock);
  }

  // We will have ended up with an empty block as our continuation block (or, in
  // the degenerate case where there were zero conditions, we have the original
  // condition block).  Redirect that to the thenBlock.
  condBlock = builder.getInsertionBlock();
  if (condBlock->empty()) {
    condBlock->replaceAllUsesWith(thenBlock);
    condBlock->eraseFromFunction();
  } else {
    builder.create<BranchOp>(loc, thenBlock);
  }

  // Ok, we're done!
  ifInst->erase();
  return false;
}

// Convert an "affine_apply" operation into a sequence of arithmetic
// instructions using the StandardOps dialect.  Return true on error.
bool LowerAffinePass::lowerAffineApply(AffineApplyOp *op) {
  FuncBuilder builder(op->getInstruction());
  auto maybeExpandedMap =
      expandAffineMap(&builder, op->getLoc(), op->getAffineMap(),
                      llvm::to_vector<8>(op->getOperands()));
  if (!maybeExpandedMap)
    return true;

  Value *original = op->getResult();
  Value *expanded = (*maybeExpandedMap)[0];
  if (!expanded)
    return true;
  original->replaceAllUsesWith(expanded);
  op->erase();
  return false;
}

// Entry point of the function convertor.
//
// Conversion is performed by recursively visiting instructions of a Function.
// It reasons in terms of single-entry single-exit (SESE) regions that are not
// materialized in the code.  Instead, the pointer to the last block of the
// region is maintained throughout the conversion as the insertion point of the
// IR builder since we never change the first block after its creation.  "Block"
// instructions such as loops and branches create new SESE regions for their
// bodies, and surround them with additional basic blocks for the control flow.
// Individual operations are simply appended to the end of the last basic block
// of the current region.  The SESE invariant allows us to easily handle nested
// structures of arbitrary complexity.
//
// During the conversion, we maintain a mapping between the Values present in
// the original function and their Value images in the function under
// construction.  When an Value is used, it gets replaced with the
// corresponding Value that has been defined previously.  The value flow
// starts with function arguments converted to basic block arguments.
PassResult LowerAffinePass::runOnFunction(Function *function) {
  SmallVector<Instruction *, 8> instsToRewrite;

  // Collect all the For instructions as well as AffineIfOps and AffineApplyOps.
  // We do this as a prepass to avoid invalidating the walker with our rewrite.
  function->walkInsts([&](Instruction *inst) {
    auto op = cast<OperationInst>(inst);
    if (op->isa<AffineApplyOp>() || op->isa<AffineForOp>() ||
        op->isa<AffineIfOp>())
      instsToRewrite.push_back(inst);
  });

  // Rewrite all of the ifs and fors.  We walked the instructions in preorder,
  // so we know that we will rewrite them in the same order.
  for (auto *inst : instsToRewrite) {
    auto op = cast<OperationInst>(inst);
    if (auto ifOp = op->dyn_cast<AffineIfOp>()) {
      if (lowerAffineIf(ifOp))
        return failure();
    } else if (auto forOp = op->dyn_cast<AffineForOp>()) {
      if (lowerAffineFor(forOp))
        return failure();
    } else if (lowerAffineApply(op->cast<AffineApplyOp>())) {
      return failure();
    }
  }

  return success();
}

/// Lowers If and For instructions within a function into their lower level CFG
/// equivalent blocks.
FunctionPass *mlir::createLowerAffinePass() { return new LowerAffinePass(); }

static PassRegistration<LowerAffinePass>
    pass("lower-affine",
         "Lower If, For, AffineApply instructions to primitive equivalents");
