//===- LowerIfAndFor.cpp - Lower If and For instructions to CFG -----------===//
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
// This file lowers If and For instructions within a function into their lower
// level CFG equivalent blocks.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Transforms/LoweringUtils.h"
#include "mlir/Transforms/Passes.h"
using namespace mlir;

namespace {
class LowerIfAndForPass : public FunctionPass {
public:
  LowerIfAndForPass() : FunctionPass(&passID) {}
  PassResult runOnFunction(Function *function) override;

  bool lowerForInst(ForInst *forInst);
  bool lowerIfInst(IfInst *ifInst);

  static char passID;
};
} // end anonymous namespace

char LowerIfAndForPass::passID = 0;

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
// computes the loop bounds that are loop-invariant in MLFunctions; the
// condition block that checks the exit condition of the loop; the body SESE
// region; and the end block that post-dominates the loop.  The end block of the
// loop becomes the new end of the current SESE region.  The body of the loop is
// constructed recursively after starting a new region (it may be, for example,
// a nested loop).  Induction variable modification is appended to the body SESE
// region that always loops back to the condition block.
//
//      +--------------------------------+
//      |   <code before the ForInst>    |
//      |   <compute initial %iv value>  |
//      |   br cond(%iv)                 |
//      +--------------------------------+
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
//      |   <code after the ForInst>     |
//      +--------------------------------+
//
bool LowerIfAndForPass::lowerForInst(ForInst *forInst) {
  auto loc = forInst->getLoc();

  // Start by splitting the block containing the 'for' into two parts.  The part
  // before will get the init code, the part after will be the end point.
  auto *initBlock = forInst->getBlock();
  auto *endBlock = initBlock->splitBlock(forInst);

  // Create the condition block, with its argument for the loop induction
  // variable.  We set it up below.
  auto *conditionBlock = new Block();
  conditionBlock->insertBefore(endBlock);
  auto *iv = conditionBlock->addArgument(IndexType::get(forInst->getContext()));

  // Create the body block, moving the body of the forInst over to it.
  auto *bodyBlock = new Block();
  bodyBlock->insertBefore(endBlock);

  auto *oldBody = forInst->getBody();
  bodyBlock->getInstructions().splice(bodyBlock->begin(),
                                      oldBody->getInstructions(),
                                      oldBody->begin(), oldBody->end());

  // The code in the body of the forInst now uses 'iv' as its indvar.
  forInst->replaceAllUsesWith(iv);

  // Append the induction variable stepping logic and branch back to the exit
  // condition block.  Construct an affine expression f : (x -> x+step) and
  // apply this expression to the induction variable.
  FuncBuilder builder(bodyBlock);
  auto affStep = builder.getAffineConstantExpr(forInst->getStep());
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
  SmallVector<Value *, 8> operands(forInst->getLowerBoundOperands());
  auto lbValues = expandAffineMap(&builder, forInst->getLoc(),
                                  forInst->getLowerBoundMap(), operands);
  if (!lbValues)
    return true;
  Value *lowerBound =
      buildMinMaxReductionSeq(loc, CmpIPredicate::SGT, *lbValues, builder);

  operands.assign(forInst->getUpperBoundOperands().begin(),
                  forInst->getUpperBoundOperands().end());
  auto ubValues = expandAffineMap(&builder, forInst->getLoc(),
                                  forInst->getUpperBoundMap(), operands);
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
  forInst->erase();
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
//      | <code before the IfInst>       |
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
//      |   <code after the IfInst>      |
//      +--------------------------------+
//
bool LowerIfAndForPass::lowerIfInst(IfInst *ifInst) {
  auto loc = ifInst->getLoc();

  // Start by splitting the block containing the 'if' into two parts.  The part
  // before will contain the condition, the part after will be the continuation
  // point.
  auto *condBlock = ifInst->getBlock();
  auto *continueBlock = condBlock->splitBlock(ifInst);

  // Create a block for the 'then' code, inserting it between the cond and
  // continue blocks.  Move the instructions over from the IfInst and add a
  // branch to the continuation point.
  Block *thenBlock = new Block();
  thenBlock->insertBefore(continueBlock);

  auto *oldThen = ifInst->getThen();
  thenBlock->getInstructions().splice(thenBlock->begin(),
                                      oldThen->getInstructions(),
                                      oldThen->begin(), oldThen->end());
  FuncBuilder builder(thenBlock);
  builder.create<BranchOp>(loc, continueBlock);

  // Handle the 'else' block the same way, but we skip it if we have no else
  // code.
  Block *elseBlock = continueBlock;
  if (auto *oldElse = ifInst->getElse()) {
    elseBlock = new Block();
    elseBlock->insertBefore(continueBlock);

    elseBlock->getInstructions().splice(elseBlock->begin(),
                                        oldElse->getInstructions(),
                                        oldElse->begin(), oldElse->end());
    builder.setInsertionPointToEnd(elseBlock);
    builder.create<BranchOp>(loc, continueBlock);
   }

  // Ok, now we just have to handle the condition logic.
  auto integerSet = ifInst->getCondition().getIntegerSet();

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
PassResult LowerIfAndForPass::runOnFunction(Function *function) {
  SmallVector<Instruction *, 8> instsToRewrite;

  // Collect all the If and For statements.  We do this as a prepass to avoid
  // invalidating the walker with our rewrite.
  function->walkInsts([&](Instruction *inst) {
    if (isa<IfInst>(inst) || isa<ForInst>(inst))
      instsToRewrite.push_back(inst);
  });

  // Rewrite all of the ifs and fors.  We walked the instructions in preorder,
  // so we know that we will rewrite them in the same order.
  for (auto *inst : instsToRewrite)
    if (auto *ifInst = dyn_cast<IfInst>(inst)) {
      if (lowerIfInst(ifInst))
        return failure();
    } else {
      if (lowerForInst(cast<ForInst>(inst)))
        return failure();
    }

  return success();
}

/// Lowers If and For instructions within a function into their lower level CFG
/// equivalent blocks.
FunctionPass *mlir::createLowerIfAndForPass() {
  return new LowerIfAndForPass();
}

static PassRegistration<LowerIfAndForPass>
    pass("lower-if-and-for",
         "Lower If and For instructions to CFG equivalents");
