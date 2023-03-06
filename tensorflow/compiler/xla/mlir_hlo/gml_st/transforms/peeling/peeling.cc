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

#include "gml_st/transforms/peeling/peeling.h"

#include "gml_st/IR/gml_st_ops.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace gml_st {
namespace {

bool isATensor(Type t) { return t.isa<TensorType>(); }

/// Return true if the given op has only tensor-typed results or operands.
bool hasTensorSemantics(Operation *op) {
  return llvm::all_of(op->getResultTypes(), isATensor) ||
         llvm::all_of(op->getOperandTypes(), isATensor);
}

LogicalResult peelLoop(RewriterBase &b, ParallelOp loopOp, int64_t idx,
                       ParallelOp &result, Value &splitBound) {
  if (!hasTensorSemantics(loopOp)) return failure();

  Value lb = loopOp.getLowerBound()[idx], ub = loopOp.getUpperBound()[idx],
        step = loopOp.getStep()[idx];
  auto ubInt = getConstantIntValue(ub);

  auto loc = loopOp.getLoc();
  AffineExpr exprLb, exprUb, exprStep;
  bindSymbols(b.getContext(), exprLb, exprUb, exprStep);
  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap = AffineMap::get(0, 3, exprUb - ((exprUb - exprLb) % exprStep));
  SmallVector<Value> operands{lb, ub, step};
  canonicalizeMapAndOperands(&modMap, &operands);
  modMap = simplifyAffineMap(modMap);
  RewriterBase::InsertionGuard guard(b);
  b.setInsertionPoint(loopOp);
  splitBound = b.createOrFold<AffineApplyOp>(loc, modMap, operands);

  // No specialization necessary if step already divides upper bound evenly.
  if (splitBound == ub || (ubInt && ubInt == getConstantIntValue(splitBound)))
    return failure();

  // Create remainder loop.
  IRMapping bvm;
  for (const auto &[res, termDst] :
       llvm::zip(loopOp.getResults(), loopOp.getLoopLikeOpInits())) {
    bvm.map(termDst, res);
  }
  b.setInsertionPointAfter(loopOp);
  auto remainderLoop = cast<ParallelOp>(b.clone(*loopOp.getOperation(), bvm));

  Operation *remainderLoopOp = remainderLoop.getOperation();

  for (const auto &[oldRes, newRes] :
       llvm::zip(loopOp.getResults(), remainderLoop.getResults())) {
    SmallPtrSet<Operation *, 4> exceptions({remainderLoopOp});
    for (OpOperand &use : oldRes.getUses()) {
      Operation *user = use.getOwner();
      if (user->getParentOp() == remainderLoopOp) exceptions.insert(user);
    }
    oldRes.replaceAllUsesExcept(newRes, exceptions);
  }

  // Set new loop bounds.
  b.updateRootInPlace(loopOp, [&]() {
    SmallVector<Value> ubs = loopOp.getUpperBound();
    ubs[idx] = splitBound;
    loopOp.getUpperBoundMutable().assign(ubs);
  });
  SmallVector<Value> lbs = remainderLoop.getLowerBound();
  lbs[idx] = splitBound;
  b.updateRootInPlace(remainderLoop, [&]() {
    remainderLoop.getLowerBoundMutable().assign(lbs);
  });

  result = remainderLoop;
  return success();
}

template <typename OpTy>
void rewriteAffineOpAfterPeeling(RewriterBase &rewriter, Operation *mainLoop,
                                 Operation *remainderLoop, Value mainIv,
                                 Value remainderIv, Value ub, Value step) {
  mainLoop->walk([&](OpTy affineOp) {
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, mainIv, ub, step,
                                     /*insideLoop=*/true);
  });
  remainderLoop->walk([&](OpTy affineOp) {
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, remainderIv, ub, step,
                                     /*insideLoop=*/false);
  });
}

}  // namespace

GmlStPeelingResult peelAllLoops(ParallelOp loop,
                                mlir::PatternRewriter &rewriter) {
  GmlStPeelingResult peelingResult;

  bool hasMainLoop = true;
  for (unsigned peeledIdx = 0; peeledIdx < loop.getNumLoops(); ++peeledIdx) {
    int64_t numLoops = loop.getNumLoops();
    if (peeledIdx < 0 || numLoops <= peeledIdx) continue;

    Value ub = loop.getUpperBound()[peeledIdx];
    Value step = loop.getStep()[peeledIdx];
    auto ubInt = getConstantIntValue(ub);
    auto stepInt = getConstantIntValue(step);

    // If the loop is smaller than the step, then append loop as tail. Needs to
    // be done only once.
    if (ubInt && stepInt && ubInt < stepInt) {
      if (hasMainLoop) {
        peelingResult.tailLoops.push_back(loop);
        hasMainLoop = false;
      }
      continue;
    }

    ParallelOp remainderLoop;
    Value splitBound;
    if (failed(peelLoop(rewriter, loop, peeledIdx, remainderLoop, splitBound)))
      continue;

    // Rewrite affine.min and affine.max ops.
    Value mainIv = loop.getInductionVars()[peeledIdx],
          remainderIv = remainderLoop.getInductionVars()[peeledIdx];

    rewriteAffineOpAfterPeeling<AffineMinOp>(rewriter, loop, remainderLoop,
                                             mainIv, remainderIv, ub, step);
    rewriteAffineOpAfterPeeling<AffineMaxOp>(rewriter, loop, remainderLoop,
                                             mainIv, remainderIv, ub, step);

    // Mark the new loop if one was created.
    peelingResult.tailLoops.push_back(remainderLoop);
  }

  // Update main loop if applicable.
  if (hasMainLoop) peelingResult.mainLoop = loop;

  return peelingResult;
}

SCFForPeelingResult peelSCFForOp(RewriterBase &rewriter, scf::ForOp loop) {
  // Peeling fails, if the step divides the upper bound. In that case,
  // we still want to return {loop, nullptr}.
  scf::ForOp tailLoop;
  return succeeded(scf::peelForLoopAndSimplifyBounds(rewriter, loop, tailLoop))
             ? SCFForPeelingResult{loop, tailLoop}
             : SCFForPeelingResult{loop, nullptr};
}

}  // namespace gml_st
}  // namespace mlir
