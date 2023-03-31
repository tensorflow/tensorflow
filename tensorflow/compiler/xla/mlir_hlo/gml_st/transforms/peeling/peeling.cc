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
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

/// Peel `loopOp`. If peeling is taking place, this function updates `loopOp`
/// (the main loop) upper bound in place, creates a remainder loop and puts it
/// into `result`.
LogicalResult peelLoop(RewriterBase &b, scf::ForallOp loopOp, int64_t idx,
                       scf::ForallOp &result, Value &splitBound) {
  if (!hasTensorSemantics(loopOp)) return failure();

  Location loc = loopOp.getLoc();
  Value lb =
      getValueOrCreateConstantIndexOp(b, loc, loopOp.getMixedLowerBound()[idx]);
  Value ub =
      getValueOrCreateConstantIndexOp(b, loc, loopOp.getMixedUpperBound()[idx]);
  Value step =
      getValueOrCreateConstantIndexOp(b, loc, loopOp.getMixedStep()[idx]);
  auto ubInt = getConstantIntValue(ub);

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
       llvm::zip(loopOp.getResults(), loopOp.getOutputs())) {
    bvm.map(termDst, res);
  }
  b.setInsertionPointAfter(loopOp);
  auto remainderLoop =
      cast<scf::ForallOp>(b.clone(*loopOp.getOperation(), bvm));

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
  SmallVector<OpFoldResult> ubs = loopOp.getMixedUpperBound();
  ubs[idx] = splitBound;
  SmallVector<Value> dynamicUbs;
  SmallVector<int64_t> staticUbs;
  dispatchIndexOpFoldResults(ubs, dynamicUbs, staticUbs);
  b.updateRootInPlace(loopOp, [&]() {
    loopOp.getDynamicUpperBoundMutable().assign(dynamicUbs);
    loopOp.setStaticUpperBound(staticUbs);
  });

  SmallVector<OpFoldResult> lbs = remainderLoop.getMixedLowerBound();
  lbs[idx] = splitBound;
  SmallVector<Value> dynamicLbs;
  SmallVector<int64_t> staticLbs;
  dispatchIndexOpFoldResults(lbs, dynamicLbs, staticLbs);
  b.updateRootInPlace(remainderLoop, [&]() {
    remainderLoop.getDynamicLowerBoundMutable().assign(dynamicLbs);
    remainderLoop.setStaticLowerBound(staticLbs);
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

GmlStPeelingResult peelAllLoops(scf::ForallOp loop,
                                mlir::PatternRewriter &rewriter) {
  GmlStPeelingResult peelingResult;

  // If upper bound is smaller than step for all iteration domains, we don't
  // need to peel, as the original loop will eventually be canonicalized (loop
  // of single iteration). Returning the original loop as a tail loop because
  // some transformations (e.g. transformReduceForCpu) need to post-process the
  // tail loops after peeling.
  if (llvm::all_of(llvm::zip(loop.getMixedUpperBound(), loop.getMixedStep()),
                   [](auto tuple) {
                     auto ubInt = getConstantIntValue(std::get<0>(tuple));
                     auto stepInt = getConstantIntValue(std::get<1>(tuple));
                     return ubInt && stepInt && ubInt < stepInt;
                   })) {
    peelingResult.tailLoops.push_back(loop);
    return peelingResult;
  }

  for (unsigned peeledIdx = 0; peeledIdx < loop.getRank(); ++peeledIdx) {
    OpFoldResult ubOfr = loop.getMixedUpperBound()[peeledIdx];
    OpFoldResult stepOfr = loop.getMixedStep()[peeledIdx];
    auto ubInt = getConstantIntValue(ubOfr);
    auto stepInt = getConstantIntValue(stepOfr);

    // If the upper bound is smaller than the step, don't peel this dimension.
    if (ubInt && stepInt && ubInt <= stepInt) {
      continue;
    }

    Location loc = loop.getLoc();
    Value ub = getValueOrCreateConstantIndexOp(rewriter, loc, ubOfr);
    Value step = getValueOrCreateConstantIndexOp(rewriter, loc, stepOfr);
    scf::ForallOp remainderLoop;
    Value splitBound;
    if (failed(peelLoop(rewriter, loop, peeledIdx, remainderLoop, splitBound)))
      continue;

    // Rewrite affine.min and affine.max ops.
    Value mainIv = loop.getInductionVars()[peeledIdx];
    Value remainderIv = remainderLoop.getInductionVars()[peeledIdx];

    rewriteAffineOpAfterPeeling<AffineMinOp>(rewriter, loop, remainderLoop,
                                             mainIv, remainderIv, ub, step);
    rewriteAffineOpAfterPeeling<AffineMaxOp>(rewriter, loop, remainderLoop,
                                             mainIv, remainderIv, ub, step);

    peelingResult.tailLoops.push_back(remainderLoop);
  }

  peelingResult.mainLoop = loop;

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
