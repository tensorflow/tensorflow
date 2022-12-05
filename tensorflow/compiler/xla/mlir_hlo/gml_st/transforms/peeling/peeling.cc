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

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/transforms.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"

namespace mlir {
namespace gml_st {
namespace {

bool isATensor(Type t) { return t.isa<TensorType>(); }

/// Return true if the given op has only tensor-typed results or operands.
bool hasTensorSemantics(Operation *op) {
  return llvm::all_of(op->getResultTypes(), isATensor) ||
         llvm::all_of(op->getOperandTypes(), isATensor);
}

/// Rewrite a LoopOp/ParallelOp/ForOp with bounds/step that potentially do not
/// divide evenly into two LoopOp/ParallelOp/ForOps: One where the step divides
/// the iteration space evenly, followed another one for the last (partial)
/// iteration (if any). This function only rewrites the `idx`-th loop of the
/// loop nest represented by the LoopOp/ParallelOp/ForOp. To peel the entire
/// loop nest, this function must be called multiple times.
///
/// This function rewrites the given LoopOp/ParallelOp/ForOp in-place and
/// creates a new LoopOp/ParallelOp/ForOp for the last iteration. It replaces
/// all uses of the original LoopOp/ParallelOp/ForOp with the results of the
/// newly generated one.
///
/// The newly generated LoopOp/ParallelOp/ForOp is returned via `result`. The
/// boundary at which the loop is split (new upper bound) is returned via
/// `splitBound`.  The return value indicates whether the
/// LoopOp/ParallelOp/ForOp was rewritten or not.
template <typename LoopTy>
LogicalResult peelLoop(RewriterBase &b, LoopTy loopOp, int64_t idx,
                       LoopTy &result, Value &splitBound) {
  if (!hasTensorSemantics(loopOp)) return failure();

  Value lb = loopOp.getLowerBound()[idx], ub = loopOp.getUpperBound()[idx],
        step = loopOp.getStep()[idx];
  auto ubInt = getConstantIntValue(ub);

  auto loc = loopOp.getLoc();
  AffineExpr exprLb, exprUb, exprStep;
  bindSymbols(b.getContext(), exprLb, exprUb, exprStep);
  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap = AffineMap::get(0, 3, {exprUb - ((exprUb - exprLb) % exprStep)});
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
  BlockAndValueMapping bvm;
  for (const auto &[res, termDst] :
       llvm::zip(loopOp.getResults(), loopOp.getLoopLikeOpInits())) {
    bvm.map(termDst, res);
  }
  b.setInsertionPointAfter(loopOp);
  auto remainderLoop = cast<LoopTy>(b.clone(*loopOp.getOperation(), bvm));

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

template <typename OpTy, bool IsMin>
void rewriteAffineOpAfterPeeling(RewriterBase &rewriter, Operation *mainLoop,
                                 Operation *remainderLoop, Value mainIv,
                                 Value remainderIv, Value ub, Value step) {
  mainLoop->walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.getOperands(), IsMin, mainIv, ub,
                                     step, /*insideLoop=*/true);
  });
  remainderLoop->walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.getOperands(), IsMin, remainderIv,
                                     ub, step, /*insideLoop=*/false);
  });
}

template <typename LoopTy>
FailureOr<LoopTy> peelAndCanonicalizeGmlStLoopImpl(RewriterBase &rewriter,
                                                   LoopTy loopOp, int64_t idx) {
  int64_t numLoops = loopOp.getNumLoops();
  if (idx < 0 || numLoops <= idx) return failure();

  Value ub = loopOp.getUpperBound()[idx];
  LoopTy remainderLoop;
  Value splitBound;
  if (failed(
          peelLoop<LoopTy>(rewriter, loopOp, idx, remainderLoop, splitBound)))
    return failure();

  // Rewrite affine.min and affine.max ops.
  Value mainIv = loopOp.getInductionVars()[idx], step = loopOp.getStep()[idx],
        remainderIv = remainderLoop.getInductionVars()[idx];

  rewriteAffineOpAfterPeeling<AffineMinOp, /*IsMin=*/true>(
      rewriter, loopOp, remainderLoop, mainIv, remainderIv, ub, step);
  rewriteAffineOpAfterPeeling<AffineMaxOp, /*IsMin=*/false>(
      rewriter, loopOp, remainderLoop, mainIv, remainderIv, ub, step);

  return remainderLoop;
}

template <typename LoopTy>
PeelingResult peelAllLoopsImpl(LoopTy loop, mlir::PatternRewriter &rewriter) {
  setLabel(loop, kPeelingAppliedLabel);
  PeelingResult peelingResult;
  for (unsigned peeledIdx = 0; peeledIdx < loop.getNumLoops(); ++peeledIdx) {
    auto peel =
        peelAndCanonicalizeGmlStLoopImpl<LoopTy>(rewriter, loop, peeledIdx);
    if (failed(peel)) continue;
    // Mark the new loop if one was created.
    setLabel(peel->getOperation(), kPeelingAppliedLabel);
    peelingResult.push_back(*peel);
  }
  return peelingResult;
}
}  // namespace

PeelingResult peelAllLoops(LoopOp loop, mlir::PatternRewriter &rewriter) {
  return peelAllLoopsImpl<LoopOp>(loop, rewriter);
}

PeelingResult peelAllLoops(ForOp loop, mlir::PatternRewriter &rewriter) {
  return peelAllLoopsImpl<ForOp>(loop, rewriter);
}

PeelingResult peelAllLoops(ParallelOp loop, mlir::PatternRewriter &rewriter) {
  return peelAllLoopsImpl<ParallelOp>(loop, rewriter);
}

FailureOr<LoopOp> peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                               LoopOp loopOp, int64_t idx) {
  return peelAndCanonicalizeGmlStLoopImpl<LoopOp>(rewriter, loopOp, idx);
}

FailureOr<ForOp> peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                              ForOp loopOp, int64_t idx) {
  return peelAndCanonicalizeGmlStLoopImpl<ForOp>(rewriter, loopOp, idx);
}

FailureOr<ParallelOp> peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                                   ParallelOp loopOp,
                                                   int64_t idx) {
  return peelAndCanonicalizeGmlStLoopImpl<ParallelOp>(rewriter, loopOp, idx);
}

}  // namespace gml_st
}  // namespace mlir
