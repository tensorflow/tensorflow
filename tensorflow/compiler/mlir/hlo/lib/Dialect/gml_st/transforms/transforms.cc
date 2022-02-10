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

#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir {
namespace gml_st {
namespace {

/// Rewrite a LoopOp with bounds/step that potentially do not divide evenly
/// into two LoopOps: One where the step divides the iteration space
/// evenly, followed another one for the last (partial) iteration (if any). This
/// function only rewrites the `idx`-th loop of the loop nest represented by
/// the LoopOp. To peel the entire loop nest, this function must be called
/// multiple times.
///
/// This function rewrites the given LoopOp in-place and creates a new
/// LoopOp for the last iteration. It replaces all uses of the original
/// LoopOp with the results of the newly generated one.
///
/// The newly generated LoopOp is returned via `result`. The boundary
/// at which the loop is split (new upper bound) is returned via `splitBound`.
/// The return value indicates whether the LoopOp was rewritten or not.
static LogicalResult peelLoop(RewriterBase &b, LoopOp loopOp, int64_t idx,
                              LoopOp &result, Value &splitBound) {
  Value lb = loopOp.lowerBound()[idx], ub = loopOp.upperBound()[idx],
        step = loopOp.step()[idx];
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
  b.setInsertionPointAfter(loopOp);
  auto remainderLoop = cast<LoopOp>(b.clone(*loopOp.getOperation()));
  loopOp.replaceAllUsesWith(remainderLoop->getResults());
  // Outputs: Take tensors from main loop's results. Take memrefs from main
  // loop's outputs.
  SmallVector<Value> remainderOutputs;
  for (unsigned o = 0, t = 0; o < loopOp.getNumOutputs(); ++o) {
    remainderOutputs.push_back(loopOp.outputs()[o].getType().isa<MemRefType>()
                                   ? loopOp.outputs()[o]
                                   : loopOp->getResult(t++));
  }
  remainderLoop.outputsMutable().assign(remainderOutputs);

  // Set new loop bounds.
  b.updateRootInPlace(loopOp, [&]() {
    SmallVector<Value> ubs = loopOp.upperBound();
    ubs[idx] = splitBound;
    loopOp.upperBoundMutable().assign(ubs);
  });
  SmallVector<Value> lbs = remainderLoop.lowerBound();
  lbs[idx] = splitBound;
  remainderLoop.lowerBoundMutable().assign(lbs);

  result = remainderLoop;
  return success();
}

template <typename OpTy, bool IsMin>
static void rewriteAffineOpAfterPeeling(RewriterBase &rewriter, LoopOp mainLoop,
                                        LoopOp remainderLoop, Value mainIv,
                                        Value remainderIv, Value ub,
                                        Value step) {
  mainLoop.walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.operands(), IsMin, mainIv, ub,
                                     step, /*insideLoop=*/true);
  });
  remainderLoop.walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.operands(), IsMin, remainderIv,
                                     ub, step, /*insideLoop=*/false);
  });
}

}  // namespace

LogicalResult peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                           LoopOp loopOp, int64_t idx,
                                           LoopOp &result) {
  int64_t numLoops = loopOp.iterator_types().size();
  if (idx < 0 || numLoops <= idx) return failure();

  Value ub = loopOp.upperBound()[idx];
  LoopOp remainderLoop;
  Value splitBound;
  if (failed(peelLoop(rewriter, loopOp, idx, remainderLoop, splitBound)))
    return failure();

  // Rewrite affine.min and affine.max ops.
  Value mainIv = loopOp.getInductionVars()[idx], step = loopOp.step()[idx],
        remainderIv = remainderLoop.getInductionVars()[idx];

  rewriteAffineOpAfterPeeling<AffineMinOp, /*IsMin=*/true>(
      rewriter, loopOp, remainderLoop, mainIv, remainderIv, ub, step);
  rewriteAffineOpAfterPeeling<AffineMaxOp, /*IsMin=*/false>(
      rewriter, loopOp, remainderLoop, mainIv, remainderIv, ub, step);

  result = remainderLoop;
  return success();
}

}  // namespace gml_st
}  // namespace mlir
