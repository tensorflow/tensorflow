//===- LoopUtils.cpp ---- Misc utilities for loop transformation ----------===//
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
// This file implements miscellaneous loop transformation routines.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LoopUtils.h"

#include "mlir/AffineOps/AffineOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/Module.h"

#define DEBUG_TYPE "LoopUtils"

using namespace mlir;

/// Computes the cleanup loop lower bound of the loop being unrolled with
/// the specified unroll factor; this bound will also be upper bound of the main
/// part of the unrolled loop. Computes the bound as an AffineMap with its
/// operands or a null map when the trip count can't be expressed as an affine
/// expression.
void mlir::getCleanupLoopLowerBound(AffineForOp forOp, unsigned unrollFactor,
                                    AffineMap *map,
                                    SmallVectorImpl<Value *> *operands,
                                    OpBuilder &b) {
  auto lbMap = forOp.getLowerBoundMap();

  // Single result lower bound map only.
  if (lbMap.getNumResults() != 1) {
    *map = AffineMap();
    return;
  }

  AffineMap tripCountMap;
  SmallVector<Value *, 4> tripCountOperands;
  buildTripCountMapAndOperands(forOp, &tripCountMap, &tripCountOperands);

  // Sometimes the trip count cannot be expressed as an affine expression.
  if (!tripCountMap) {
    *map = AffineMap();
    return;
  }

  unsigned step = forOp.getStep();

  SmallVector<Value *, 4> lbOperands(forOp.getLowerBoundOperands());
  auto lb = b.create<AffineApplyOp>(forOp.getLoc(), lbMap, lbOperands);

  // For each upper bound expr, get the range.
  // Eg: affine.for %i = lb to min (ub1, ub2),
  // where tripCountExprs yield (tr1, tr2), we create affine.apply's:
  // lb + tr1 - tr1 % ufactor, lb + tr2 - tr2 % ufactor; the results of all
  // these affine.apply's make up the cleanup loop lower bound.
  SmallVector<AffineExpr, 4> bumpExprs(tripCountMap.getNumResults());
  SmallVector<Value *, 4> bumpValues(tripCountMap.getNumResults());
  for (unsigned i = 0, e = tripCountMap.getNumResults(); i < e; i++) {
    auto tripCountExpr = tripCountMap.getResult(i);
    bumpExprs[i] = (tripCountExpr - tripCountExpr % unrollFactor) * step;
    auto bumpMap = b.getAffineMap(tripCountMap.getNumDims(),
                                  tripCountMap.getNumSymbols(), bumpExprs[i]);
    bumpValues[i] =
        b.create<AffineApplyOp>(forOp.getLoc(), bumpMap, tripCountOperands);
  }

  SmallVector<AffineExpr, 4> newUbExprs(tripCountMap.getNumResults());
  for (unsigned i = 0, e = bumpExprs.size(); i < e; i++)
    newUbExprs[i] = b.getAffineDimExpr(0) + b.getAffineDimExpr(i + 1);

  operands->clear();
  operands->push_back(lb);
  operands->append(bumpValues.begin(), bumpValues.end());
  *map = b.getAffineMap(1 + tripCountMap.getNumResults(), 0, newUbExprs);
  // Simplify the map + operands.
  fullyComposeAffineMapAndOperands(map, operands);
  *map = simplifyAffineMap(*map);
  canonicalizeMapAndOperands(map, operands);
  // Remove any affine.apply's that became dead from the simplification above.
  for (auto *v : bumpValues) {
    if (v->use_empty()) {
      v->getDefiningOp()->erase();
    }
  }
  if (lb.use_empty())
    lb.erase();
}

/// Promotes the loop body of a forOp to its containing block if the forOp
/// was known to have a single iteration.
// TODO(bondhugula): extend this for arbitrary affine bounds.
LogicalResult mlir::promoteIfSingleIteration(AffineForOp forOp) {
  Optional<uint64_t> tripCount = getConstantTripCount(forOp);
  if (!tripCount.hasValue() || tripCount.getValue() != 1)
    return failure();

  // TODO(mlir-team): there is no builder for a max.
  if (forOp.getLowerBoundMap().getNumResults() != 1)
    return failure();

  // Replaces all IV uses to its single iteration value.
  auto *iv = forOp.getInductionVar();
  Operation *op = forOp.getOperation();
  if (!iv->use_empty()) {
    if (forOp.hasConstantLowerBound()) {
      OpBuilder topBuilder(op->getParentOfType<FuncOp>().getBody());
      auto constOp = topBuilder.create<ConstantIndexOp>(
          forOp.getLoc(), forOp.getConstantLowerBound());
      iv->replaceAllUsesWith(constOp);
    } else {
      AffineBound lb = forOp.getLowerBound();
      SmallVector<Value *, 4> lbOperands(lb.operand_begin(), lb.operand_end());
      OpBuilder builder(op->getBlock(), Block::iterator(op));
      if (lb.getMap() == builder.getDimIdentityMap()) {
        // No need of generating an affine.apply.
        iv->replaceAllUsesWith(lbOperands[0]);
      } else {
        auto affineApplyOp = builder.create<AffineApplyOp>(
            op->getLoc(), lb.getMap(), lbOperands);
        iv->replaceAllUsesWith(affineApplyOp);
      }
    }
  }
  // Move the loop body operations, except for terminator, to the loop's
  // containing block.
  auto *block = op->getBlock();
  forOp.getBody()->getOperations().back().erase();
  block->getOperations().splice(Block::iterator(op),
                                forOp.getBody()->getOperations());
  forOp.erase();
  return success();
}

/// Promotes all single iteration for op's in the FuncOp, i.e., moves
/// their body into the containing Block.
void mlir::promoteSingleIterationLoops(FuncOp f) {
  // Gathers all innermost loops through a post order pruned walk.
  f.walk<AffineForOp>(
      [](AffineForOp forOp) { promoteIfSingleIteration(forOp); });
}

/// Generates a 'affine.for' op with the specified lower and upper bounds
/// while generating the right IV remappings for the shifted operations. The
/// operation blocks that go into the loop are specified in instGroupQueue
/// starting from the specified offset, and in that order; the first element of
/// the pair specifies the shift applied to that group of operations; note
/// that the shift is multiplied by the loop step before being applied. Returns
/// nullptr if the generated loop simplifies to a single iteration one.
static AffineForOp
generateLoop(AffineMap lbMap, AffineMap ubMap,
             const std::vector<std::pair<uint64_t, ArrayRef<Operation *>>>
                 &instGroupQueue,
             unsigned offset, AffineForOp srcForInst, OpBuilder b) {
  SmallVector<Value *, 4> lbOperands(srcForInst.getLowerBoundOperands());
  SmallVector<Value *, 4> ubOperands(srcForInst.getUpperBoundOperands());

  assert(lbMap.getNumInputs() == lbOperands.size());
  assert(ubMap.getNumInputs() == ubOperands.size());

  auto loopChunk =
      b.create<AffineForOp>(srcForInst.getLoc(), lbOperands, lbMap, ubOperands,
                            ubMap, srcForInst.getStep());
  auto *loopChunkIV = loopChunk.getInductionVar();
  auto *srcIV = srcForInst.getInductionVar();

  BlockAndValueMapping operandMap;

  OpBuilder bodyBuilder = loopChunk.getBodyBuilder();
  for (auto it = instGroupQueue.begin() + offset, e = instGroupQueue.end();
       it != e; ++it) {
    uint64_t shift = it->first;
    auto insts = it->second;
    // All 'same shift' operations get added with their operands being
    // remapped to results of cloned operations, and their IV used remapped.
    // Generate the remapping if the shift is not zero: remappedIV = newIV -
    // shift.
    if (!srcIV->use_empty() && shift != 0) {
      auto ivRemap = bodyBuilder.create<AffineApplyOp>(
          srcForInst.getLoc(),
          bodyBuilder.getSingleDimShiftAffineMap(
              -static_cast<int64_t>(srcForInst.getStep() * shift)),
          loopChunkIV);
      operandMap.map(srcIV, ivRemap);
    } else {
      operandMap.map(srcIV, loopChunkIV);
    }
    for (auto *op : insts) {
      if (!isa<AffineTerminatorOp>(op))
        bodyBuilder.clone(*op, operandMap);
    }
  };
  if (succeeded(promoteIfSingleIteration(loopChunk)))
    return AffineForOp();
  return loopChunk;
}

/// Skew the operations in the body of a 'affine.for' operation with the
/// specified operation-wise shifts. The shifts are with respect to the
/// original execution order, and are multiplied by the loop 'step' before being
/// applied. A shift of zero for each operation will lead to no change.
// The skewing of operations with respect to one another can be used for
// example to allow overlap of asynchronous operations (such as DMA
// communication) with computation, or just relative shifting of operations
// for better register reuse, locality or parallelism. As such, the shifts are
// typically expected to be at most of the order of the number of operations.
// This method should not be used as a substitute for loop distribution/fission.
// This method uses an algorithm// in time linear in the number of operations
// in the body of the for loop - (using the 'sweep line' paradigm). This method
// asserts preservation of SSA dominance. A check for that as well as that for
// memory-based depedence preservation check rests with the users of this
// method.
LogicalResult mlir::instBodySkew(AffineForOp forOp, ArrayRef<uint64_t> shifts,
                                 bool unrollPrologueEpilogue) {
  if (forOp.getBody()->begin() == std::prev(forOp.getBody()->end()))
    return success();

  // If the trip counts aren't constant, we would need versioning and
  // conditional guards (or context information to prevent such versioning). The
  // better way to pipeline for such loops is to first tile them and extract
  // constant trip count "full tiles" before applying this.
  auto mayBeConstTripCount = getConstantTripCount(forOp);
  if (!mayBeConstTripCount.hasValue()) {
    LLVM_DEBUG(forOp.emitRemark("non-constant trip count loop not handled"));
    return success();
  }
  uint64_t tripCount = mayBeConstTripCount.getValue();

  assert(isInstwiseShiftValid(forOp, shifts) &&
         "shifts will lead to an invalid transformation\n");

  int64_t step = forOp.getStep();

  unsigned numChildInsts = forOp.getBody()->getOperations().size();

  // Do a linear time (counting) sort for the shifts.
  uint64_t maxShift = 0;
  for (unsigned i = 0; i < numChildInsts; i++) {
    maxShift = std::max(maxShift, shifts[i]);
  }
  // Such large shifts are not the typical use case.
  if (maxShift >= numChildInsts) {
    forOp.emitWarning("not shifting because shifts are unrealistically large");
    return success();
  }

  // An array of operation groups sorted by shift amount; each group has all
  // operations with the same shift in the order in which they appear in the
  // body of the 'affine.for' op.
  std::vector<std::vector<Operation *>> sortedInstGroups(maxShift + 1);
  unsigned pos = 0;
  for (auto &op : *forOp.getBody()) {
    auto shift = shifts[pos++];
    sortedInstGroups[shift].push_back(&op);
  }

  // Unless the shifts have a specific pattern (which actually would be the
  // common use case), prologue and epilogue are not meaningfully defined.
  // Nevertheless, if 'unrollPrologueEpilogue' is set, we will treat the first
  // loop generated as the prologue and the last as epilogue and unroll these
  // fully.
  AffineForOp prologue;
  AffineForOp epilogue;

  // Do a sweep over the sorted shifts while storing open groups in a
  // vector, and generating loop portions as necessary during the sweep. A block
  // of operations is paired with its shift.
  std::vector<std::pair<uint64_t, ArrayRef<Operation *>>> instGroupQueue;

  auto origLbMap = forOp.getLowerBoundMap();
  uint64_t lbShift = 0;
  OpBuilder b(forOp.getOperation());
  for (uint64_t d = 0, e = sortedInstGroups.size(); d < e; ++d) {
    // If nothing is shifted by d, continue.
    if (sortedInstGroups[d].empty())
      continue;
    if (!instGroupQueue.empty()) {
      assert(d >= 1 &&
             "Queue expected to be empty when the first block is found");
      // The interval for which the loop needs to be generated here is:
      // [lbShift, min(lbShift + tripCount, d)) and the body of the
      // loop needs to have all operations in instQueue in that order.
      AffineForOp res;
      if (lbShift + tripCount * step < d * step) {
        res = generateLoop(
            b.getShiftedAffineMap(origLbMap, lbShift),
            b.getShiftedAffineMap(origLbMap, lbShift + tripCount * step),
            instGroupQueue, 0, forOp, b);
        // Entire loop for the queued op groups generated, empty it.
        instGroupQueue.clear();
        lbShift += tripCount * step;
      } else {
        res = generateLoop(b.getShiftedAffineMap(origLbMap, lbShift),
                           b.getShiftedAffineMap(origLbMap, d), instGroupQueue,
                           0, forOp, b);
        lbShift = d * step;
      }
      if (!prologue && res)
        prologue = res;
      epilogue = res;
    } else {
      // Start of first interval.
      lbShift = d * step;
    }
    // Augment the list of operations that get into the current open interval.
    instGroupQueue.push_back({d, sortedInstGroups[d]});
  }

  // Those operations groups left in the queue now need to be processed (FIFO)
  // and their loops completed.
  for (unsigned i = 0, e = instGroupQueue.size(); i < e; ++i) {
    uint64_t ubShift = (instGroupQueue[i].first + tripCount) * step;
    epilogue = generateLoop(b.getShiftedAffineMap(origLbMap, lbShift),
                            b.getShiftedAffineMap(origLbMap, ubShift),
                            instGroupQueue, i, forOp, b);
    lbShift = ubShift;
    if (!prologue)
      prologue = epilogue;
  }

  // Erase the original for op.
  forOp.erase();

  if (unrollPrologueEpilogue && prologue)
    loopUnrollFull(prologue);
  if (unrollPrologueEpilogue && !epilogue &&
      epilogue.getOperation() != prologue.getOperation())
    loopUnrollFull(epilogue);

  return success();
}

// Collect perfectly nested loops starting from `rootForOps`.  Loops are
// perfectly nested if each loop is the first and only non-terminator operation
// in the parent loop.  Collect at most `maxLoops` loops and append them to
// `forOps`.
template <typename T>
void getPerfectlyNestedLoopsImpl(
    SmallVectorImpl<T> &forOps, T rootForOp,
    unsigned maxLoops = std::numeric_limits<unsigned>::max()) {
  for (unsigned i = 0; i < maxLoops; ++i) {
    forOps.push_back(rootForOp);
    // FIXME: ForOp and AffineForOp currently provide different names to access
    // the region ("region" and "getRegion").  Remove this generic access when
    // AffineForOp moves to ODS and also gets "region".
    Block &body = rootForOp.getOperation()->getRegion(0).front();
    if (body.begin() != std::prev(body.end(), 2))
      return;

    rootForOp = dyn_cast<T>(&body.front());
    if (!rootForOp)
      return;
  }
}

/// Get perfectly nested sequence of loops starting at root of loop nest
/// (the first op being another AffineFor, and the second op - a terminator).
/// A loop is perfectly nested iff: the first op in the loop's body is another
/// AffineForOp, and the second op is a terminator).
void mlir::getPerfectlyNestedLoops(SmallVectorImpl<AffineForOp> &nestedLoops,
                                   AffineForOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}

void mlir::getPerfectlyNestedLoops(SmallVectorImpl<loop::ForOp> &nestedLoops,
                                   loop::ForOp root) {
  getPerfectlyNestedLoopsImpl(nestedLoops, root);
}

/// Unrolls this loop completely.
LogicalResult mlir::loopUnrollFull(AffineForOp forOp) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  if (mayBeConstantTripCount.hasValue()) {
    uint64_t tripCount = mayBeConstantTripCount.getValue();
    if (tripCount == 1) {
      return promoteIfSingleIteration(forOp);
    }
    return loopUnrollByFactor(forOp, tripCount);
  }
  return failure();
}

/// Unrolls and jams this loop by the specified factor or by the trip count (if
/// constant) whichever is lower.
LogicalResult mlir::loopUnrollUpToFactor(AffineForOp forOp,
                                         uint64_t unrollFactor) {
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);

  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollFactor)
    return loopUnrollByFactor(forOp, mayBeConstantTripCount.getValue());
  return loopUnrollByFactor(forOp, unrollFactor);
}

/// Unrolls this loop by the specified factor. Returns success if the loop
/// is successfully unrolled.
LogicalResult mlir::loopUnrollByFactor(AffineForOp forOp,
                                       uint64_t unrollFactor) {
  assert(unrollFactor >= 1 && "unroll factor should be >= 1");

  if (unrollFactor == 1)
    return promoteIfSingleIteration(forOp);

  if (forOp.getBody()->empty() ||
      forOp.getBody()->begin() == std::prev(forOp.getBody()->end()))
    return failure();

  // Loops where the lower bound is a max expression isn't supported for
  // unrolling since the trip count can be expressed as an affine function when
  // both the lower bound and the upper bound are multi-result maps. However,
  // one meaningful way to do such unrolling would be to specialize the loop for
  // the 'hotspot' case and unroll that hotspot.
  if (forOp.getLowerBoundMap().getNumResults() != 1)
    return failure();

  // If the trip count is lower than the unroll factor, no unrolled body.
  // TODO(bondhugula): option to specify cleanup loop unrolling.
  Optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(forOp);
  if (mayBeConstantTripCount.hasValue() &&
      mayBeConstantTripCount.getValue() < unrollFactor)
    return failure();

  // Generate the cleanup loop if trip count isn't a multiple of unrollFactor.
  Operation *op = forOp.getOperation();
  if (getLargestDivisorOfTripCount(forOp) % unrollFactor != 0) {
    OpBuilder builder(op->getBlock(), ++Block::iterator(op));
    auto cleanupForInst = cast<AffineForOp>(builder.clone(*op));
    AffineMap cleanupMap;
    SmallVector<Value *, 4> cleanupOperands;
    getCleanupLoopLowerBound(forOp, unrollFactor, &cleanupMap, &cleanupOperands,
                             builder);
    assert(cleanupMap &&
           "cleanup loop lower bound map for single result lower bound maps "
           "can always be determined");
    cleanupForInst.setLowerBound(cleanupOperands, cleanupMap);
    // Promote the loop body up if this has turned into a single iteration loop.
    promoteIfSingleIteration(cleanupForInst);

    // Adjust upper bound of the original loop; this is the same as the lower
    // bound of the cleanup loop.
    forOp.setUpperBound(cleanupOperands, cleanupMap);
  }

  // Scale the step of loop being unrolled by unroll factor.
  int64_t step = forOp.getStep();
  forOp.setStep(step * unrollFactor);

  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  OpBuilder builder = forOp.getBodyBuilder();

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(forOp.getBody()->end(), 2);

  // Unroll the contents of 'forOp' (append unrollFactor-1 additional copies).
  auto *forOpIV = forOp.getInductionVar();
  for (unsigned i = 1; i < unrollFactor; i++) {
    BlockAndValueMapping operandMap;

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV->use_empty()) {
      // iv' = iv + 1/2/3...unrollFactor-1;
      auto d0 = builder.getAffineDimExpr(0);
      auto bumpMap = builder.getAffineMap(1, 0, {d0 + i * step});
      auto ivUnroll =
          builder.create<AffineApplyOp>(forOp.getLoc(), bumpMap, forOpIV);
      operandMap.map(forOpIV, ivUnroll);
    }

    // Clone the original body of 'forOp'.
    for (auto it = forOp.getBody()->begin(); it != std::next(srcBlockEnd);
         it++) {
      builder.clone(*it, operandMap);
    }
  }

  // Promote the loop body up if this has turned into a single iteration loop.
  promoteIfSingleIteration(forOp);
  return success();
}

/// Performs loop interchange on 'forOpA' and 'forOpB', where 'forOpB' is
/// nested within 'forOpA' as the only non-terminator operation in its block.
void mlir::interchangeLoops(AffineForOp forOpA, AffineForOp forOpB) {
  auto *forOpAInst = forOpA.getOperation();

  assert(&*forOpA.getBody()->begin() == forOpB.getOperation());
  auto &forOpABody = forOpA.getBody()->getOperations();
  auto &forOpBBody = forOpB.getBody()->getOperations();

  // 1) Splice forOpA's non-terminator operations (which is just forOpB) just
  // before forOpA (in ForOpA's parent's block) this should leave 'forOpA's
  // body containing only the terminator.
  forOpAInst->getBlock()->getOperations().splice(Block::iterator(forOpAInst),
                                                 forOpABody, forOpABody.begin(),
                                                 std::prev(forOpABody.end()));
  // 2) Splice forOpB's non-terminator operations into the beginning of forOpA's
  // body (this leaves forOpB's body containing only the terminator).
  forOpABody.splice(forOpABody.begin(), forOpBBody, forOpBBody.begin(),
                    std::prev(forOpBBody.end()));
  // 3) Splice forOpA into the beginning of forOpB's body.
  forOpBBody.splice(forOpBBody.begin(), forOpAInst->getBlock()->getOperations(),
                    Block::iterator(forOpAInst));
}

// Checks each dependence component against the permutation to see if the
// desired loop interchange would violate dependences by making the
// dependence componenent lexicographically negative.
static bool checkLoopInterchangeDependences(
    const std::vector<llvm::SmallVector<DependenceComponent, 2>> &depCompsVec,
    ArrayRef<AffineForOp> loops, ArrayRef<unsigned> loopPermMap) {
  // Invert permutation map.
  unsigned maxLoopDepth = loops.size();
  llvm::SmallVector<unsigned, 4> loopPermMapInv;
  loopPermMapInv.resize(maxLoopDepth);
  for (unsigned i = 0; i < maxLoopDepth; ++i)
    loopPermMapInv[loopPermMap[i]] = i;

  // Check each dependence component against the permutation to see if the
  // desired loop interchange permutation would make the dependence vectors
  // lexicographically negative.
  // Example 1: [-1, 1][0, 0]
  // Example 2: [0, 0][-1, 1]
  for (unsigned i = 0, e = depCompsVec.size(); i < e; ++i) {
    const llvm::SmallVector<DependenceComponent, 2> &depComps = depCompsVec[i];
    assert(depComps.size() >= maxLoopDepth);
    // Check if the first non-zero dependence component is positive.
    // This iterates through loops in the desired order.
    for (unsigned j = 0; j < maxLoopDepth; ++j) {
      unsigned permIndex = loopPermMapInv[j];
      assert(depComps[permIndex].lb.hasValue());
      int64_t depCompLb = depComps[permIndex].lb.getValue();
      if (depCompLb > 0)
        break;
      if (depCompLb < 0)
        return false;
    }
  }
  return true;
}

/// Checks if the loop interchange permutation 'loopPermMap' of the perfectly
/// nested sequence of loops in 'loops' would violate dependences.
bool mlir::isValidLoopInterchangePermutation(ArrayRef<AffineForOp> loops,
                                             ArrayRef<unsigned> loopPermMap) {
  // Gather dependence components for dependences between all ops in loop nest
  // rooted at 'loops[0]', at loop depths in range [1, maxLoopDepth].
  assert(loopPermMap.size() == loops.size());
  unsigned maxLoopDepth = loops.size();
  std::vector<llvm::SmallVector<DependenceComponent, 2>> depCompsVec;
  getDependenceComponents(loops[0], maxLoopDepth, &depCompsVec);
  return checkLoopInterchangeDependences(depCompsVec, loops, loopPermMap);
}

/// Performs a sequence of loop interchanges of loops in perfectly nested
/// sequence of loops in 'loops', as specified by permutation in 'loopPermMap'.
unsigned mlir::interchangeLoops(ArrayRef<AffineForOp> loops,
                                ArrayRef<unsigned> loopPermMap) {
  Optional<unsigned> loopNestRootIndex;
  for (int i = loops.size() - 1; i >= 0; --i) {
    int permIndex = static_cast<int>(loopPermMap[i]);
    // Store the index of the for loop which will be the new loop nest root.
    if (permIndex == 0)
      loopNestRootIndex = i;
    if (permIndex > i) {
      // Sink loop 'i' by 'permIndex - i' levels deeper into the loop nest.
      sinkLoop(loops[i], permIndex - i);
    }
  }
  assert(loopNestRootIndex.hasValue());
  return loopNestRootIndex.getValue();
}

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
AffineForOp mlir::sinkSequentialLoops(AffineForOp forOp) {
  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, forOp);
  if (loops.size() < 2)
    return forOp;

  // Gather dependence components for dependences between all ops in loop nest
  // rooted at 'loops[0]', at loop depths in range [1, maxLoopDepth].
  unsigned maxLoopDepth = loops.size();
  std::vector<llvm::SmallVector<DependenceComponent, 2>> depCompsVec;
  getDependenceComponents(loops[0], maxLoopDepth, &depCompsVec);

  // Mark loops as either parallel or sequential.
  llvm::SmallVector<bool, 8> isParallelLoop(maxLoopDepth, true);
  for (unsigned i = 0, e = depCompsVec.size(); i < e; ++i) {
    llvm::SmallVector<DependenceComponent, 2> &depComps = depCompsVec[i];
    assert(depComps.size() >= maxLoopDepth);
    for (unsigned j = 0; j < maxLoopDepth; ++j) {
      DependenceComponent &depComp = depComps[j];
      assert(depComp.lb.hasValue() && depComp.ub.hasValue());
      if (depComp.lb.getValue() != 0 || depComp.ub.getValue() != 0)
        isParallelLoop[j] = false;
    }
  }

  // Count the number of parallel loops.
  unsigned numParallelLoops = 0;
  for (unsigned i = 0, e = isParallelLoop.size(); i < e; ++i)
    if (isParallelLoop[i])
      ++numParallelLoops;

  // Compute permutation of loops that sinks sequential loops (and thus raises
  // parallel loops) while preserving relative order.
  llvm::SmallVector<unsigned, 4> loopPermMap(maxLoopDepth);
  unsigned nextSequentialLoop = numParallelLoops;
  unsigned nextParallelLoop = 0;
  for (unsigned i = 0; i < maxLoopDepth; ++i) {
    if (isParallelLoop[i]) {
      loopPermMap[i] = nextParallelLoop++;
    } else {
      loopPermMap[i] = nextSequentialLoop++;
    }
  }

  // Check if permutation 'loopPermMap' would violate dependences.
  if (!checkLoopInterchangeDependences(depCompsVec, loops, loopPermMap))
    return forOp;
  // Perform loop interchange according to permutation 'loopPermMap'.
  unsigned loopNestRootIndex = interchangeLoops(loops, loopPermMap);
  return loops[loopNestRootIndex];
}

/// Performs a series of loop interchanges to sink 'forOp' 'loopDepth' levels
/// deeper in the loop nest.
void mlir::sinkLoop(AffineForOp forOp, unsigned loopDepth) {
  for (unsigned i = 0; i < loopDepth; ++i) {
    AffineForOp nextForOp = cast<AffineForOp>(forOp.getBody()->front());
    interchangeLoops(forOp, nextForOp);
  }
}

// Factors out common behavior to add a new `iv` (resp. `iv` + `offset`) to the
// lower (resp. upper) loop bound. When called for both the lower and upper
// bounds, the resulting IR resembles:
//
// ```mlir
//    affine.for %i = max (`iv, ...) to min (`iv` + `offset`) {
//      ...
//    }
// ```
static void augmentMapAndBounds(OpBuilder &b, Value *iv, AffineMap *map,
                                SmallVector<Value *, 4> *operands,
                                int64_t offset = 0) {
  auto bounds = llvm::to_vector<4>(map->getResults());
  bounds.push_back(b.getAffineDimExpr(map->getNumDims()) + offset);
  operands->insert(operands->begin() + map->getNumDims(), iv);
  *map = b.getAffineMap(map->getNumDims() + 1, map->getNumSymbols(), bounds);
  canonicalizeMapAndOperands(map, operands);
}

// Clone the original body of `forOp` into the body of `newForOp` while
// substituting `oldIv` in place of
// `forOp.getInductionVariable()` and ignoring the terminator.
// Note: `newForOp` may be nested under `forOp`.
template <typename ForOpType>
void cloneLoopBodyInto(ForOpType forOp, Value *oldIv, ForOpType newForOp) {
  BlockAndValueMapping map;
  map.map(oldIv, newForOp.getInductionVar());
  OpBuilder b = newForOp.getBodyBuilder();
  for (auto &op : *forOp.getBody()) {
    // Step over newForOp in case it is nested under forOp.
    if (&op == newForOp.getOperation()) {
      continue;
    }
    if (op.isKnownTerminator()) {
      continue;
    }
    auto *instClone = b.clone(op, map);
    unsigned idx = 0;
    for (auto r : op.getResults()) {
      // Since we do a forward pass over the body, we iteratively augment
      // the `map` with everything we clone.
      map.map(r, instClone->getResult(idx++));
    }
  }
}

// Stripmines `forOp` by `factor` and sinks it under each of the `targets`.
// Stripmine-sink is a primitive building block for generalized tiling of
// imperfectly nested loops.
// This transformation is purely mechanical and does not check legality,
// profitability or even structural correctness. It is the user's
// responsibility to specify `targets` that are dominated by `forOp`.
// Returns the new AffineForOps, one per `targets`, nested immediately under
// each of the `targets`.
static SmallVector<AffineForOp, 8>
stripmineSink(AffineForOp forOp, uint64_t factor,
              ArrayRef<AffineForOp> targets) {
  auto originalStep = forOp.getStep();
  auto scaledStep = originalStep * factor;
  forOp.setStep(scaledStep);

  auto *op = forOp.getOperation();
  OpBuilder b(op->getBlock(), ++Block::iterator(op));

  // Lower-bound map creation.
  auto lbMap = forOp.getLowerBoundMap();
  SmallVector<Value *, 4> lbOperands(forOp.getLowerBoundOperands());
  augmentMapAndBounds(b, forOp.getInductionVar(), &lbMap, &lbOperands);

  // Upper-bound map creation.
  auto ubMap = forOp.getUpperBoundMap();
  SmallVector<Value *, 4> ubOperands(forOp.getUpperBoundOperands());
  augmentMapAndBounds(b, forOp.getInductionVar(), &ubMap, &ubOperands,
                      /*offset=*/scaledStep);

  SmallVector<AffineForOp, 8> innerLoops;
  for (auto t : targets) {
    // Insert newForOp before the terminator of `t`.
    OpBuilder b = t.getBodyBuilder();
    auto newForOp = b.create<AffineForOp>(t.getLoc(), lbOperands, lbMap,
                                          ubOperands, ubMap, originalStep);
    cloneLoopBodyInto(t, forOp.getInductionVar(), newForOp);
    // Remove all operations from `t` except `newForOp`.
    auto rit = ++newForOp.getOperation()->getReverseIterator();
    auto re = t.getBody()->rend();
    for (auto &op : llvm::make_early_inc_range(llvm::make_range(rit, re))) {
      op.erase();
    }
    innerLoops.push_back(newForOp);
  }

  return innerLoops;
}

static Loops stripmineSink(loop::ForOp forOp, Value *factor,
                           ArrayRef<loop::ForOp> targets) {
  auto *originalStep = forOp.step();
  auto *iv = forOp.getInductionVar();

  OpBuilder b(forOp);
  forOp.setStep(b.create<MulIOp>(forOp.getLoc(), originalStep, factor));

  Loops innerLoops;
  for (auto t : targets) {
    // Save information for splicing ops out of t when done
    auto begin = t.getBody()->begin();
    auto nOps = t.getBody()->getOperations().size();

    // Insert newForOp before the terminator of `t`.
    OpBuilder b(t.getBodyBuilder());
    Value *stepped = b.create<AddIOp>(t.getLoc(), iv, forOp.step());
    Value *less = b.create<CmpIOp>(t.getLoc(), CmpIPredicate::SLT,
                                   forOp.upperBound(), stepped);
    Value *ub =
        b.create<SelectOp>(t.getLoc(), less, forOp.upperBound(), stepped);

    // Splice [begin, begin + nOps - 1) into `newForOp` and replace uses.
    auto newForOp = b.create<loop::ForOp>(t.getLoc(), iv, ub, originalStep);
    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        t.getBody()->getOperations(), begin, std::next(begin, nOps - 1));
    replaceAllUsesInRegionWith(iv, newForOp.getInductionVar(),
                               newForOp.region());

    innerLoops.push_back(newForOp);
  }

  return innerLoops;
}

// Stripmines a `forOp` by `factor` and sinks it under a single `target`.
// Returns the new AffineForOps, nested immediately under `target`.
template <typename ForType, typename SizeType>
static ForType stripmineSink(ForType forOp, SizeType factor, ForType target) {
  // TODO(ntv): Use cheap structural assertions that targets are nested under
  // forOp and that targets are not nested under each other when DominanceInfo
  // exposes the capability. It seems overkill to construct a whole function
  // dominance tree at this point.
  auto res = stripmineSink(forOp, factor, ArrayRef<ForType>{target});
  assert(res.size() == 1 && "Expected 1 inner forOp");
  return res[0];
}

template <typename ForType, typename SizeType>
static SmallVector<SmallVector<ForType, 8>, 8>
tileImpl(ArrayRef<ForType> forOps, ArrayRef<SizeType> sizes,
         ArrayRef<ForType> targets) {
  SmallVector<SmallVector<ForType, 8>, 8> res;
  SmallVector<ForType, 8> currentTargets(targets.begin(), targets.end());
  for (auto it : llvm::zip(forOps, sizes)) {
    auto step = stripmineSink(std::get<0>(it), std::get<1>(it), currentTargets);
    res.push_back(step);
    currentTargets = step;
  }
  return res;
}

SmallVector<SmallVector<AffineForOp, 8>, 8>
mlir::tile(ArrayRef<AffineForOp> forOps, ArrayRef<uint64_t> sizes,
           ArrayRef<AffineForOp> targets) {
  return tileImpl(forOps, sizes, targets);
}

SmallVector<Loops, 8> mlir::tile(ArrayRef<loop::ForOp> forOps,
                                 ArrayRef<Value *> sizes,
                                 ArrayRef<loop::ForOp> targets) {
  return tileImpl(forOps, sizes, targets);
}

template <typename ForType, typename SizeType>
static SmallVector<ForType, 8>
tileImpl(ArrayRef<ForType> forOps, ArrayRef<SizeType> sizes, ForType target) {
  SmallVector<ForType, 8> res;
  for (auto loops : tile(forOps, sizes, ArrayRef<ForType>{target})) {
    assert(loops.size() == 1);
    res.push_back(loops[0]);
  }
  return res;
}

SmallVector<AffineForOp, 8> mlir::tile(ArrayRef<AffineForOp> forOps,
                                       ArrayRef<uint64_t> sizes,
                                       AffineForOp target) {
  return tileImpl(forOps, sizes, target);
}

Loops mlir::tile(ArrayRef<loop::ForOp> forOps, ArrayRef<Value *> sizes,
                 loop::ForOp target) {
  return tileImpl(forOps, sizes, target);
}

Loops mlir::tilePerfectlyNested(loop::ForOp rootForOp,
                                ArrayRef<Value *> sizes) {
  // Collect prefectly nested loops.  If more size values provided than nested
  // loops available, truncate `sizes`.
  SmallVector<loop::ForOp, 4> forOps;
  forOps.reserve(sizes.size());
  getPerfectlyNestedLoopsImpl(forOps, rootForOp, sizes.size());
  if (forOps.size() < sizes.size())
    sizes = sizes.take_front(forOps.size());

  return ::tile(forOps, sizes, forOps.back());
}

// Build the IR that performs ceil division of a positive value by a constant:
//    ceildiv(a, B) = divis(a + (B-1), B)
// where divis is roundning-to-zero division.
static Value *ceilDivPositive(OpBuilder &builder, Location loc, Value *dividend,
                              int64_t divisor) {
  assert(divisor > 0 && "expected positive divisor");
  assert(dividend->getType().isIndex() && "expected index-typed value");

  Value *divisorMinusOneCst = builder.create<ConstantIndexOp>(loc, divisor - 1);
  Value *divisorCst = builder.create<ConstantIndexOp>(loc, divisor);
  Value *sum = builder.create<AddIOp>(loc, dividend, divisorMinusOneCst);
  return builder.create<DivISOp>(loc, sum, divisorCst);
}

// Build the IR that performs ceil division of a positive value by another
// positive value:
//    ceildiv(a, b) = divis(a + (b - 1), b)
// where divis is rounding-to-zero division.
static Value *ceilDivPositive(OpBuilder &builder, Location loc, Value *dividend,
                              Value *divisor) {
  assert(dividend->getType().isIndex() && "expected index-typed value");

  Value *cstOne = builder.create<ConstantIndexOp>(loc, 1);
  Value *divisorMinusOne = builder.create<SubIOp>(loc, divisor, cstOne);
  Value *sum = builder.create<AddIOp>(loc, dividend, divisorMinusOne);
  return builder.create<DivISOp>(loc, sum, divisor);
}

TileLoops mlir::extractFixedOuterLoops(loop::ForOp rootForOp,
                                       ArrayRef<int64_t> sizes) {
  // Collect prefectly nested loops.  If more size values provided than nested
  // loops available, truncate `sizes`.
  SmallVector<loop::ForOp, 4> forOps;
  forOps.reserve(sizes.size());
  getPerfectlyNestedLoopsImpl(forOps, rootForOp, sizes.size());
  if (forOps.size() < sizes.size())
    sizes = sizes.take_front(forOps.size());

  // Compute the tile sizes such that i-th outer loop executes size[i]
  // iterations.  Given that the loop current executes
  //   numIterations = ceildiv((upperBound - lowerBound), step)
  // iterations, we need to tile with size ceildiv(numIterations, size[i]).
  SmallVector<Value *, 4> tileSizes;
  tileSizes.reserve(sizes.size());
  for (unsigned i = 0, e = sizes.size(); i < e; ++i) {
    assert(sizes[i] > 0 && "expected strictly positive size for strip-mining");

    auto forOp = forOps[i];
    OpBuilder builder(forOp);
    auto loc = forOp.getLoc();
    Value *diff =
        builder.create<SubIOp>(loc, forOp.upperBound(), forOp.lowerBound());
    Value *numIterations = ceilDivPositive(builder, loc, diff, forOp.step());
    Value *iterationsPerBlock =
        ceilDivPositive(builder, loc, numIterations, sizes[i]);
    tileSizes.push_back(iterationsPerBlock);
  }

  // Call parametric tiling with the given sizes.
  auto intraTile = tile(forOps, tileSizes, forOps.back());
  return std::make_pair(forOps, intraTile);
}

// Replaces all uses of `orig` with `replacement` except if the user is listed
// in `exceptions`.
static void
replaceAllUsesExcept(Value *orig, Value *replacement,
                     const SmallPtrSetImpl<Operation *> &exceptions) {
  for (auto &use : orig->getUses()) {
    if (exceptions.count(use.getOwner()) == 0)
      use.set(replacement);
  }
}

// Transform a loop with a strictly positive step
//   for %i = %lb to %ub step %s
// into a 0-based loop with step 1
//   for %ii = 0 to ceildiv(%ub - %lb, %s) step 1 {
//     %i = %ii * %s + %lb
// Insert the induction variable remapping in the body of `inner`, which is
// expected to be either `loop` or another loop perfectly nested under `loop`.
// Insert the definition of new bounds immediate before `outer`, which is
// expected to be either `loop` or its parent in the loop nest.
static void normalizeLoop(loop::ForOp loop, loop::ForOp outer,
                          loop::ForOp inner) {
  OpBuilder builder(outer);
  Location loc = loop.getLoc();

  // Check if the loop is already known to have a constant zero lower bound or
  // a constant one step.
  bool isZeroBased = false;
  if (auto ubCst =
          dyn_cast_or_null<ConstantIndexOp>(loop.lowerBound()->getDefiningOp()))
    isZeroBased = ubCst.getValue() == 0;

  bool isStepOne = false;
  if (auto stepCst =
          dyn_cast_or_null<ConstantIndexOp>(loop.step()->getDefiningOp()))
    isStepOne = stepCst.getValue() == 1;

  if (isZeroBased && isStepOne)
    return;

  // Compute the number of iterations the loop executes: ceildiv(ub - lb, step)
  // assuming the step is strictly positive.  Update the bounds and the step
  // of the loop to go from 0 to the number of iterations, if necessary.
  // TODO(zinenko): introduce support for negative steps or emit dynamic asserts
  // on step positivity, whatever gets implemented first.
  Value *diff =
      builder.create<SubIOp>(loc, loop.upperBound(), loop.lowerBound());
  Value *numIterations = ceilDivPositive(builder, loc, diff, loop.step());
  loop.setUpperBound(numIterations);

  Value *lb = loop.lowerBound();
  if (!isZeroBased) {
    Value *cst0 = builder.create<ConstantIndexOp>(loc, 0);
    loop.setLowerBound(cst0);
  }

  Value *step = loop.step();
  if (!isStepOne) {
    Value *cst1 = builder.create<ConstantIndexOp>(loc, 1);
    loop.setStep(cst1);
  }

  // Insert code computing the value of the original loop induction variable
  // from the "normalized" one.
  builder.setInsertionPointToStart(inner.getBody());
  Value *scaled =
      isStepOne ? loop.getInductionVar()
                : builder.create<MulIOp>(loc, loop.getInductionVar(), step);
  Value *shifted =
      isZeroBased ? scaled : builder.create<AddIOp>(loc, scaled, lb);

  SmallPtrSet<Operation *, 2> preserve{scaled->getDefiningOp(),
                                       shifted->getDefiningOp()};
  replaceAllUsesExcept(loop.getInductionVar(), shifted, preserve);
}

void mlir::coalesceLoops(MutableArrayRef<loop::ForOp> loops) {
  if (loops.size() < 2)
    return;

  loop::ForOp innermost = loops.back();
  loop::ForOp outermost = loops.front();

  // 1. Make sure all loops iterate from 0 to upperBound with step 1.  This
  // allows the following code to assume upperBound is the number of iterations.
  for (auto loop : loops)
    normalizeLoop(loop, outermost, innermost);

  // 2. Emit code computing the upper bound of the coalesced loop as product
  // of the number of iterations of all loops.
  OpBuilder builder(outermost);
  Location loc = outermost.getLoc();
  Value *upperBound = outermost.upperBound();
  for (auto loop : loops.drop_front())
    upperBound = builder.create<MulIOp>(loc, upperBound, loop.upperBound());
  outermost.setUpperBound(upperBound);

  builder.setInsertionPointToStart(outermost.getBody());

  // 3. Remap induction variables.  For each original loop, the value of the
  // induction variable can be obtained by dividing the induction variable of
  // the linearized loop by the total number of iterations of the loops nested
  // in it modulo the number of iterations in this loop (remove the values
  // related to the outer loops):
  //   iv_i = floordiv(iv_linear, product-of-loop-ranges-until-i) mod range_i.
  // Compute these iteratively from the innermost loop by creating a "running
  // quotient" of division by the range.
  Value *previous = outermost.getInductionVar();
  for (unsigned i = 0, e = loops.size(); i < e; ++i) {
    unsigned idx = loops.size() - i - 1;
    if (i != 0)
      previous =
          builder.create<DivISOp>(loc, previous, loops[idx + 1].upperBound());

    Value *iv = (i == e - 1) ? previous
                             : builder.create<RemISOp>(loc, previous,
                                                       loops[idx].upperBound());
    replaceAllUsesInRegionWith(loops[idx].getInductionVar(), iv,
                               loops.back().region());
  }

  // 4. Move the operations from the innermost just above the second-outermost
  // loop, delete the extra terminator and the second-outermost loop.
  loop::ForOp second = loops[1];
  innermost.getBody()->back().erase();
  outermost.getBody()->getOperations().splice(
      Block::iterator(second.getOperation()),
      innermost.getBody()->getOperations());
  second.erase();
}
