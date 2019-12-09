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

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "LoopUtils"

using namespace mlir;
using llvm::SetVector;
using llvm::SmallMapVector;

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
  auto lb = b.create<AffineApplyOp>(forOp.getLoc(), lbMap,
                                    forOp.getLowerBoundOperands());

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
    auto bumpMap = AffineMap::get(tripCountMap.getNumDims(),
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
  *map = AffineMap::get(1 + tripCountMap.getNumResults(), 0, newUbExprs);
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
  f.walk([](AffineForOp forOp) { promoteIfSingleIteration(forOp); });
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
// memory-based dependence preservation check rests with the users of this
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
    Block &body = rootForOp.region().front();
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
      auto bumpMap = AffineMap::get(1, 0, {d0 + i * step});
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
// dependence component lexicographically negative.
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
  *map = AffineMap::get(map->getNumDims() + 1, map->getNumSymbols(), bounds);
  canonicalizeMapAndOperands(map, operands);
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

  auto *iv = forOp.getInductionVar();
  SmallVector<AffineForOp, 8> innerLoops;
  for (auto t : targets) {
    // Insert newForOp before the terminator of `t`.
    OpBuilder b = t.getBodyBuilder();
    auto newForOp = b.create<AffineForOp>(t.getLoc(), lbOperands, lbMap,
                                          ubOperands, ubMap, originalStep);
    auto begin = t.getBody()->begin();
    // Skip terminator and `newForOp` which is just before the terminator.
    auto nOps = t.getBody()->getOperations().size() - 2;
    newForOp.getBody()->getOperations().splice(
        newForOp.getBody()->getOperations().begin(),
        t.getBody()->getOperations(), begin, std::next(begin, nOps));
    replaceAllUsesInRegionWith(iv, newForOp.getInductionVar(),
                               newForOp.region());
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
    Value *less = b.create<CmpIOp>(t.getLoc(), CmpIPredicate::slt,
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
  // Collect perfectly nested loops.  If more size values provided than nested
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
// where divis is rounding-to-zero division.
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

// Hoist the ops within `outer` that appear before `inner`.
// Such ops include the ops that have been introduced by parametric tiling.
// Ops that come from triangular loops (i.e. that belong to the program slice
// rooted at `outer`) and ops that have side effects cannot be hoisted.
// Return failure when any op fails to hoist.
static LogicalResult hoistOpsBetween(loop::ForOp outer, loop::ForOp inner) {
  SetVector<Operation *> forwardSlice;
  getForwardSlice(outer.getOperation(), &forwardSlice, [&inner](Operation *op) {
    return op != inner.getOperation();
  });
  LogicalResult status = success();
  SmallVector<Operation *, 8> toHoist;
  for (auto &op : outer.getBody()->getOperations()) {
    // Stop when encountering the inner loop.
    if (&op == inner.getOperation())
      break;
    // Skip over non-hoistable ops.
    if (forwardSlice.count(&op) > 0) {
      status = failure();
      continue;
    }
    // Skip loop::ForOp, these are not considered a failure.
    if (op.getNumRegions() > 0)
      continue;
    // Skip other ops with regions.
    if (op.getNumRegions() > 0) {
      status = failure();
      continue;
    }
    // Skip if op has side effects.
    // TODO(ntv): loads to immutable memory regions are ok.
    if (!op.hasNoSideEffect()) {
      status = failure();
      continue;
    }
    toHoist.push_back(&op);
  }
  auto *outerForOp = outer.getOperation();
  for (auto *op : toHoist)
    op->moveBefore(outerForOp);
  return status;
}

// Traverse the interTile and intraTile loops and try to hoist ops such that
// bands of perfectly nested loops are isolated.
// Return failure if either perfect interTile or perfect intraTile bands cannot
// be formed.
static LogicalResult tryIsolateBands(const TileLoops &tileLoops) {
  LogicalResult status = success();
  auto &interTile = tileLoops.first;
  auto &intraTile = tileLoops.second;
  auto size = interTile.size();
  assert(size == intraTile.size());
  if (size <= 1)
    return success();
  for (unsigned s = 1; s < size; ++s)
    status = succeeded(status) ? hoistOpsBetween(intraTile[0], intraTile[s])
                               : failure();
  for (unsigned s = 1; s < size; ++s)
    status = succeeded(status) ? hoistOpsBetween(interTile[0], interTile[s])
                               : failure();
  return status;
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
  TileLoops tileLoops = std::make_pair(forOps, intraTile);

  // TODO(ntv, zinenko) for now we just ignore the result of band isolation.
  // In the future, mapping decisions may be impacted by the ability to
  // isolate perfectly nested bands.
  tryIsolateBands(tileLoops);

  return tileLoops;
}

// Replaces all uses of `orig` with `replacement` except if the user is listed
// in `exceptions`.
static void
replaceAllUsesExcept(Value *orig, Value *replacement,
                     const SmallPtrSetImpl<Operation *> &exceptions) {
  for (auto &use : llvm::make_early_inc_range(orig->getUses())) {
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

void mlir::mapLoopToProcessorIds(loop::ForOp forOp,
                                 ArrayRef<Value *> processorId,
                                 ArrayRef<Value *> numProcessors) {
  assert(processorId.size() == numProcessors.size());
  if (processorId.empty())
    return;

  OpBuilder b(forOp);
  Location loc(forOp.getLoc());
  Value *mul = processorId.front();
  for (unsigned i = 1, e = processorId.size(); i < e; ++i)
    mul = b.create<AddIOp>(loc, b.create<MulIOp>(loc, mul, numProcessors[i]),
                           processorId[i]);
  Value *lb = b.create<AddIOp>(loc, forOp.lowerBound(),
                               b.create<MulIOp>(loc, forOp.step(), mul));
  forOp.setLowerBound(lb);

  Value *step = forOp.step();
  for (auto *numProcs : numProcessors)
    step = b.create<MulIOp>(loc, step, numProcs);
  forOp.setStep(step);
}

/// Given a memref region, determine the lowest depth at which transfers can be
/// placed for it, and return the corresponding block, start and end positions
/// in the block for placing incoming (read) and outgoing (write) copies
/// respectively. The lowest depth depends on whether the region being accessed
/// is hoistable with respect to one or more immediately surrounding loops.
static void
findHighestBlockForPlacement(const MemRefRegion &region, Block &block,
                             Block::iterator &begin, Block::iterator &end,
                             Block **copyPlacementBlock,
                             Block::iterator *copyInPlacementStart,
                             Block::iterator *copyOutPlacementStart) {
  const auto *cst = region.getConstraints();
  SmallVector<Value *, 4> symbols;
  cst->getIdValues(cst->getNumDimIds(), cst->getNumDimAndSymbolIds(), &symbols);

  SmallVector<AffineForOp, 4> enclosingFors;
  getLoopIVs(*block.begin(), &enclosingFors);
  // Walk up loop parents till we find an IV on which this region is
  // symbolic/variant.
  auto it = enclosingFors.rbegin();
  for (auto e = enclosingFors.rend(); it != e; ++it) {
    // TODO(bondhugula): also need to be checking this for regions symbols that
    // aren't loop IVs, whether we are within their resp. defs' dominance scope.
    if (llvm::is_contained(symbols, it->getInductionVar()))
      break;
  }

  if (it != enclosingFors.rbegin()) {
    auto lastInvariantIV = *std::prev(it);
    *copyInPlacementStart = Block::iterator(lastInvariantIV.getOperation());
    *copyOutPlacementStart = std::next(*copyInPlacementStart);
    *copyPlacementBlock = lastInvariantIV.getOperation()->getBlock();
  } else {
    *copyInPlacementStart = begin;
    *copyOutPlacementStart = end;
    *copyPlacementBlock = &block;
  }
}

// Info comprising stride and number of elements transferred every stride.
struct StrideInfo {
  int64_t stride;
  int64_t numEltPerStride;
};

/// Returns striding information for a copy/transfer of this region with
/// potentially multiple striding levels from outermost to innermost. For an
/// n-dimensional region, there can be at most n-1 levels of striding
/// successively nested.
//  TODO(bondhugula): make this work with non-identity layout maps.
static void getMultiLevelStrides(const MemRefRegion &region,
                                 ArrayRef<int64_t> bufferShape,
                                 SmallVectorImpl<StrideInfo> *strideInfos) {
  if (bufferShape.size() <= 1)
    return;

  int64_t numEltPerStride = 1;
  int64_t stride = 1;
  for (int d = bufferShape.size() - 1; d >= 1; d--) {
    int64_t dimSize = region.memref->getType().cast<MemRefType>().getDimSize(d);
    stride *= dimSize;
    numEltPerStride *= bufferShape[d];
    // A stride is needed only if the region has a shorter extent than the
    // memref along the dimension *and* has an extent greater than one along the
    // next major dimension.
    if (bufferShape[d] < dimSize && bufferShape[d - 1] > 1) {
      strideInfos->push_back({stride, numEltPerStride});
    }
  }
}

/// Generates a point-wise copy from/to `memref' to/from `fastMemRef' and
/// returns the outermost AffineForOp of the copy loop nest. `memIndicesStart'
/// holds the lower coordinates of the region in the original memref to copy
/// in/out. If `copyOut' is true, generates a copy-out; otherwise a copy-in.
static AffineForOp generatePointWiseCopy(Location loc, Value *memref,
                                         Value *fastMemRef,
                                         AffineMap memAffineMap,
                                         ArrayRef<Value *> memIndicesStart,
                                         ArrayRef<int64_t> fastBufferShape,
                                         bool isCopyOut, OpBuilder b) {
  assert(!memIndicesStart.empty() && "only 1-d or more memrefs");

  // The copy-in nest is generated as follows as an example for a 2-d region:
  // for x = ...
  //   for y = ...
  //     fast_buf[x][y] = buf[mem_x + x][mem_y + y]

  SmallVector<Value *, 4> fastBufIndices, memIndices;
  AffineForOp copyNestRoot;
  for (unsigned d = 0, e = fastBufferShape.size(); d < e; ++d) {
    auto forOp = b.create<AffineForOp>(loc, 0, fastBufferShape[d]);
    if (d == 0)
      copyNestRoot = forOp;
    b = forOp.getBodyBuilder();
    fastBufIndices.push_back(forOp.getInductionVar());

    Value *memBase =
        (memAffineMap == b.getMultiDimIdentityMap(memAffineMap.getNumDims()))
            ? memIndicesStart[d]
            : b.create<AffineApplyOp>(
                  loc,
                  AffineMap::get(memAffineMap.getNumDims(),
                                 memAffineMap.getNumSymbols(),
                                 memAffineMap.getResult(d)),
                  memIndicesStart);

    // Construct the subscript for the slow memref being copied.
    auto memIndex = b.create<AffineApplyOp>(
        loc,
        AffineMap::get(2, 0, b.getAffineDimExpr(0) + b.getAffineDimExpr(1)),
        ValueRange({memBase, forOp.getInductionVar()}));
    memIndices.push_back(memIndex);
  }

  if (!isCopyOut) {
    // Copy in.
    auto load = b.create<AffineLoadOp>(loc, memref, memIndices);
    b.create<AffineStoreOp>(loc, load, fastMemRef, fastBufIndices);
    return copyNestRoot;
  }

  // Copy out.
  auto load = b.create<AffineLoadOp>(loc, fastMemRef, fastBufIndices);
  b.create<AffineStoreOp>(loc, load, memref, memIndices);
  return copyNestRoot;
}

static InFlightDiagnostic LLVM_ATTRIBUTE_UNUSED
emitRemarkForBlock(Block &block) {
  return block.getParentOp()->emitRemark();
}

/// Creates a buffer in the faster memory space for the specified memref region;
/// generates a copy from the lower memory space to this one, and replaces all
/// loads/stores in the block range [`begin', `end') of `block' to load/store
/// from that buffer. Returns failure if copies could not be generated due to
/// yet unimplemented cases. `copyInPlacementStart` and `copyOutPlacementStart`
/// in copyPlacementBlock specify the insertion points where the incoming copies
/// and outgoing copies, respectively, should be inserted (the insertion happens
/// right before the insertion point). Since `begin` can itself be invalidated
/// due to the memref rewriting done from this method, the output argument
/// `nBegin` is set to its replacement (set to `begin` if no invalidation
/// happens). Since outgoing copies could have  been inserted at `end`, the
/// output argument `nEnd` is set to the new end. `sizeInBytes` is set to the
/// size of the fast buffer allocated.
static LogicalResult generateCopy(
    const MemRefRegion &region, Block *block, Block::iterator begin,
    Block::iterator end, Block *copyPlacementBlock,
    Block::iterator copyInPlacementStart, Block::iterator copyOutPlacementStart,
    AffineCopyOptions copyOptions, DenseMap<Value *, Value *> &fastBufferMap,
    DenseSet<Operation *> &copyNests, uint64_t *sizeInBytes,
    Block::iterator *nBegin, Block::iterator *nEnd) {
  *nBegin = begin;
  *nEnd = end;

  FuncOp f = begin->getParentOfType<FuncOp>();
  OpBuilder topBuilder(f.getBody());
  Value *zeroIndex = topBuilder.create<ConstantIndexOp>(f.getLoc(), 0);

  if (begin == end)
    return success();

  // Is the copy out point at the end of the block where we are doing
  // explicit copying.
  bool isCopyOutAtEndOfBlock = (end == copyOutPlacementStart);

  // Copies for read regions are going to be inserted at 'begin'.
  OpBuilder prologue(copyPlacementBlock, copyInPlacementStart);
  // Copies for write regions are going to be inserted at 'end'.
  OpBuilder epilogue(copyPlacementBlock, copyOutPlacementStart);
  OpBuilder &b = region.isWrite() ? epilogue : prologue;

  // Builder to create constants at the top level.
  auto func = copyPlacementBlock->getParent()->getParentOfType<FuncOp>();
  OpBuilder top(func.getBody());

  auto loc = region.loc;
  auto *memref = region.memref;
  auto memRefType = memref->getType().cast<MemRefType>();

  auto layoutMaps = memRefType.getAffineMaps();
  if (layoutMaps.size() > 1 ||
      (layoutMaps.size() == 1 && !layoutMaps[0].isIdentity())) {
    LLVM_DEBUG(llvm::dbgs() << "Non-identity layout map not yet supported\n");
    return failure();
  }

  // Indices to use for the copying.
  // Indices for the original memref being copied from/to.
  SmallVector<Value *, 4> memIndices;
  // Indices for the faster buffer being copied into/from.
  SmallVector<Value *, 4> bufIndices;

  unsigned rank = memRefType.getRank();
  SmallVector<int64_t, 4> fastBufferShape;

  // Compute the extents of the buffer.
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  Optional<int64_t> numElements = region.getConstantBoundingSizeAndShape(
      &fastBufferShape, &lbs, &lbDivisors);
  if (!numElements.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-constant region size not supported\n");
    return failure();
  }

  if (numElements.getValue() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Nothing to copy\n");
    *sizeInBytes = 0;
    return success();
  }

  const FlatAffineConstraints *cst = region.getConstraints();
  // 'regionSymbols' hold values that this memory region is symbolic/parametric
  // on; these typically include loop IVs surrounding the level at which the
  // copy generation is being done or other valid symbols in MLIR.
  SmallVector<Value *, 8> regionSymbols;
  cst->getIdValues(rank, cst->getNumIds(), &regionSymbols);

  // Construct the index expressions for the fast memory buffer. The index
  // expression for a particular dimension of the fast buffer is obtained by
  // subtracting out the lower bound on the original memref's data region
  // along the corresponding dimension.

  // Index start offsets for faster memory buffer relative to the original.
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);
  for (unsigned d = 0; d < rank; d++) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++) {
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    }
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);

    // Set copy start location for this dimension in the lower memory space
    // memref.
    if (auto caf = offset.dyn_cast<AffineConstantExpr>()) {
      auto indexVal = caf.getValue();
      if (indexVal == 0) {
        memIndices.push_back(zeroIndex);
      } else {
        memIndices.push_back(
            top.create<ConstantIndexOp>(loc, indexVal).getResult());
      }
    } else {
      // The coordinate for the start location is just the lower bound along the
      // corresponding dimension on the memory region (stored in 'offset').
      auto map = AffineMap::get(
          cst->getNumDimIds() + cst->getNumSymbolIds() - rank, 0, offset);
      memIndices.push_back(b.create<AffineApplyOp>(loc, map, regionSymbols));
    }
    // The fast buffer is copied into at location zero; addressing is relative.
    bufIndices.push_back(zeroIndex);

    // Record the offsets since they are needed to remap the memory accesses of
    // the original memref further below.
    offsets.push_back(offset);
  }

  // The faster memory space buffer.
  Value *fastMemRef;

  // Check if a buffer was already created.
  bool existingBuf = fastBufferMap.count(memref) > 0;
  if (!existingBuf) {
    AffineMap fastBufferLayout = b.getMultiDimIdentityMap(rank);
    auto fastMemRefType =
        MemRefType::get(fastBufferShape, memRefType.getElementType(),
                        fastBufferLayout, copyOptions.fastMemorySpace);

    // Create the fast memory space buffer just before the 'affine.for'
    // operation.
    fastMemRef = prologue.create<AllocOp>(loc, fastMemRefType).getResult();
    // Record it.
    fastBufferMap[memref] = fastMemRef;
    // fastMemRefType is a constant shaped memref.
    *sizeInBytes = getMemRefSizeInBytes(fastMemRefType).getValue();
    LLVM_DEBUG(emitRemarkForBlock(*block)
               << "Creating fast buffer of type " << fastMemRefType
               << " and size " << llvm::divideCeil(*sizeInBytes, 1024)
               << " KiB\n");
  } else {
    // Reuse the one already created.
    fastMemRef = fastBufferMap[memref];
    *sizeInBytes = 0;
  }

  auto numElementsSSA =
      top.create<ConstantIndexOp>(loc, numElements.getValue());

  SmallVector<StrideInfo, 4> strideInfos;
  getMultiLevelStrides(region, fastBufferShape, &strideInfos);

  // TODO(bondhugula): use all stride levels once DmaStartOp is extended for
  // multi-level strides.
  if (strideInfos.size() > 1) {
    LLVM_DEBUG(llvm::dbgs() << "Only up to one level of stride supported\n");
    return failure();
  }

  Value *stride = nullptr;
  Value *numEltPerStride = nullptr;
  if (!strideInfos.empty()) {
    stride = top.create<ConstantIndexOp>(loc, strideInfos[0].stride);
    numEltPerStride =
        top.create<ConstantIndexOp>(loc, strideInfos[0].numEltPerStride);
  }

  // Record the last operation where we want the memref replacement to end. We
  // later do the memref replacement only in [begin, postDomFilter] so
  // that the original memref's used in the data movement code themselves don't
  // get replaced.
  auto postDomFilter = std::prev(end);

  // Create fully composed affine maps for each memref.
  auto memAffineMap = b.getMultiDimIdentityMap(memIndices.size());
  fullyComposeAffineMapAndOperands(&memAffineMap, &memIndices);
  auto bufAffineMap = b.getMultiDimIdentityMap(bufIndices.size());
  fullyComposeAffineMapAndOperands(&bufAffineMap, &bufIndices);

  if (!copyOptions.generateDma) {
    // Point-wise copy generation.
    auto copyNest = generatePointWiseCopy(loc, memref, fastMemRef, memAffineMap,
                                          memIndices, fastBufferShape,
                                          /*isCopyOut=*/region.isWrite(), b);

    // Record this so that we can skip it from yet another copy.
    copyNests.insert(copyNest);

    // Since new ops are being appended (for copy out's), adjust the end to
    // mark end of block range being processed if necessary.
    if (region.isWrite() && isCopyOutAtEndOfBlock)
      *nEnd = Block::iterator(copyNest.getOperation());
  } else {
    // DMA generation.
    // Create a tag (single element 1-d memref) for the DMA.
    auto tagMemRefType = MemRefType::get({1}, top.getIntegerType(32), {},
                                         copyOptions.tagMemorySpace);
    auto tagMemRef = prologue.create<AllocOp>(loc, tagMemRefType);

    SmallVector<Value *, 4> tagIndices({zeroIndex});
    auto tagAffineMap = b.getMultiDimIdentityMap(tagIndices.size());
    fullyComposeAffineMapAndOperands(&tagAffineMap, &tagIndices);
    if (!region.isWrite()) {
      // DMA non-blocking read from original buffer to fast buffer.
      b.create<AffineDmaStartOp>(loc, memref, memAffineMap, memIndices,
                                 fastMemRef, bufAffineMap, bufIndices,
                                 tagMemRef, tagAffineMap, tagIndices,
                                 numElementsSSA, stride, numEltPerStride);
    } else {
      // DMA non-blocking write from fast buffer to the original memref.
      auto op = b.create<AffineDmaStartOp>(
          loc, fastMemRef, bufAffineMap, bufIndices, memref, memAffineMap,
          memIndices, tagMemRef, tagAffineMap, tagIndices, numElementsSSA,
          stride, numEltPerStride);
      // Since new ops may be appended at 'end' (for outgoing DMAs), adjust the
      // end to mark end of block range being processed.
      if (isCopyOutAtEndOfBlock)
        *nEnd = Block::iterator(op.getOperation());
    }

    // Matching DMA wait to block on completion; tag always has a 0 index.
    b.create<AffineDmaWaitOp>(loc, tagMemRef, tagAffineMap, zeroIndex,
                              numElementsSSA);

    // Generate dealloc for the tag.
    auto tagDeallocOp = epilogue.create<DeallocOp>(loc, tagMemRef);
    if (*nEnd == end && isCopyOutAtEndOfBlock)
      // Since new ops are being appended (for outgoing DMAs), adjust the end to
      // mark end of range of the original.
      *nEnd = Block::iterator(tagDeallocOp.getOperation());
  }

  // Generate dealloc for the buffer.
  if (!existingBuf) {
    auto bufDeallocOp = epilogue.create<DeallocOp>(loc, fastMemRef);
    // When generating pointwise copies, `nEnd' has to be set to deallocOp on
    // the fast buffer (since it marks the new end insertion point).
    if (!copyOptions.generateDma && *nEnd == end && isCopyOutAtEndOfBlock)
      *nEnd = Block::iterator(bufDeallocOp.getOperation());
  }

  // Replace all uses of the old memref with the faster one while remapping
  // access indices (subtracting out lower bound offsets for each dimension).
  // Ex: to replace load %A[%i, %j] with load %Abuf[%i - %iT, %j - %jT],
  // index remap will be (%i, %j) -> (%i - %iT, %j - %jT),
  // i.e., affine.apply (d0, d1, d2, d3) -> (d2-d0, d3-d1) (%iT, %jT, %i, %j),
  // and (%iT, %jT) will be the 'extraOperands' for 'rep all memref uses with'.
  // d2, d3 correspond to the original indices (%i, %j).
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    // The starting operands of indexRemap will be regionSymbols (the symbols on
    // which the memref region is parametric); then those corresponding to
    // the memref's original indices follow.
    auto dimExpr = b.getAffineDimExpr(regionSymbols.size() + i);
    remapExprs.push_back(dimExpr - offsets[i]);
  }
  auto indexRemap = AffineMap::get(regionSymbols.size() + rank, 0, remapExprs);

  // Record the begin since it may be invalidated by memref replacement.
  Block::iterator prevOfBegin;
  bool isBeginAtStartOfBlock = (begin == block->begin());
  if (!isBeginAtStartOfBlock)
    prevOfBegin = std::prev(begin);

  // *Only* those uses within the range [begin, end) of 'block' are replaced.
  replaceAllMemRefUsesWith(memref, fastMemRef,
                           /*extraIndices=*/{}, indexRemap,
                           /*extraOperands=*/regionSymbols,
                           /*symbolOperands=*/{},
                           /*domInstFilter=*/&*begin,
                           /*postDomInstFilter=*/&*postDomFilter);

  *nBegin = isBeginAtStartOfBlock ? block->begin() : std::next(prevOfBegin);

  return success();
}

/// Construct the memref region to just include the entire memref. Returns false
/// dynamic shaped memref's for now. `numParamLoopIVs` is the number of
/// enclosing loop IVs of opInst (starting from the outermost) that the region
/// is parametric on.
static bool getFullMemRefAsRegion(Operation *opInst, unsigned numParamLoopIVs,
                                  MemRefRegion *region) {
  unsigned rank;
  if (auto loadOp = dyn_cast<AffineLoadOp>(opInst)) {
    rank = loadOp.getMemRefType().getRank();
    region->memref = loadOp.getMemRef();
    region->setWrite(false);
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst)) {
    rank = storeOp.getMemRefType().getRank();
    region->memref = storeOp.getMemRef();
    region->setWrite(true);
  } else {
    assert(false && "expected load or store op");
    return false;
  }
  auto memRefType = region->memref->getType().cast<MemRefType>();
  if (!memRefType.hasStaticShape())
    return false;

  auto *regionCst = region->getConstraints();

  // Just get the first numSymbols IVs, which the memref region is parametric
  // on.
  SmallVector<AffineForOp, 4> ivs;
  getLoopIVs(*opInst, &ivs);
  ivs.resize(numParamLoopIVs);
  SmallVector<Value *, 4> symbols;
  extractForInductionVars(ivs, &symbols);
  regionCst->reset(rank, numParamLoopIVs, 0);
  regionCst->setIdValues(rank, rank + numParamLoopIVs, symbols);

  // Memref dim sizes provide the bounds.
  for (unsigned d = 0; d < rank; d++) {
    auto dimSize = memRefType.getDimSize(d);
    assert(dimSize > 0 && "filtered dynamic shapes above");
    regionCst->addConstantLowerBound(d, 0);
    regionCst->addConstantUpperBound(d, dimSize - 1);
  }
  return true;
}

/// Generates copies for a contiguous sequence of operations in `block` in the
/// iterator range [`begin', `end'), where `end' can't be past the terminator of
/// the block (since additional operations are potentially inserted right before
/// `end'. Returns the total size of the fast buffers used.
//  Since we generate alloc's and dealloc's for all fast buffers (before and
//  after the range of operations resp.), all of the fast memory capacity is
//  assumed to be available for processing this block range.
uint64_t mlir::affineDataCopyGenerate(Block::iterator begin,
                                      Block::iterator end,
                                      const AffineCopyOptions &copyOptions,
                                      DenseSet<Operation *> &copyNests) {
  if (begin == end)
    return 0;

  assert(begin->getBlock() == std::prev(end)->getBlock() &&
         "Inconsistent block begin/end args");
  assert(end != end->getBlock()->end() && "end can't be the block terminator");

  Block *block = begin->getBlock();

  // Copies will be generated for this depth, i.e., symbolic in all loops
  // surrounding the this block range.
  unsigned copyDepth = getNestingDepth(*begin);

  LLVM_DEBUG(llvm::dbgs() << "Generating copies at depth " << copyDepth
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "from begin: " << *begin << "\n");
  LLVM_DEBUG(llvm::dbgs() << "to inclusive end: " << *std::prev(end) << "\n");

  // List of memory regions to copy for. We need a map vector to have a
  // guaranteed iteration order to write test cases. CHECK-DAG doesn't help here
  // since the alloc's for example are identical except for the SSA id.
  SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4> readRegions;
  SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4> writeRegions;

  // Map from original memref's to the fast buffers that their accesses are
  // replaced with.
  DenseMap<Value *, Value *> fastBufferMap;

  // To check for errors when walking the block.
  bool error = false;

  // Walk this range of operations  to gather all memory regions.
  block->walk(begin, end, [&](Operation *opInst) {
    // Gather regions to allocate to buffers in faster memory space.
    if (auto loadOp = dyn_cast<AffineLoadOp>(opInst)) {
      if ((loadOp.getMemRefType().getMemorySpace() !=
           copyOptions.slowMemorySpace))
        return;
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(opInst)) {
      if (storeOp.getMemRefType().getMemorySpace() !=
          copyOptions.slowMemorySpace)
        return;
    } else {
      // Neither load nor a store op.
      return;
    }

    // Compute the MemRefRegion accessed.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(region->compute(opInst, copyDepth))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Error obtaining memory region: semi-affine maps?\n");
      LLVM_DEBUG(llvm::dbgs() << "over-approximating to the entire memref\n");
      if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
        LLVM_DEBUG(
            opInst->emitError("non-constant memref sizes not yet supported"));
        error = true;
        return;
      }
    }

    // Each memref has a single buffer associated with it irrespective of how
    // many load's and store's happen on it.
    // TODO(bondhugula): in the future, when regions don't intersect and satisfy
    // other properties (based on load/store regions), we could consider
    // multiple buffers per memref.

    // Add to the appropriate region if it's not already in it, or take a
    // bounding box union with the existing one if it's already in there.
    // Note that a memref may have both read and write regions - so update the
    // region in the other list if one exists (write in case of read and vice
    // versa) since there is a single bounding box for a memref across all reads
    // and writes that happen on it.

    // Attempts to update; returns true if 'region' exists in targetRegions.
    auto updateRegion =
        [&](const SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4>
                &targetRegions) {
          auto it = targetRegions.find(region->memref);
          if (it == targetRegions.end())
            return false;

          // Perform a union with the existing region.
          if (failed(it->second->unionBoundingBox(*region))) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Memory region bounding box failed; "
                          "over-approximating to the entire memref\n");
            // If the union fails, we will overapproximate.
            if (!getFullMemRefAsRegion(opInst, copyDepth, region.get())) {
              LLVM_DEBUG(opInst->emitError(
                  "non-constant memref sizes not yet supported"));
              error = true;
              return true;
            }
            it->second->getConstraints()->clearAndCopyFrom(
                *region->getConstraints());
          } else {
            // Union was computed and stored in 'it->second': copy to 'region'.
            region->getConstraints()->clearAndCopyFrom(
                *it->second->getConstraints());
          }
          return true;
        };

    bool existsInRead = updateRegion(readRegions);
    if (error)
      return;
    bool existsInWrite = updateRegion(writeRegions);
    if (error)
      return;

    // Finally add it to the region list.
    if (region->isWrite() && !existsInWrite) {
      writeRegions[region->memref] = std::move(region);
    } else if (!region->isWrite() && !existsInRead) {
      readRegions[region->memref] = std::move(region);
    }
  });

  if (error) {
    begin->emitError(
        "copy generation failed for one or more memref's in this block\n");
    return 0;
  }

  uint64_t totalCopyBuffersSizeInBytes = 0;
  bool ret = true;
  auto processRegions =
      [&](const SmallMapVector<Value *, std::unique_ptr<MemRefRegion>, 4>
              &regions) {
        for (const auto &regionEntry : regions) {
          // For each region, hoist copy in/out past all hoistable
          // 'affine.for's.
          Block::iterator copyInPlacementStart, copyOutPlacementStart;
          Block *copyPlacementBlock;
          findHighestBlockForPlacement(
              *regionEntry.second, *block, begin, end, &copyPlacementBlock,
              &copyInPlacementStart, &copyOutPlacementStart);

          uint64_t sizeInBytes;
          Block::iterator nBegin, nEnd;
          LogicalResult iRet = generateCopy(
              *regionEntry.second, block, begin, end, copyPlacementBlock,
              copyInPlacementStart, copyOutPlacementStart, copyOptions,
              fastBufferMap, copyNests, &sizeInBytes, &nBegin, &nEnd);
          if (succeeded(iRet)) {
            // begin/end could have been invalidated, and need update.
            begin = nBegin;
            end = nEnd;
            totalCopyBuffersSizeInBytes += sizeInBytes;
          }
          ret = ret & succeeded(iRet);
        }
      };
  processRegions(readRegions);
  processRegions(writeRegions);

  if (!ret) {
    begin->emitError(
        "copy generation failed for one or more memref's in this block\n");
    return totalCopyBuffersSizeInBytes;
  }

  // For a range of operations, a note will be emitted at the caller.
  AffineForOp forOp;
  uint64_t sizeInKib = llvm::divideCeil(totalCopyBuffersSizeInBytes, 1024);
  if (llvm::DebugFlag && (forOp = dyn_cast<AffineForOp>(&*begin))) {
    forOp.emitRemark()
        << sizeInKib
        << " KiB of copy buffers in fast memory space for this block\n";
  }

  if (totalCopyBuffersSizeInBytes > copyOptions.fastMemCapacityBytes) {
    StringRef str = "Total size of all copy buffers' for this block "
                    "exceeds fast memory capacity\n";
    block->getParentOp()->emitError(str);
  }

  return totalCopyBuffersSizeInBytes;
}
