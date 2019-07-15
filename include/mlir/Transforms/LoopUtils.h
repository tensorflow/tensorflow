//===- LoopUtils.h - Loop transformation utilities --------------*- C++ -*-===//
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
// This header file defines prototypes for various loop transformation utility
// methods: these are not passes by themselves but are used either by passes,
// optimization sequences, or in turn by other transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOP_UTILS_H
#define MLIR_TRANSFORMS_LOOP_UTILS_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class AffineMap;
class AffineForOp;
class FuncOp;
class OpBuilder;
class Value;

namespace loop {
class ForOp;
} // end namespace loop

/// Unrolls this for operation completely if the trip count is known to be
/// constant. Returns failure otherwise.
LogicalResult loopUnrollFull(AffineForOp forOp);

/// Unrolls this for operation by the specified unroll factor. Returns failure
/// if the loop cannot be unrolled either due to restrictions or due to invalid
/// unroll factors.
LogicalResult loopUnrollByFactor(AffineForOp forOp, uint64_t unrollFactor);

/// Unrolls this loop by the specified unroll factor or its trip count,
/// whichever is lower.
LogicalResult loopUnrollUpToFactor(AffineForOp forOp, uint64_t unrollFactor);

/// Get perfectly nested sequence of loops starting at root of loop nest
/// (the first op being another AffineFor, and the second op - a terminator).
/// A loop is perfectly nested iff: the first op in the loop's body is another
/// AffineForOp, and the second op is a terminator).
void getPerfectlyNestedLoops(SmallVectorImpl<AffineForOp> &nestedLoops,
                             AffineForOp root);
void getPerfectlyNestedLoops(SmallVectorImpl<loop::ForOp> &nestedLoops,
                             loop::ForOp root);

/// Unrolls and jams this loop by the specified factor. Returns success if the
/// loop is successfully unroll-jammed.
LogicalResult loopUnrollJamByFactor(AffineForOp forOp,
                                    uint64_t unrollJamFactor);

/// Unrolls and jams this loop by the specified factor or by the trip count (if
/// constant), whichever is lower.
LogicalResult loopUnrollJamUpToFactor(AffineForOp forOp,
                                      uint64_t unrollJamFactor);

/// Promotes the loop body of a AffineForOp to its containing block if the
/// AffineForOp was known to have a single iteration.
LogicalResult promoteIfSingleIteration(AffineForOp forOp);

/// Promotes all single iteration AffineForOp's in the Function, i.e., moves
/// their body into the containing Block.
void promoteSingleIterationLoops(FuncOp f);

/// Computes the cleanup loop lower bound of the loop being unrolled with
/// the specified unroll factor; this bound will also be upper bound of the main
/// part of the unrolled loop. Computes the bound as an AffineMap with its
/// operands or a null map when the trip count can't be expressed as an affine
/// expression.
void getCleanupLoopLowerBound(AffineForOp forOp, unsigned unrollFactor,
                              AffineMap *map,
                              SmallVectorImpl<Value *> *operands,
                              OpBuilder &builder);

/// Skew the operations in the body of a 'affine.for' operation with the
/// specified operation-wise shifts. The shifts are with respect to the
/// original execution order, and are multiplied by the loop 'step' before being
/// applied.
LLVM_NODISCARD
LogicalResult instBodySkew(AffineForOp forOp, ArrayRef<uint64_t> shifts,
                           bool unrollPrologueEpilogue = false);

/// Tiles the specified band of perfectly nested loops creating tile-space loops
/// and intra-tile loops. A band is a contiguous set of loops.
LLVM_NODISCARD
LogicalResult tileCodeGen(MutableArrayRef<AffineForOp> band,
                          ArrayRef<unsigned> tileSizes);

/// Performs loop interchange on 'forOpA' and 'forOpB'. Requires that 'forOpA'
/// and 'forOpB' are part of a perfectly nested sequence of loops.
void interchangeLoops(AffineForOp forOpA, AffineForOp forOpB);

/// Checks if the loop interchange permutation 'loopPermMap', of the perfectly
/// nested sequence of loops in 'loops', would violate dependences (loop 'i' in
/// 'loops' is mapped to location 'j = 'loopPermMap[i]' in the interchange).
bool isValidLoopInterchangePermutation(ArrayRef<AffineForOp> loops,
                                       ArrayRef<unsigned> loopPermMap);

/// Performs a sequence of loop interchanges on perfectly nested 'loops', as
/// specified by permutation 'loopPermMap' (loop 'i' in 'loops' is mapped to
/// location 'j = 'loopPermMap[i]' after the loop interchange).
unsigned interchangeLoops(ArrayRef<AffineForOp> loops,
                          ArrayRef<unsigned> loopPermMap);

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
// Returns AffineForOp of the root of the new loop nest after loop interchanges.
AffineForOp sinkSequentialLoops(AffineForOp forOp);

/// Sinks 'forOp' by 'loopDepth' levels by performing a series of loop
/// interchanges. Requires that 'forOp' is part of a perfect nest with
/// 'loopDepth' AffineForOps consecutively nested under it.
void sinkLoop(AffineForOp forOp, unsigned loopDepth);

/// Performs tiling fo imperfectly nested loops (with interchange) by
/// strip-mining the `forOps` by `sizes` and sinking them, in their order of
/// occurrence in `forOps`, under each of the `targets`.
/// Returns the new AffineForOps, one per each of (`forOps`, `targets`) pair,
/// nested immediately under each of `targets`.
SmallVector<SmallVector<AffineForOp, 8>, 8> tile(ArrayRef<AffineForOp> forOps,
                                                 ArrayRef<uint64_t> sizes,
                                                 ArrayRef<AffineForOp> targets);

/// Performs tiling (with interchange) by strip-mining the `forOps` by `sizes`
/// and sinking them, in their order of occurrence in `forOps`, under `target`.
/// Returns the new AffineForOps, one per `forOps`, nested immediately under
/// `target`.
SmallVector<AffineForOp, 8> tile(ArrayRef<AffineForOp> forOps,
                                 ArrayRef<uint64_t> sizes, AffineForOp target);

/// Tile a nest of standard for loops rooted at `rootForOp` with the given
/// (parametric) sizes. Sizes are expected to be strictly positive values at
/// runtime.  If more sizes than loops provided, discard the trailing values in
/// sizes.  Assumes the loop nest is permutable.
void tile(loop::ForOp rootForOp, ArrayRef<Value *> sizes);

/// Tile a nest of standard for loops rooted at `rootForOp` by finding such
/// parametric tile sizes that the outer loops have a fixed number of iterations
/// as defined in `sizes`.
void extractFixedOuterLoops(loop::ForOp rootFOrOp, ArrayRef<int64_t> sizes);

/// Replace a perfect nest of "for" loops with a single linearized loop. Assumes
/// `loops` contains a list of perfectly nested loops with bounds and steps
/// independent of any loop induction variable involved in the nest.
void coalesceLoops(MutableArrayRef<loop::ForOp> loops);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_UTILS_H
