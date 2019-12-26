//===- LoopUtils.h - Loop transformation utilities --------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various loop transformation utility
// methods: these are not passes by themselves but are used either by passes,
// optimization sequences, or in turn by other transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOP_UTILS_H
#define MLIR_TRANSFORMS_LOOP_UTILS_H

#include "mlir/IR/Block.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
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
                              AffineMap *map, SmallVectorImpl<Value> *operands,
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
using Loops = SmallVector<loop::ForOp, 8>;
using TileLoops = std::pair<Loops, Loops>;
SmallVector<SmallVector<AffineForOp, 8>, 8> tile(ArrayRef<AffineForOp> forOps,
                                                 ArrayRef<uint64_t> sizes,
                                                 ArrayRef<AffineForOp> targets);
SmallVector<Loops, 8> tile(ArrayRef<loop::ForOp> forOps, ArrayRef<Value> sizes,
                           ArrayRef<loop::ForOp> targets);

/// Performs tiling (with interchange) by strip-mining the `forOps` by `sizes`
/// and sinking them, in their order of occurrence in `forOps`, under `target`.
/// Returns the new AffineForOps, one per `forOps`, nested immediately under
/// `target`.
SmallVector<AffineForOp, 8> tile(ArrayRef<AffineForOp> forOps,
                                 ArrayRef<uint64_t> sizes, AffineForOp target);
Loops tile(ArrayRef<loop::ForOp> forOps, ArrayRef<Value> sizes,
           loop::ForOp target);

/// Tile a nest of loop::ForOp loops rooted at `rootForOp` with the given
/// (parametric) sizes. Sizes are expected to be strictly positive values at
/// runtime.  If more sizes than loops are provided, discard the trailing values
/// in sizes.  Assumes the loop nest is permutable.
/// Returns the newly created intra-tile loops.
Loops tilePerfectlyNested(loop::ForOp rootForOp, ArrayRef<Value> sizes);

/// Explicit copy / DMA generation options for mlir::affineDataCopyGenerate.
struct AffineCopyOptions {
  // True if DMAs should be generated instead of point-wise copies.
  bool generateDma;
  // The slower memory space from which data is to be moved.
  unsigned slowMemorySpace;
  // Memory space of the faster one (typically a scratchpad).
  unsigned fastMemorySpace;
  // Memory space to place tags in: only meaningful for DMAs.
  unsigned tagMemorySpace;
  // Capacity of the fast memory space in bytes.
  uint64_t fastMemCapacityBytes;
};

/// Performs explicit copying for the contiguous sequence of operations in the
/// block iterator range [`begin', `end'), where `end' can't be past the
/// terminator of the block (since additional operations are potentially
/// inserted right before `end`. Returns the total size of fast memory space
/// buffers used. `copyOptions` provides various parameters, and the output
/// argument `copyNests` is the set of all copy nests inserted, each represented
/// by its root affine.for. Since we generate alloc's and dealloc's for all fast
/// buffers (before and after the range of operations resp. or at a hoisted
/// position), all of the fast memory capacity is assumed to be available for
/// processing this block range.
uint64_t affineDataCopyGenerate(Block::iterator begin, Block::iterator end,
                                const AffineCopyOptions &copyOptions,
                                DenseSet<Operation *> &copyNests);

/// Tile a nest of standard for loops rooted at `rootForOp` by finding such
/// parametric tile sizes that the outer loops have a fixed number of iterations
/// as defined in `sizes`.
TileLoops extractFixedOuterLoops(loop::ForOp rootFOrOp,
                                 ArrayRef<int64_t> sizes);

/// Replace a perfect nest of "for" loops with a single linearized loop. Assumes
/// `loops` contains a list of perfectly nested loops with bounds and steps
/// independent of any loop induction variable involved in the nest.
void coalesceLoops(MutableArrayRef<loop::ForOp> loops);

/// Maps `forOp` for execution on a parallel grid of virtual `processorIds` of
/// size given by `numProcessors`. This is achieved by embedding the SSA values
/// corresponding to `processorIds` and `numProcessors` into the bounds and step
/// of the `forOp`. No check is performed on the legality of the rewrite, it is
/// the caller's responsibility to ensure legality.
///
/// Requires that `processorIds` and `numProcessors` have the same size and that
/// for each idx, `processorIds`[idx] takes, at runtime, all values between 0
/// and `numProcessors`[idx] - 1. This corresponds to traditional use cases for:
///   1. GPU (threadIdx, get_local_id(), ...)
///   2. MPI (MPI_Comm_rank)
///   3. OpenMP (omp_get_thread_num)
///
/// Example:
/// Assuming a 2-d grid with processorIds = [blockIdx.x, threadIdx.x] and
/// numProcessors = [gridDim.x, blockDim.x], the loop:
///
/// ```
///    loop.for %i = %lb to %ub step %step {
///      ...
///    }
/// ```
///
/// is rewritten into a version resembling the following pseudo-IR:
///
/// ```
///    loop.for %i = %lb + %step * (threadIdx.x + blockIdx.x * blockDim.x)
///       to %ub step %gridDim.x * blockDim.x * %step {
///      ...
///    }
/// ```
void mapLoopToProcessorIds(loop::ForOp forOp, ArrayRef<Value> processorId,
                           ArrayRef<Value> numProcessors);
} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_UTILS_H
