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
class Function;
class FuncBuilder;
template <typename T> class OpPointer;
class Value;

/// Unrolls this for instruction completely if the trip count is known to be
/// constant. Returns failure otherwise.
LogicalResult loopUnrollFull(OpPointer<AffineForOp> forOp);
/// Unrolls this for instruction by the specified unroll factor. Returns failure
/// if the loop cannot be unrolled either due to restrictions or due to invalid
/// unroll factors.
LogicalResult loopUnrollByFactor(OpPointer<AffineForOp> forOp,
                                 uint64_t unrollFactor);
/// Unrolls this loop by the specified unroll factor or its trip count,
/// whichever is lower.
LogicalResult loopUnrollUpToFactor(OpPointer<AffineForOp> forOp,
                                   uint64_t unrollFactor);

/// Unrolls and jams this loop by the specified factor. Returns success if the
/// loop is successfully unroll-jammed.
LogicalResult loopUnrollJamByFactor(OpPointer<AffineForOp> forOp,
                                    uint64_t unrollJamFactor);

/// Unrolls and jams this loop by the specified factor or by the trip count (if
/// constant), whichever is lower.
LogicalResult loopUnrollJamUpToFactor(OpPointer<AffineForOp> forOp,
                                      uint64_t unrollJamFactor);

/// Promotes the loop body of a AffineForOp to its containing block if the
/// AffineForOp was known to have a single iteration.
LogicalResult promoteIfSingleIteration(OpPointer<AffineForOp> forOp);

/// Promotes all single iteration AffineForOp's in the Function, i.e., moves
/// their body into the containing Block.
void promoteSingleIterationLoops(Function *f);

/// Computes the cleanup loop lower bound of the loop being unrolled with
/// the specified unroll factor; this bound will also be upper bound of the main
/// part of the unrolled loop. Computes the bound as an AffineMap with its
/// operands or a null map when the trip count can't be expressed as an affine
/// expression.
void getCleanupLoopLowerBound(OpPointer<AffineForOp> forOp,
                              unsigned unrollFactor, AffineMap *map,
                              SmallVectorImpl<Value *> *operands,
                              FuncBuilder *builder);

/// Skew the instructions in the body of a 'for' instruction with the specified
/// instruction-wise shifts. The shifts are with respect to the original
/// execution order, and are multiplied by the loop 'step' before being applied.
LLVM_NODISCARD
LogicalResult instBodySkew(OpPointer<AffineForOp> forOp,
                           ArrayRef<uint64_t> shifts,
                           bool unrollPrologueEpilogue = false);

/// Tiles the specified band of perfectly nested loops creating tile-space loops
/// and intra-tile loops. A band is a contiguous set of loops.
LLVM_NODISCARD
LogicalResult tileCodeGen(MutableArrayRef<OpPointer<AffineForOp>> band,
                          ArrayRef<unsigned> tileSizes);

/// Performs loop interchange on 'forOpA' and 'forOpB'. Requires that 'forOpA'
/// and 'forOpB' are part of a perfectly nested sequence of loops.
void interchangeLoops(OpPointer<AffineForOp> forOpA,
                      OpPointer<AffineForOp> forOpB);

/// Sinks 'forOp' by 'loopDepth' levels by performing a series of loop
/// interchanges. Requires that 'forOp' is part of a perfect nest with
/// 'loopDepth' AffineForOps consecutively nested under it.
void sinkLoop(OpPointer<AffineForOp> forOp, unsigned loopDepth);

/// Performs tiling fo imperfectly nested loops (with interchange) by
/// strip-mining the `forOps` by `sizes` and sinking them, in their order of
/// occurrence in `forOps`, under each of the `targets`.
/// Returns the new AffineForOps, one per each of (`forOps`, `targets`) pair,
/// nested immediately under each of `targets`.
SmallVector<SmallVector<OpPointer<AffineForOp>, 8>, 8>
tile(ArrayRef<OpPointer<AffineForOp>> forOps, ArrayRef<uint64_t> sizes,
     ArrayRef<OpPointer<AffineForOp>> targets);

/// Performs tiling (with interchange) by strip-mining the `forOps` by `sizes`
/// and sinking them, in their order of occurrence in `forOps`, under `target`.
/// Returns the new AffineForOps, one per `forOps`, nested immediately under
/// `target`.
SmallVector<OpPointer<AffineForOp>, 8>
tile(ArrayRef<OpPointer<AffineForOp>> forOps, ArrayRef<uint64_t> sizes,
     OpPointer<AffineForOp> target);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_UTILS_H
