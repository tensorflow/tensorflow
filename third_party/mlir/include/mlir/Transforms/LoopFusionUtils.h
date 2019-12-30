//===- LoopFusionUtils.h - Loop fusion utilities ----------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various loop fusion utility
// methods: these are not passes by themselves but are used either by passes,
// optimization sequences, or in turn by other transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOP_FUSION_UTILS_H
#define MLIR_TRANSFORMS_LOOP_FUSION_UTILS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class AffineForOp;
struct ComputationSliceState;
class Operation;

// TODO(andydavis) Extend this module to include utility functions for querying
// fusion cost/storage reduction, and for performing the loop fusion
// transformation.

struct FusionResult {
  enum ResultEnum {
    Success,
    FailPrecondition,     // Failed precondition for fusion. (e.g. same block).
    FailBlockDependence,  // Fusion would violate another dependence in block.
    FailFusionDependence, // Fusion would reverse dependences between loops.
    FailComputationSlice, // Unable to compute src loop computation slice.
  } value;
  FusionResult(ResultEnum v) : value(v) {}
};

/// Checks the feasibility of fusing the loop nest rooted at 'srcForOp' into the
/// loop nest rooted at 'dstForOp' at 'dstLoopDepth'. Returns FusionResult
/// 'Success' if fusion of the src/dst loop nests is feasible (i.e. they are
/// in the same block and dependences would not be violated). Otherwise
/// returns a FusionResult explaining why fusion is not feasible.
/// NOTE: This function is not feature complete and should only be used in
/// testing.
/// TODO(andydavis) Update comments when this function is fully implemented.
FusionResult canFuseLoops(AffineForOp srcForOp, AffineForOp dstForOp,
                          unsigned dstLoopDepth,
                          ComputationSliceState *srcSlice);

/// LoopNestStats aggregates various per-loop statistics (eg. loop trip count
/// and operation count) for a loop nest up until (and including) the innermost
/// loop body.
struct LoopNestStats {
  /// Map from AffineForOp to immediate child AffineForOps in its loop body.
  DenseMap<Operation *, SmallVector<AffineForOp, 2>> loopMap;
  /// Map from AffineForOp to count of operations in its loop body.
  DenseMap<Operation *, uint64_t> opCountMap;
  /// Map from AffineForOp to its constant trip count.
  DenseMap<Operation *, uint64_t> tripCountMap;
};

/// Collect loop nest statistics (eg. loop trip count and operation count)
/// in 'stats' for loop nest rooted at 'forOp'. Returns true on success,
/// returns false otherwise.
// TODO(andydavis) Consider moving this to LoopUtils.
bool getLoopNestStats(AffineForOp forOp, LoopNestStats *stats);

/// Computes the total cost of the loop nest rooted at 'forOp' using 'stats'.
/// Currently, the total cost is computed by counting the total operation
/// instance count (i.e. total number of operations in the loop body * loop
/// trip count) for the entire loop nest.
// TODO(andydavis) Improve this cost model.
int64_t getComputeCost(AffineForOp forOp, LoopNestStats &stats);

/// Computes and returns in 'computeCost', the total compute cost of fusing the
/// 'slice' of the loop nest rooted at 'srcForOp' into 'dstForOp'. Currently,
/// the total cost is computed by counting the total operation instance count
/// (i.e. total number of operations in the loop body * loop trip count) for
/// the entire loop nest.
/// Returns true on success, failure otherwise (e.g. non-constant trip counts).
// TODO(andydavis) Improve this cost model.
bool getFusionComputeCost(AffineForOp srcForOp, LoopNestStats &srcStats,
                          AffineForOp dstForOp, LoopNestStats &dstStats,
                          ComputationSliceState *slice, int64_t *computeCost);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_FUSION_UTILS_H
