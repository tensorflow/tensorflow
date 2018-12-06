//===- LoopAnalysis.h - loop analysis methods -------------------*- C++ -*-===//
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
// This header file defines prototypes for methods to analyze loops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_LOOP_ANALYSIS_H
#define MLIR_ANALYSIS_LOOP_ANALYSIS_H

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

class AffineExpr;
class AffineMap;
class ForStmt;
class MemRefType;
class MLValue;
class OperationStmt;

/// Returns the trip count of the loop as an affine expression if the latter is
/// expressible as an affine expression, and nullptr otherwise. The trip count
/// expression is simplified before returning.
AffineExpr getTripCountExpr(const ForStmt &forStmt);

/// Returns the trip count of the loop if it's a constant, None otherwise. This
/// uses affine expression analysis and is able to determine constant trip count
/// in non-trivial cases.
llvm::Optional<uint64_t> getConstantTripCount(const ForStmt &forStmt);

/// Returns the greatest known integral divisor of the trip count. Affine
/// expression analysis is used (indirectly through getTripCount), and
/// this method is thus able to determine non-trivial divisors.
uint64_t getLargestDivisorOfTripCount(const ForStmt &forStmt);

/// Given an induction variable `iv` of type ForStmt and an `index` of type
/// IndexType, returns `true` if `index` is independent of `iv` and false
/// otherwise.
/// The determination supports composition with at most one AffineApplyOp.
/// The at most one AffineApplyOp comes from the fact that composition of
/// AffineApplyOp need to be canonicalized by construction to avoid writing code
/// that composes arbitrary numbers of AffineApplyOps everywhere. To achieve
/// this, at the very least, the compose-affine-apply pass must have been run.
///
/// Prerequisites:
///   1. `iv` and `index` of the proper type;
///   2. at most one reachable AffineApplyOp from index;
bool isAccessInvariant(const MLValue &iv, const MLValue &index);

/// Given an induction variable `iv` of type ForStmt and `indices` of type
/// IndexType, returns the set of `indices` that are independent of `iv`.
///
/// Prerequisites (inherited from `isAccessInvariant` above):
///   1. `iv` and `indices` of the proper type;
///   2. at most one reachable AffineApplyOp from index;
llvm::DenseSet<const MLValue *, llvm::DenseMapInfo<const MLValue *>>
getInvariantAccesses(const MLValue &iv,
                     llvm::ArrayRef<const MLValue *> indices);

/// Checks whether the loop is structurally vectorizable; i.e.:
/// 1. the loop has proper dependence semantics (parallel, reduction, etc);
/// 2. no conditionals are nested under the loop;
/// 3. all nested load/stores are to scalar MemRefs.
/// TODO(ntv): implement dependence semantics
/// TODO(ntv): relax the no-conditionals restriction
bool isVectorizableLoop(const ForStmt &loop);

/// Checks whether the loop is structurally vectorizable and that all the LoadOp
/// and StoreOp matched have access indexing functions that are are either:
///   1. invariant along the loop induction variable created by 'loop';
///   2. varying along the 'fastestVaryingDim' memory dimension.
bool isVectorizableLoopAlongFastestVaryingMemRefDim(const ForStmt &loop,
                                                    unsigned fastestVaryingDim);

/// Checks where SSA dominance would be violated if a for stmt's body statements
/// are shifted by the specified shifts. This method checks if a 'def' and all
/// its uses have the same shift factor.
// TODO(mlir-team): extend this to check for memory-based dependence
// violation when we have the support.
bool isStmtwiseShiftValid(const ForStmt &forStmt,
                          llvm::ArrayRef<uint64_t> shifts);
} // end namespace mlir

#endif // MLIR_ANALYSIS_LOOP_ANALYSIS_H
