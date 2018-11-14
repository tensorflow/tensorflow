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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {

class AffineExpr;
class ForStmt;
class MemRefType;
class MLValue;

// TODO(ntv): Drop this once we have proper Ops.
static constexpr auto kVectorTransferReadOpName = "vector_transfer_read";
static constexpr auto kVectorTransferWriteOpName = "vector_transfer_write";

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

/// Given a MemRef accessed by `indices` and a dimension `dim`, determines
/// whether indices[dim] is independent of the value `input`.
// For now we assume no layout map or identity layout map in the MemRef.
// TODO(ntv): support more than identity layout map.
bool isAccessInvariant(const MLValue &input, MemRefType memRefType,
                       llvm::ArrayRef<const MLValue *> indices, unsigned dim);

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
