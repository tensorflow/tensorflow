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
class AffineForOp;
class AffineMap;
class MemRefType;
class NestedPattern;
class Operation;
class Value;

// TODO(riverriddle) Remove this after Value is value-typed.
using ValuePtr = Value *;

/// Returns the trip count of the loop as an affine map with its corresponding
/// operands if the latter is expressible as an affine expression, and nullptr
/// otherwise. This method always succeeds as long as the lower bound is not a
/// multi-result map. The trip count expression is simplified before returning.
/// This method only utilizes map composition to construct lower and upper
/// bounds before computing the trip count expressions
// TODO(mlir-team): this should be moved into 'Transforms/' and be replaced by a
// pure analysis method relying on FlatAffineConstraints
void buildTripCountMapAndOperands(AffineForOp forOp, AffineMap *map,
                                  SmallVectorImpl<ValuePtr> *operands);

/// Returns the trip count of the loop if it's a constant, None otherwise. This
/// uses affine expression analysis and is able to determine constant trip count
/// in non-trivial cases.
Optional<uint64_t> getConstantTripCount(AffineForOp forOp);

/// Returns the greatest known integral divisor of the trip count. Affine
/// expression analysis is used (indirectly through getTripCount), and
/// this method is thus able to determine non-trivial divisors.
uint64_t getLargestDivisorOfTripCount(AffineForOp forOp);

/// Given an induction variable `iv` of type AffineForOp and `indices` of type
/// IndexType, returns the set of `indices` that are independent of `iv`.
///
/// Prerequisites (inherited from `isAccessInvariant` above):
///   1. `iv` and `indices` of the proper type;
///   2. at most one affine.apply is reachable from each index in `indices`;
///
/// Emits a note if it encounters a chain of affine.apply and conservatively
///  those cases.
DenseSet<ValuePtr, DenseMapInfo<ValuePtr>>
getInvariantAccesses(ValuePtr iv, ArrayRef<ValuePtr> indices);

using VectorizableLoopFun = std::function<bool(AffineForOp)>;

/// Checks whether the loop is structurally vectorizable; i.e.:
///   1. no conditionals are nested under the loop;
///   2. all nested load/stores are to scalar MemRefs.
/// TODO(ntv): relax the no-conditionals restriction
bool isVectorizableLoopBody(AffineForOp loop,
                            NestedPattern &vectorTransferMatcher);

/// Checks whether the loop is structurally vectorizable and that all the LoadOp
/// and StoreOp matched have access indexing functions that are are either:
///   1. invariant along the loop induction variable created by 'loop';
///   2. varying along at most one memory dimension. If such a unique dimension
///      is found, it is written into `memRefDim`.
bool isVectorizableLoopBody(AffineForOp loop, int *memRefDim,
                            NestedPattern &vectorTransferMatcher);

/// Checks where SSA dominance would be violated if a for op's body
/// operations are shifted by the specified shifts. This method checks if a
/// 'def' and all its uses have the same shift factor.
// TODO(mlir-team): extend this to check for memory-based dependence
// violation when we have the support.
bool isInstwiseShiftValid(AffineForOp forOp, ArrayRef<uint64_t> shifts);
} // end namespace mlir

#endif // MLIR_ANALYSIS_LOOP_ANALYSIS_H
