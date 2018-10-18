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

#include "llvm/ADT/Optional.h"

namespace mlir {

class AffineExpr;
class ForStmt;

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

/// Checks whether all the LoadOp and StoreOp matched have access indexing
/// functions that are are either:
///   1. invariant along the loop induction variable;
///   2. varying along the fastest varying memory dimension only.
// TODO(ntv): return for each statement the required action to make the loop
// vectorizable. A function over the actions will give us a cost model.
bool isVectorizableLoop(const ForStmt &loop);

} // end namespace mlir

#endif // MLIR_ANALYSIS_LOOP_ANALYSIS_H
