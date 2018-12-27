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

namespace mlir {

class AffineMap;
class ForStmt;
class Function;
using MLFunction = Function;
class FuncBuilder;

// Values that can be used to signal success/failure. This can be implicitly
// converted to/from boolean values, with false representing success and true
// failure.
struct LLVM_NODISCARD UtilResult {
  enum ResultEnum { Success, Failure } value;
  UtilResult(ResultEnum v) : value(v) {}
  operator bool() const { return value == Failure; }
};

/// Unrolls this for statement completely if the trip count is known to be
/// constant. Returns false otherwise.
bool loopUnrollFull(ForStmt *forStmt);
/// Unrolls this for statement by the specified unroll factor. Returns false if
/// the loop cannot be unrolled either due to restrictions or due to invalid
/// unroll factors.
bool loopUnrollByFactor(ForStmt *forStmt, uint64_t unrollFactor);
/// Unrolls this loop by the specified unroll factor or its trip count,
/// whichever is lower.
bool loopUnrollUpToFactor(ForStmt *forStmt, uint64_t unrollFactor);

/// Unrolls and jams this loop by the specified factor. Returns true if the loop
/// is successfully unroll-jammed.
bool loopUnrollJamByFactor(ForStmt *forStmt, uint64_t unrollJamFactor);

/// Unrolls and jams this loop by the specified factor or by the trip count (if
/// constant), whichever is lower.
bool loopUnrollJamUpToFactor(ForStmt *forStmt, uint64_t unrollJamFactor);

/// Promotes the loop body of a ForStmt to its containing block if the ForStmt
/// was known to have a single iteration. Returns false otherwise.
bool promoteIfSingleIteration(ForStmt *forStmt);

/// Promotes all single iteration ForStmt's in the MLFunction, i.e., moves
/// their body into the containing StmtBlock.
void promoteSingleIterationLoops(MLFunction *f);

/// Returns the lower bound of the cleanup loop when unrolling a loop
/// with the specified unroll factor.
AffineMap getCleanupLoopLowerBound(const ForStmt &forStmt,
                                   unsigned unrollFactor, FuncBuilder *builder);

/// Returns the upper bound of an unrolled loop when unrolling with
/// the specified trip count, stride, and unroll factor.
AffineMap getUnrolledLoopUpperBound(const ForStmt &forStmt,
                                    unsigned unrollFactor,
                                    FuncBuilder *builder);

/// Skew the statements in the body of a 'for' statement with the specified
/// statement-wise shifts. The shifts are with respect to the original execution
/// order, and are multiplied by the loop 'step' before being applied.
UtilResult stmtBodySkew(ForStmt *forStmt, ArrayRef<uint64_t> shifts,
                        bool unrollPrologueEpilogue = false);

/// Tiles the specified band of perfectly nested loops creating tile-space loops
/// and intra-tile loops. A band is a contiguous set of loops.
UtilResult tileCodeGen(ArrayRef<ForStmt *> band, ArrayRef<unsigned> tileSizes);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_UTILS_H
