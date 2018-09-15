//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose pass constructors in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_PASSES_H
#define MLIR_TRANSFORMS_PASSES_H

#include "mlir/Support/LLVM.h"

namespace mlir {

class ForStmt;
class FunctionPass;
class MLFunction;
class MLFunctionPass;
class ModulePass;

/// Creates a loop unrolling pass. Default option or command-line options take
/// effect if -1 is passed as parameter.
MLFunctionPass *createLoopUnrollPass(int unrollFactor = -1,
                                     int unrollFull = -1);

/// Unrolls this loop completely.
bool loopUnrollFull(ForStmt *forStmt);
/// Unrolls this loop by the specified unroll factor.
bool loopUnrollByFactor(ForStmt *forStmt, uint64_t unrollFactor);

/// Creates a loop unroll jam pass to unroll jam by the specified factor. A
/// factor of -1 lets the pass use the default factor or the one on the command
/// line if provided.
MLFunctionPass *createLoopUnrollAndJamPass(int unrollJamFactor = -1);

/// Unrolls and jams this loop by the specified factor.
bool loopUnrollJamByFactor(ForStmt *forStmt, uint64_t unrollJamFactor);

/// Creates an affine expression simplification pass.
FunctionPass *createSimplifyAffineExprPass();

/// Replaces all ML functions in the module with equivalent CFG functions.
/// Function references are appropriately patched to refer to the newly
/// generated CFG functions.
ModulePass *createConvertToCFGPass();

/// Promotes the loop body of a ForStmt to its containing block if the ForStmt
/// was known to have a single iteration. Returns false otherwise.
bool promoteIfSingleIteration(ForStmt *forStmt);

/// Promotes all single iteration ForStmt's in the MLFunction, i.e., moves
/// their body into the containing StmtBlock.
void promoteSingleIterationLoops(MLFunction *f);

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_H
