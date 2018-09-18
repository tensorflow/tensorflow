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

class FunctionPass;
class MLFunctionPass;
class ModulePass;

/// Creates a loop unrolling pass. Default option or command-line options take
/// effect if -1 is passed as parameter.
MLFunctionPass *createLoopUnrollPass(int unrollFactor = -1,
                                     int unrollFull = -1);

/// Creates a loop unroll jam pass to unroll jam by the specified factor. A
/// factor of -1 lets the pass use the default factor or the one on the command
/// line if provided.
MLFunctionPass *createLoopUnrollAndJamPass(int unrollJamFactor = -1);

/// Creates an affine expression simplification pass.
FunctionPass *createSimplifyAffineExprPass();

/// Replaces all ML functions in the module with equivalent CFG functions.
/// Function references are appropriately patched to refer to the newly
/// generated CFG functions.
ModulePass *createConvertToCFGPass();

} // end namespace mlir

#endif // MLIR_TRANSFORMS_PASSES_H
