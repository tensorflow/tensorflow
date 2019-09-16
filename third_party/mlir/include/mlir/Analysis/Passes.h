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
// This header file defines prototypes that expose pass constructors in the
// analysis library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PASSES_H
#define MLIR_ANALYSIS_PASSES_H

#include "mlir/Support/LLVM.h"
#include <memory>

namespace mlir {

class FuncOp;
template <typename T> class OpPassBase;

/// Creates a pass to check memref accesses in a Function.
std::unique_ptr<OpPassBase<FuncOp>> createMemRefBoundCheckPass();

/// Creates a pass to check memref access dependences in a Function.
std::unique_ptr<OpPassBase<FuncOp>> createTestMemRefDependenceCheckPass();

/// Creates a pass to test parallelism detection; emits note for parallel loops.
std::unique_ptr<OpPassBase<FuncOp>> createParallelismDetectionTestPass();

} // end namespace mlir

#endif // MLIR_ANALYSIS_PASSES_H
