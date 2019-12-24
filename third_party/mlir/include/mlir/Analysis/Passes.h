//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
