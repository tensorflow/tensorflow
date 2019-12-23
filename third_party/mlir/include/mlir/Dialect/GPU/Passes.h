//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PASSES_H_
#define MLIR_DIALECT_GPU_PASSES_H_

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OpPassBase;

std::unique_ptr<OpPassBase<ModuleOp>> createGpuKernelOutliningPass();

} // namespace mlir

#endif // MLIR_DIALECT_GPU_PASSES_H_
