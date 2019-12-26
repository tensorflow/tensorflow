//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
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

#ifndef MLIR_DIALECT_LINALG_PASSES_H_
#define MLIR_DIALECT_LINALG_PASSES_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class FuncOp;
class ModuleOp;
template <typename T> class OpPassBase;

namespace linalg {
std::unique_ptr<OpPassBase<FuncOp>> createLinalgFusionPass();

std::unique_ptr<OpPassBase<FuncOp>>
createLinalgTilingPass(ArrayRef<int64_t> tileSizes = {});

std::unique_ptr<OpPassBase<FuncOp>>
createLinalgPromotionPass(bool dynamicBuffers);

/// Create a pass to convert Linalg operations to loop.for loops and
/// std.load/std.store accesses.
std::unique_ptr<OpPassBase<FuncOp>> createConvertLinalgToLoopsPass();

/// Create a pass to convert Linalg operations to affine.for loops and
/// affine_load/affine_store accesses.
/// Placeholder for now, this is NYI.
std::unique_ptr<OpPassBase<FuncOp>> createConvertLinalgToAffineLoopsPass();

/// Create a pass to convert Linalg operations to the LLVMIR dialect.
std::unique_ptr<OpPassBase<ModuleOp>> createConvertLinalgToLLVMPass();

} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_PASSES_H_
