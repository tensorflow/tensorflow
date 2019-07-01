//===- Transforms.h - Linalg dialect Transformations definition -----------===//
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

#ifndef LINALG4_TRANSFORMS_H_
#define LINALG4_TRANSFORMS_H_

#include "linalg3/Transforms.h"
#include "mlir/Support/LLVM.h"

namespace linalg {

/// Rewrites a linalg `op` in tiled loop form and erases `op`.
llvm::Optional<llvm::SmallVector<mlir::AffineForOp, 8>>
writeAsTiledLoops(mlir::Operation *op, llvm::ArrayRef<uint64_t> tileSizes);

/// Rewrites a linalg `op` in tiled view form and erases `op`.
llvm::Optional<llvm::SmallVector<mlir::AffineForOp, 8>>
writeAsTiledViews(mlir::Operation *op, llvm::ArrayRef<mlir::Value *> tileSizes);

/// Apply `writeAsTiledLoops` on all linalg ops. This is a convenience function
/// and is not exposed as a pass because a fixed set of tile sizes for all ops
/// in a function can generally not be specified.
void lowerToTiledLoops(mlir::Function f, llvm::ArrayRef<uint64_t> tileSizes);

/// Apply `writeAsTiledViews` on all linalg ops. This is a convenience function
/// and is not exposed as a pass because a fixed set of tile sizes for all ops
/// in a function can generally not be specified.
void lowerToTiledViews(mlir::Function f,
                       llvm::ArrayRef<mlir::Value *> tileSizes);

} // namespace linalg

#endif // LINALG4_TRANSFORMS_H_
