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

#ifndef LINALG3_TRANSFORMS_H_
#define LINALG3_TRANSFORMS_H_

#include "linalg2/Transforms.h"

namespace mlir {
class Function;
} // namespace mlir

namespace linalg {

/// Traverses `f` and rewrites linalg.slice, and the operations it depends on,
/// to only use linalg.view operations.
void composeSliceOps(mlir::Function *f);

/// Traverses `f` and rewrites linalg.load and linalg.store to affine.load and
/// affine.store operations.
void lowerLinalgLoadStores(mlir::Function *f);

/// Traverses `f` and rewrites linalg.matmul (resp. linalg.matvec)
/// as linalg.matvec (resp. linalg.dot).
void lowerToFinerGrainedTensorContraction(mlir::Function *f);

/// Traverses `f` and rewrites linalg operations in loop form.
void lowerToLoops(mlir::Function *f);

} // namespace linalg

#endif // LINALG3_TRANSFORMS_H_
