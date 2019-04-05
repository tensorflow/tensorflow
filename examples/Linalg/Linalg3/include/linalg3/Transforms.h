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
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {
class AffineForOp;
class AffineMap;
class Function;
class FunctionPassBase;
class Operation;
class Value;
} // namespace mlir

namespace linalg {

struct RangeParts {
  explicit RangeParts(unsigned reserved);
  RangeParts(llvm::ArrayRef<mlir::Value *> ranges);
  llvm::SmallVector<mlir::Value *, 4> makeRanges();

  llvm::SmallVector<mlir::Value *, 4> mins;
  llvm::SmallVector<mlir::Value *, 4> maxes;
  llvm::SmallVector<mlir::Value *, 4> steps;
};

mlir::Value *
makeFoldedComposedAffineApply(mlir::AffineMap map,
                              llvm::ArrayRef<mlir::Value *> operandsRef);

llvm::SmallVector<mlir::Value *, 4>
makeGenericLoopRanges(mlir::AffineMap operandRangesToLoopMaps,
                      llvm::ArrayRef<mlir::Value *> ranges,
                      llvm::ArrayRef<mlir::Value *> tileSizes = {});

/// Traverses `f` and rewrites linalg.slice, and the operations it depends on,
/// to only use linalg.view operations.
void composeSliceOps(mlir::Function *f);

/// Traverses `f` and rewrites linalg.matmul (resp. linalg.matvec)
/// as linalg.matvec (resp. linalg.dot).
void lowerToFinerGrainedTensorContraction(mlir::Function *f);

/// Operation-wise writing of linalg operations to loop form.
/// It is the caller's responsibility to erase the `op` if necessary.
/// This returns the enclosing loops around the body of `op` for further
/// composition of transformations.
llvm::Optional<llvm::SmallVector<mlir::AffineForOp, 4>>
writeAsLoops(mlir::Operation *op);

/// Traverses `f` and rewrites linalg operations in loop form.
void lowerToLoops(mlir::Function *f);

/// Creates a pass that rewrites linalg.load and linalg.store to affine.load and
/// affine.store operations.
mlir::FunctionPassBase *createLowerLinalgLoadStorePass();

} // namespace linalg

#endif // LINALG3_TRANSFORMS_H_
