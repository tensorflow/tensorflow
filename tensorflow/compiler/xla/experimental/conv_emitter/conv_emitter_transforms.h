/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_EXPERIMENTAL_CONV_EMITTER_CONV_EMITTER_TRANSFORMS_H_
#define TENSORFLOW_COMPILER_XLA_EXPERIMENTAL_CONV_EMITTER_CONV_EMITTER_TRANSFORMS_H_

#include "absl/types/span.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/tsl/platform/types.h"

namespace xla {
namespace experimental {

struct BoundAffineMap {
  mlir::AffineMap affine_map;
  std::vector<mlir::Value> operands;
};

BoundAffineMap GetBoundAffineMapFrom(mlir::Operation* op);
mlir::Operation* CloneWithNewAffineMap(mlir::Operation* op,
                                       BoundAffineMap new_affine,
                                       mlir::OpBuilder builder);

bool IsSimpleLoop(mlir::AffineForOp loop);
std::vector<mlir::AffineForOp> CreateNestedSimpleLoops(
    absl::Span<const int64_t> upper_bounds, mlir::OpBuilder builder);
void SetBoundForSimpleLoop(mlir::AffineForOp loop, mlir::AffineExpr new_bound,
                           mlir::OpBuilder builder);

// Tile a loop with trip count N by `size`. For now, N has to be a multiple of
// size, but later this constraint will be removed.
//
// The major loop (with trip count N / size) stays as-is, while the minor loop
// (with trip count `size`) will take over the body of `target`, and be placed
// as the new body of `target`.
//
// `target` has to be within the same "perfectly nested loop group" as `loop`.
// See the documentation for mlir::getPerfectlyNestedLoops.
//
// Example:
// Before tiling `loop` with tile size X:
//   for (loop in N)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         // pass loop into affine maps
// After:
//   for (loop in N / X)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         for (tiled_loop in X)
//           // rewrite all affine exprs from loop to `loop * X + tiled_loop`.
//
// Design note:
// TileLoop is different from mlir::tile. At the moment, mlir::tile is not well
// documented about the exact tiling semantics, but the observed behavior is:
//   for (i from 0 to N)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         // pass i into affine maps
// =>
//   for (i from 0 to N, step = X)
//     for (unrelated_loop in ...)
//       for (target in ...)
//         for (j from i to min(i + X, N), step = 1)
//           // pass j into affine maps
//
// There are two differences between mlir::tile and TileLoop:
// * TileLoop always puts the tiling logic "stepping" logic into AffineExprs.
//   With that all index calculation is done in AffineExprs and easier to
//   analyze in a single place.
// * TileLoop doesn't plan to use max() and min() to resolve the issue when
//   N % X != 0. max() and min() are not representable in AffineExprs.
//   TODO(timshen): support the case where N % X != 0.
//
// TODO(timshen): consider the possibility to reuse mlir::tile's logic to
// achieve the same goal.
mlir::AffineForOp TileLoop(mlir::AffineForOp loop, int64_t size,
                           mlir::AffineForOp target);

// Sinks a segment of perfectly nested loops to the bottom. It implements this
// by rotating the loop nest by rotate_amount.
void SinkPerfectlyNestedLoops(llvm::MutableArrayRef<mlir::AffineForOp> loops,
                              int rotate_amount);

}  // namespace experimental
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_EXPERIMENTAL_CONV_EMITTER_CONV_EMITTER_TRANSFORMS_H_
