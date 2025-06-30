/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILE_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILE_H_

#include <cstdint>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// A map from tile IDs, sizes and runtime variables to tile's offsets, sizes,
// strides and upper bounds. Offsets-sizes-strides define what slice to extract,
// upper bounds define masking, i.e. if the tile attempts to extract elements
// with the indices outside of the bounds, the tile will be masked.
//
// (tile IDs) [tile sizes] {runtime variables} ->
//     offsets [offsets_]  sizes [sizes_] strides [strides_]
//     upper bounds [upper_bounds_]
//
// tile IDs correspond to the dimension variables of the affine expressions;
// tile sizes and RT vars correspond to the symbol variables.
//
// The masking condition of the upper bound can be written as:
// dimension_index < upper_bounds[i](tile IDs)
//
// In most of the cases, the upper bounds will coincide with the shape of the
// tensor from which the tile is extracted.
//
// One example when upper bound does not match the shape is a reshape:
// output = s32[2, 17] reshape (s32[34] input)
//
// If we propagate the `output` tile with the ts0 == 1,
//
// (tid0, tid1)[ts1] -> offsets [tid0, tid1 * ts1] sizes [1, ts1] strides [1, 1]
//              upper bounds [2, 17]
//
// to the `input` we will get a stricter upper bound
//
// (tid0, tid1)[ts1] -> offsets [17 * tid0 + tid1 * ts1] sizes [ts1] strides [1]
//              upper bounds [17 * tid0]
class ExperimentalSymbolicTile {
 public:
  ExperimentalSymbolicTile(mlir::MLIRContext* mlir_context,
                           int64_t num_tile_ids,
                           llvm::ArrayRef<mlir::AffineExpr> offsets,
                           llvm::ArrayRef<mlir::AffineExpr> sizes,
                           llvm::ArrayRef<mlir::AffineExpr> strides,
                           llvm::ArrayRef<mlir::AffineExpr> upper_bounds,
                           llvm::ArrayRef<const HloInstruction*> rt_vars);

  std::string ToString() const;

  llvm::ArrayRef<mlir::AffineExpr> offsets() const { return offsets_; }
  llvm::ArrayRef<mlir::AffineExpr> sizes() const { return sizes_; }
  llvm::ArrayRef<mlir::AffineExpr> strides() const { return strides_; }
  llvm::ArrayRef<mlir::AffineExpr> upper_bounds() const {
    return upper_bounds_;
  }

  int64_t num_tile_ids() const { return num_tile_ids_; }
  int64_t num_result_dims() const { return offsets().size(); }

  llvm::ArrayRef<const HloInstruction*> rt_vars() const { return rt_vars_; }
  int64_t num_rt_vars() const { return rt_vars_.size(); }

  mlir::MLIRContext* mlir_context() const { return mlir_context_; }

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ExperimentalSymbolicTile& tile) {
    sink.Append(tile.ToString());
  }

 private:
  mlir::MLIRContext* mlir_context_;
  int64_t num_tile_ids_;
  llvm::SmallVector<mlir::AffineExpr> offsets_;
  llvm::SmallVector<mlir::AffineExpr> sizes_;
  llvm::SmallVector<mlir::AffineExpr> strides_;
  llvm::SmallVector<mlir::AffineExpr> upper_bounds_;
  llvm::SmallVector<const HloInstruction*> rt_vars_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILE_H_
