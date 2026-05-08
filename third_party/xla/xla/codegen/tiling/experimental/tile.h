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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_TILE_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_TILE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/symbolic_expr.h"

namespace xla::gpu::experimental {

class TilingSpace;

// Tiling of a single dimension.
//
// Offsets, sizes and strides define the slice of the tensor dimension. Upper
// bounds set the range [0, upper_bound), values outside of this range are
// masked.
//
// Expressions for offset, size, stride and upper bound are SymbolicExpr
// functions. The TilingSpace keeps track of all dimensions and symbols we use
// in the expressions and allows to create a consistent mapping from dimensions
// and runtime variables to symbolic expression dimensions and symbols.
//
// N.B.! not all of the symbols that the TilingSpace defines are used in
// every expression. That depends on the position of the instruction in
// the graph and the traversal path that we took.
// Number of dimensions equals to the number of dimensions of the instruction
// output, parallel dimensions the corresponding root instruction are followed
// by sequential dimensions.
//
// Symbols are:
//  - tile sizes of all dimensions, followed by
//  - runtime variables.
struct DimTile {
  bool operator==(const DimTile& other) const;

  SymbolicExpr offset;
  SymbolicExpr size;
  SymbolicExpr stride;
  // The masking condition of the upper bound can be written as:
  // dimension_index < upper_bounds(tile IDs)[tile sizes]{runtime variables}
  //
  // In most of the cases, the upper bounds will coincide with the shape of the
  // tensor from which the tile is extracted. One example when upper bound does
  // not match the shape is a reshape:
  //
  // output = s32[2, 17] reshape (s32[34] input)
  //
  // If we propagate the `output` Tile with the tile size of first
  // dimension equal to 1
  //
  // (tid0, tid1)[ts1] -> offsets [tid0, tid1 * ts1]
  //                      sizes [1, ts1]
  //                      strides [1, 1]
  //                      upper bounds [2, 17]
  //
  // then for to the `input` instruction we will get a stricter upper bound
  //
  // (tid0, tid1)[ts1] -> offsets [17 * tid0 + tid1 * ts1]
  //                      sizes [ts1]
  //                      strides [1]
  //                      upper bounds [17 * tid0]
  SymbolicExpr upper_bound;
};

template <typename H>
H AbslHashValue(H h, const DimTile& dim_tile) {
  llvm::hash_code dim_tile_hash = llvm::hash_combine(
      dim_tile.offset, dim_tile.size, dim_tile.stride, dim_tile.upper_bound);
  return H::combine(std::move(h), static_cast<size_t>(dim_tile_hash));
}

// Tile is a collection of tilings for every dimension of output tensor
// of an HLO instruction. TiledHloInstruction associates a Tile
// with an HLO instruction.
class Tile {
 public:
  Tile(const TilingSpace& tiling_space, llvm::SmallVector<DimTile> dim_tiles);

  Tile(const TilingSpace& tiling_space, llvm::ArrayRef<SymbolicExpr> offsets,
       llvm::ArrayRef<SymbolicExpr> sizes, llvm::ArrayRef<SymbolicExpr> strides,
       llvm::ArrayRef<SymbolicExpr> upper_bounds);

  std::string ToString(bool print_variables = true) const;

  llvm::SmallVector<SymbolicExpr> offsets() const;
  llvm::SmallVector<SymbolicExpr> sizes() const;
  llvm::SmallVector<SymbolicExpr> strides() const;
  llvm::SmallVector<SymbolicExpr> upper_bounds() const;
  llvm::ArrayRef<DimTile> dim_tiles() const { return dim_tiles_; }
  int64_t num_dim_tiles() const { return dim_tiles_.size(); }

  absl::StatusOr<llvm::SmallVector<int64_t>> GetStaticTileSizes() const;
  absl::StatusOr<llvm::SmallVector<int64_t>> GetStaticTileStrides() const;

  const TilingSpace& tiling_space() const { return *tiling_space_; }
  mlir::MLIRContext* mlir_context() const;

  const std::optional<SymbolicExpr>& replica_id() const { return replica_id_; }
  void set_replica_id(std::optional<SymbolicExpr> replica_id) {
    replica_id_ = std::move(replica_id);
  }

  // Replace tiling expressions with the given map.
  void Replace(const llvm::DenseMap<SymbolicExpr, SymbolicExpr>& map);

  // Clone the tile with new dim tiles.
  // When we are propagating a tile to an input, we need to adjust the offsets
  // and upper bounds according to the input. The other fields are copied from
  // the original tile.
  Tile CloneWithNewDims(llvm::SmallVector<DimTile> new_dim_tiles) const;

  bool operator==(const Tile& other) const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Tile& tile) {
    sink.Append(tile.ToString());
  }

 private:
  const TilingSpace* tiling_space_;
  llvm::SmallVector<DimTile> dim_tiles_;
  std::optional<SymbolicExpr> replica_id_;
};

template <typename H>
H AbslHashValue(H h, const Tile& tile) {
  h = H::combine(std::move(h), &tile.tiling_space());
  for (const DimTile& dim_tile : tile.dim_tiles()) {
    h = H::combine(std::move(h), dim_tile);
  }
  if (tile.replica_id().has_value()) {
    h = H::combine(std::move(h), tile.replica_id().value());
  }
  return h;
}

// Returns a DimTile that covers the entire dimension with a single power of 2
// sized tile, i.e. offset 0, size = next_power_of_2(dim_size), stride 1,
// upper_bound = dim_size.
DimTile GetFullDimTile(int64_t dim_size, mlir::MLIRContext* ctx);

// Returns a DimTile that covers the entire dimension, i.e.
//  offset = SymbolicDimExpr(id) * SymbolicSymbolExpr(id),
//  size = SymbolicVariable(id), stride 1, upper_bound = dim_size.
DimTile GetDefaultDimTile(int64_t id, SymbolicExpr tile_size, int64_t dim_size);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_TILE_H_
