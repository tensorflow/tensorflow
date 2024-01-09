/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/model/indexing_analysis.h"

namespace xla {
namespace gpu {

// A tile describes a structured subset of indices inside an N-dimensional
// array, where the set of indices captured along each dimension can be
// expressed as a strided expression
//     offset + stride * iota(size)
// with offset, stride, and size three non-negative integers, and iota the usual
// range function.
//
// A N-dimensional symbolic tile is a function from offsets, strides, and sizes
// to a N-dimensional tile. It is encoded as an affine map
//     (stride0, offset0, ..., stride{M-1}, offset{M-1})[size0, ... size{P-1}]
//  -> (expr0, ..., expr{N-1})
// where expr0, ..., expr{N-1} are strided expressions as described above.
//
// Symbolic tiles also store, for each one of their parameters, what its
// upper bound is (accessible through `max_sizes()` for size parameters and
// `max_strides_and_offsets()` for offset and stride parameters). Size
// parameters may also be assigned a specific value (accessible through
// `sizes()`).
//
// Symbolic tiles are constructed from the shape of the N-dimensional array we
// want to tile, or by propagating (composing) an existing tile with an
// `IndexingMap`. Tile propagation may fail if the results of the produced
// affine map are not all strided expressions.
class SymbolicTile {
 public:
  SymbolicTile(absl::Span<int64_t const> target_shape,
               mlir::MLIRContext* mlir_context);

  // Applies the input indexing map to this tile. Returns a symbolic tile if the
  // composition of 'indexing_map.affine_map' with 'this->affine_map' describes
  // one. Both size and max size are set for each symbol introduced by
  // 'indexing_map.affine_map'.
  // Symbols from 'indexing_map.affine_map' precede symbols from
  // 'this->affine_map' in the resulting tile's affine map.
  std::optional<SymbolicTile> TryPropagateTileThroughIndexingMap(
      const IndexingMap& indexing_map) const;

  // The affine map underlying the symbolic tile.
  const mlir::AffineMap& affine_map() const { return affine_map_; }

  // The (optional) size for each symbol in the tile's underlying affine map.
  absl::Span<std::optional<int64_t> const> sizes() const { return sizes_; }

  // The maximum size for each symbol in the tile's underlying affine map.
  absl::Span<int64_t const> max_sizes() const { return max_sizes_; }

  // The upper bound for each dimension in the tile's underlying affine map.
  absl::Span<int64_t const> max_strides_and_offsets() const {
    return max_strides_and_offsets_;
  }

 private:
  mlir::AffineMap affine_map_;
  std::vector<std::optional<int64_t>> sizes_;
  std::vector<int64_t> max_sizes_;
  std::vector<int64_t> max_strides_and_offsets_;

  SymbolicTile(mlir::AffineMap affine_map,
               absl::Span<std::optional<int64_t> const> sizes,
               absl::Span<int64_t const> max_sizes,
               absl::Span<int64_t const> max_strides_and_offsets)
      : affine_map_(affine_map),
        sizes_(sizes.begin(), sizes.end()),
        max_sizes_(max_sizes.begin(), max_sizes.end()),
        max_strides_and_offsets_(max_strides_and_offsets.begin(),
                                 max_strides_and_offsets.end()) {}
};

std::ostream& operator<<(std::ostream& out, const SymbolicTile& symbolic_tile);
std::string ToString(const SymbolicTile& symbolic_tile);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_
