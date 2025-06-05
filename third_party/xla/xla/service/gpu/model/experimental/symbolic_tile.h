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
#include <vector>

#include "mlir/IR/AffineMap.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// A map from tile IDs, sizes and runtime variables to tile's offsets, sizes
// and strides.
//
// (tile IDs) [tile sizes] {runtime variables} -> (offsets, sizes, strides)
// tile IDs correspond to the dimension variables of the `tile_map`.
// tile sizes and RT vars correspond to the symbol variables of the
// `tile_map`.
struct ExperimentalSymbolicTile {
  std::string ToString() const;

  mlir::AffineMap offset_map() const;
  mlir::AffineMap size_map() const;
  mlir::AffineMap stride_map() const;

  int64_t num_tids() const { return tile_map.getNumDims(); }
  int64_t num_result_dims() const { return tile_map.getNumResults() / 3; }
  int64_t num_rt_vars() const { return rt_vars.size(); }

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ExperimentalSymbolicTile& tile) {
    sink.Append(tile.ToString());
  }

  mlir::AffineMap tile_map;
  std::vector<const HloInstruction*> rt_vars;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_TILE_H_
