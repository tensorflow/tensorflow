/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_MODEL_TILED_HLO_COMPUTATION_H_
#define XLA_SERVICE_GPU_MODEL_TILED_HLO_COMPUTATION_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "xla/iterator_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// A container for block-level parameters. Currently only used for Triton
// fusions.
struct BlockLevelParameters {
  // TODO(b/421837868): migrate to carry a full tiling instance wherever
  // possible?
  std::vector<std::vector<int64_t>> output_tile_sizes;

  // Triton-specific parameters.
  int64_t num_warps = 1;
  int num_ctas = 1;
  int num_stages = 1;

  // Returns a BlockLevelParameters struct from a BlockLevelFusionConfig proto.
  static BlockLevelParameters FromBlockLevelFusionConfig(
      const BlockLevelFusionConfig& config) {
    BlockLevelParameters result;
    result.num_warps = config.num_warps();
    result.num_ctas = config.num_ctas();
    result.num_stages = config.num_stages();
    result.output_tile_sizes.reserve(config.output_tiles_size());
    for (const auto& tile : config.output_tiles()) {
      result.output_tile_sizes.push_back(
          std::vector<int64_t>(tile.sizes().begin(), tile.sizes().end()));
    }
    return result;
  }

  // Returns a BlockLevelFusionConfig proto from a BlockLevelParameters struct.
  BlockLevelFusionConfig ToBlockLevelFusionConfig() const {
    BlockLevelFusionConfig config;
    for (const auto& tile_sizes : output_tile_sizes) {
      Tile tile;
      tile.mutable_sizes()->Add(tile_sizes.begin(), tile_sizes.end());
      *config.add_output_tiles() = tile;
    }
    config.set_num_warps(num_warps);
    config.set_num_ctas(num_ctas);
    config.set_num_stages(num_stages);
    return config;
  }
};

// Stores `TiledHloInstruction`s in the computation.
//  * Instructions reference each other with non-owning pointers.
//  * Instructions with the same tiling parameters are CSE-ed during
//  construction.
//  * Instructions are stored in def-before-use order.
//  * The last element in the vector in the root instruction.
class TiledHloComputation {
 public:
  // Creates a computation from a list of instructions. The instructions are
  // expected to be sorted in def-before-use order. The `roots` parameter should
  // provide the roots in the order by increasing output index, and the pointers
  // in `roots` should point to tiled hlo instructions from `instructions`.
  static TiledHloComputation FromSortedTiledHloInstructions(
      std::vector<std::unique_ptr<TiledHloInstruction>> instructions,
      std::vector<const TiledHloInstruction*> roots,
      llvm::SmallVector<int64_t> num_output_tiles_per_dim) {
    return TiledHloComputation(std::move(instructions), std::move(roots),
                               std::move(num_output_tiles_per_dim));
  }

  // Returns an iterator range over the instructions in the computation in
  // def-before-use order.
  tsl::gtl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<TiledHloInstruction>>::const_iterator>>
  instructions() const {
    return {MakeUnwrappingIterator(instructions_.begin()),
            MakeUnwrappingIterator(instructions_.end())};
  }

  // Returns the number of output tiles for each dimension.
  llvm::ArrayRef<int64_t> num_output_tiles_per_dim() const {
    return num_output_tiles_per_dim_;
  }

  // Returns the total number of output tiles.
  int64_t num_output_tiles() const {
    return Product(num_output_tiles_per_dim());
  }

  // Returns the root instructions of the computation. When a computation has
  // several outputs (i.e. it has a tuple root), the roots are the operands of
  // the root tuple. The roots are order by increasing output index, and point
  // to tiled hlo instructions from `instructions_`.
  const std::vector<const TiledHloInstruction*>& GetRoots() const {
    return roots_;
  }

  // Returns a string representation of the computation. Used only for error
  // messages and debugging.
  std::string ToString() const;

 private:
  explicit TiledHloComputation(
      std::vector<std::unique_ptr<TiledHloInstruction>> instructions,
      std::vector<const TiledHloInstruction*> roots,
      llvm::SmallVector<int64_t> num_output_tiles_per_dim)
      : instructions_(std::move(instructions)),
        roots_(std::move(roots)),
        num_output_tiles_per_dim_(std::move(num_output_tiles_per_dim)) {}

  // Stores instructions in the computation in def-before-use order.
  std::vector<std::unique_ptr<TiledHloInstruction>> instructions_;

  // Stores pointers to the root instructions. Note that they do not necessarily
  // appear all at the end of `instructions_`.
  std::vector<const TiledHloInstruction*> roots_;

  // Stores the number of output tiles for each dimension.
  llvm::SmallVector<int64_t> num_output_tiles_per_dim_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILED_HLO_COMPUTATION_H_
