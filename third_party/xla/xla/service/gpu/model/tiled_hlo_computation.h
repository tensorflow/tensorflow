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
#include "xla/hlo/analysis/indexing_map.h"
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
  std::vector<int64_t> output_tile_sizes;

  // Triton-specific parameters.
  int64_t num_warps = 1;
  int num_ctas = 1;
  int num_stages = 1;

  // Returns a BlockLevelParameters struct from a BlockLevelFusionConfig proto.
  static BlockLevelParameters FromBlockLevelFusionConfig(
      const BlockLevelFusionConfig& config) {
    return BlockLevelParameters{
        /*output_tile_sizes=*/
        std::vector<int64_t>(config.output_tile_sizes().begin(),
                             config.output_tile_sizes().end()),
        /*num_warps=*/config.num_warps()};
  }

  // Returns a BlockLevelFusionConfig proto from a BlockLevelParameters struct.
  BlockLevelFusionConfig ToBlockLevelFusionConfig() const {
    BlockLevelFusionConfig config;
    config.mutable_output_tile_sizes()->Add(output_tile_sizes.begin(),
                                            output_tile_sizes.end());
    config.set_num_warps(num_warps);
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
  // expected to be sorted in def-before-use order.
  static TiledHloComputation FromSortedTiledHloInstructions(
      std::vector<std::unique_ptr<TiledHloInstruction>> instructions,
      llvm::SmallVector<int64_t> num_output_tiles_per_dim) {
    return TiledHloComputation(std::move(instructions),
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

  // Returns the root instruction of the computation.
  const TiledHloInstruction* GetRoot() const {
    return instructions_.back().get();
  }

  // Returns a string representation of the computation. Used only for error
  // messages and debugging.
  std::string ToString() const;

 private:
  explicit TiledHloComputation(
      std::vector<std::unique_ptr<TiledHloInstruction>> instructions,
      llvm::SmallVector<int64_t> num_output_tiles_per_dim)
      : instructions_(std::move(instructions)),
        num_output_tiles_per_dim_(std::move(num_output_tiles_per_dim)) {}

  // Stores instructions in the computation in def-before-use order.
  std::vector<std::unique_ptr<TiledHloInstruction>> instructions_;

  // Stores the number of output tiles for each dimension.
  llvm::SmallVector<int64_t> num_output_tiles_per_dim_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILED_HLO_COMPUTATION_H_
