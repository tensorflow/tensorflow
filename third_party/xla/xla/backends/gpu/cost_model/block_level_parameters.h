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

#ifndef XLA_BACKENDS_GPU_COST_MODEL_BLOCK_LEVEL_PARAMETERS_H_
#define XLA_BACKENDS_GPU_COST_MODEL_BLOCK_LEVEL_PARAMETERS_H_

#include <cstdint>
#include <vector>

#include "xla/service/gpu/backend_configs.pb.h"

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
  int64_t global_scratch_memory_size = 0;
  bool is_tma_allowed = false;
  bool is_warp_specialization_allowed = false;

  // Returns a BlockLevelParameters struct from a BlockLevelFusionConfig proto.
  static BlockLevelParameters FromBlockLevelFusionConfig(
      const BlockLevelFusionConfig& config) {
    BlockLevelParameters result;
    result.num_warps = config.num_warps();
    result.num_ctas = config.num_ctas();
    result.num_stages = config.num_stages();
    result.is_tma_allowed = config.is_tma_allowed();
    result.is_warp_specialization_allowed =
        config.is_warp_specialization_allowed();
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
    config.set_is_tma_allowed(is_tma_allowed);
    config.set_is_warp_specialization_allowed(is_warp_specialization_allowed);
    return config;
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_COST_MODEL_BLOCK_LEVEL_PARAMETERS_H_
