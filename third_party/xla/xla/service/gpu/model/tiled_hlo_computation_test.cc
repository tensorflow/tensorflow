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

#include "xla/service/gpu/model/tiled_hlo_computation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(BlockLevelParametersTest,
     BlockLevelParametersCanBeParsedFromBlockLevelFusionConfig) {
  BlockLevelFusionConfig block_level_fusion_config;
  Tile tile;
  tile.mutable_sizes()->Add(18);
  tile.mutable_sizes()->Add(19);
  *block_level_fusion_config.add_output_tiles() = tile;
  block_level_fusion_config.set_num_warps(12);
  block_level_fusion_config.set_num_ctas(13);
  block_level_fusion_config.set_num_stages(14);

  BlockLevelParameters block_level_parameters =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          block_level_fusion_config);
  EXPECT_EQ(block_level_parameters.output_tile_sizes.size(), 1);
  EXPECT_THAT(block_level_parameters.output_tile_sizes[0], ElementsAre(18, 19));
  EXPECT_THAT(block_level_parameters.num_warps, 12);
  EXPECT_THAT(block_level_parameters.num_ctas, 13);
  EXPECT_THAT(block_level_parameters.num_stages, 14);
}

TEST(BlockLevelParametersTest,
     BlockLevelParametersCanBeConvertedToBlockLevelFusionConfig) {
  BlockLevelParameters block_level_parameters;
  block_level_parameters.output_tile_sizes = {{18, 19}};
  block_level_parameters.num_warps = 12;
  block_level_parameters.num_ctas = 13;
  block_level_parameters.num_stages = 14;

  BlockLevelFusionConfig block_level_fusion_config =
      block_level_parameters.ToBlockLevelFusionConfig();

  EXPECT_EQ(block_level_fusion_config.output_tiles_size(), 1);
  EXPECT_THAT(block_level_fusion_config.output_tiles(0).sizes(),
              ElementsAre(18, 19));
  EXPECT_THAT(block_level_fusion_config.num_warps(), 12);
  EXPECT_THAT(block_level_fusion_config.num_ctas(), 13);
  EXPECT_THAT(block_level_fusion_config.num_stages(), 14);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
