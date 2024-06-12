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

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(BlockLevelParametersTest,
     BlockLevelParametersCanBeParsedFromBlockLevelFusionConfig) {
  BlockLevelFusionConfig block_level_fusion_config;
  block_level_fusion_config.mutable_output_tile_sizes()->Add(18);
  block_level_fusion_config.mutable_output_tile_sizes()->Add(19);

  EXPECT_THAT(BlockLevelParameters::FromBlockLevelFusionConfig(
                  block_level_fusion_config)
                  .output_tile_sizes,
              ElementsAre(18, 19));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
