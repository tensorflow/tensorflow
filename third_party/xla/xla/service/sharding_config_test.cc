/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/sharding_config.h"

#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

class ShardingConfigTest : public ::testing::Test {
 protected:
  const Shape kShape = ShapeUtil::MakeShape(F32, {1024});
  const ShardingConfig kTestConfig{
      {{HloSharding::Manual()},
       {HloSharding::Replicate()},
       {{},
        {{HloSharding::Tile1D(kShape, 2)}, {HloSharding::Tile1D(kShape, 4)}}}}};
};

TEST_F(ShardingConfigTest, ConfigToProtoToConfigMatchesOriginal) {
  EXPECT_EQ(ShardingConfig::FromProto(ShardingConfig::ToProto(kTestConfig)),
            kTestConfig);
}

TEST_F(ShardingConfigTest, ConfigToString) {
  const std::string kExpectedConfigStr = R"(ShardingConfig {
  NodeShardingConfig {
    sharding: {manual}
  }
  NodeShardingConfig {
    sharding: {replicated}
  }
  NodeShardingConfig {
    sharding: nullopt
    nodes: [
      NodeShardingConfig {
        sharding: {devices=[2]<=[2]}
      }
      NodeShardingConfig {
        sharding: {devices=[4]<=[4]}
      }
    ]
  }
}
)";
  EXPECT_EQ(kTestConfig.ToString(), kExpectedConfigStr);
}

}  // namespace
}  // namespace xla
