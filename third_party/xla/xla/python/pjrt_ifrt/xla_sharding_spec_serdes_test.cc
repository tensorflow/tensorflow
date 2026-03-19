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

#include <memory>
#include <tuple>

#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/python/pjrt_ifrt/xla_sharding_spec.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ShardingSpecSerDesTestParam = std::tuple<SerDesVersion, int>;

class ShardingSpecSerDesTest
    : public testing::TestWithParam<ShardingSpecSerDesTestParam> {
 public:
  SerDesVersion version() const { return std::get<0>(GetParam()); }
  int num_shards() const { return std::get<1>(GetParam()); }
};

TEST_P(ShardingSpecSerDesTest, HloShardingSpecRoundTrip) {
  ASSERT_EQ(num_shards() % 2, 0);
  auto xla_hlo_sharding =
      xla::HloSharding::Tile(xla::TileAssignment({2, num_shards() / 2}));
  auto sharding_spec = HloShardingSpec::Create(num_shards(), xla_hlo_sharding);

  TF_ASSERT_OK_AND_ASSIGN(
      auto serialized,
      Serialize(*sharding_spec, std::make_unique<SerializeOptions>(version())));

  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      Deserialize<ShardingSpec>(serialized, /*options=*/nullptr));

  const auto* deserialized_spec =
      llvm::dyn_cast<HloShardingSpec>(deserialized.get());
  ASSERT_NE(deserialized_spec, nullptr);
  EXPECT_EQ(deserialized_spec->num_shards(), num_shards());
  EXPECT_EQ(deserialized_spec->xla_hlo_sharding(), xla_hlo_sharding);
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumShards, ShardingSpecSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(2, 4)));

}  // namespace
}  // namespace ifrt
}  // namespace xla
