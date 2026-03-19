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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;

using ShardingSpecSerDesTestParam = std::tuple<SerDesVersion, int>;

class ShardingSpecSerDesTest
    : public testing::TestWithParam<ShardingSpecSerDesTestParam> {
 public:
  SerDesVersion version() const { return std::get<0>(GetParam()); }
  int num_shards() const { return std::get<1>(GetParam()); }
};

TEST_P(ShardingSpecSerDesTest, SingleDeviceShardingSpecRoundTrip) {
  auto sharding_spec = SingleDeviceShardingSpec::Create();

  TF_ASSERT_OK_AND_ASSIGN(
      auto serialized,
      Serialize(*sharding_spec, std::make_unique<SerializeOptions>(version())));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding_spec,
      Deserialize<SingleDeviceShardingSpec>(serialized, /*options=*/nullptr));
}

TEST_P(ShardingSpecSerDesTest, OpaqueShardingSpecRoundTrip) {
  auto sharding_spec = OpaqueShardingSpec::Create(num_shards());

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*sharding_spec, std::move(options)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding_spec,
      Deserialize<OpaqueShardingSpec>(serialized, /*options=*/nullptr));

  EXPECT_THAT(out_sharding_spec->num_shards(), sharding_spec->num_shards());
}

TEST_P(ShardingSpecSerDesTest, ConcreteShardingSpecRoundTrip) {
  std::vector<Shape> shard_shapes(num_shards(), Shape({10, 20}));
  auto sharding_spec =
      ConcreteShardingSpec::Create(/*shape=*/Shape({10 * num_shards(), 20}),
                                   /*shard_shapes=*/shard_shapes);

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*sharding_spec, std::move(options)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding_spec,
      Deserialize<ConcreteShardingSpec>(serialized, /*options=*/nullptr));

  EXPECT_THAT(out_sharding_spec->num_shards(), sharding_spec->num_shards());
  EXPECT_THAT(out_sharding_spec->shape(), sharding_spec->shape());
  EXPECT_THAT(out_sharding_spec->shard_shapes(),
              ElementsAreArray(sharding_spec->shard_shapes()));
}

TEST_P(ShardingSpecSerDesTest, ConcreteShardingSpecWithDynamicShapeRoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape,
      DynamicShape::Create(Shape({10, 20}),
                           BoundedDynamicShapeTag({false, true})));
  std::vector<DynamicShape> shard_dynamic_shapes(num_shards(),
                                                 shard_dynamic_shape);
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({10 * num_shards(), 20}),
                           BoundedDynamicShapeTag({false, true})));
  auto sharding_spec = ConcreteShardingSpec::Create(
      /*dynamic_shape=*/dynamic_shape,
      /*shard_dynamic_shapes=*/shard_dynamic_shapes);

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*sharding_spec, std::move(options)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding_spec,
      Deserialize<ConcreteShardingSpec>(serialized, /*options=*/nullptr));

  EXPECT_THAT(out_sharding_spec->num_shards(), sharding_spec->num_shards());
  EXPECT_THAT(out_sharding_spec->dynamic_shape(),
              sharding_spec->dynamic_shape());
  EXPECT_THAT(out_sharding_spec->shard_dynamic_shapes(),
              ElementsAreArray(sharding_spec->shard_dynamic_shapes()));
}

TEST_P(ShardingSpecSerDesTest, ConcreteEvenShardingSpecRoundTrip) {
  auto sharding_spec = ConcreteEvenShardingSpec::Create(
      num_shards(),
      /*shape=*/Shape({10 * num_shards(), 20}),
      /*shard_shape=*/Shape({10, 20}), /*is_fully_replicated=*/false);

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*sharding_spec, std::move(options)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding_spec,
      Deserialize<ConcreteEvenShardingSpec>(serialized, /*options=*/nullptr));

  EXPECT_THAT(out_sharding_spec->num_shards(), sharding_spec->num_shards());
  EXPECT_THAT(out_sharding_spec->shape(), sharding_spec->shape());
  EXPECT_THAT(out_sharding_spec->shard_shape(), sharding_spec->shard_shape());
  EXPECT_THAT(out_sharding_spec->IsFullyReplicated(),
              sharding_spec->IsFullyReplicated());
}

TEST_P(ShardingSpecSerDesTest, ShardingParamShardingSpecRoundTrip) {
  auto sharding_spec = ShardingParamShardingSpec::Create(
      ShardingParam({num_shards(), 1}, {{0}, {num_shards()}}));

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(auto serialized,
                          Serialize(*sharding_spec, std::move(options)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding_spec,
      Deserialize<ShardingParamShardingSpec>(serialized, /*options=*/nullptr));

  EXPECT_THAT(out_sharding_spec->num_shards(), sharding_spec->num_shards());
  EXPECT_THAT(out_sharding_spec->sharding_param(),
              sharding_spec->sharding_param());
}

TEST_P(ShardingSpecSerDesTest,
       ShardingParamShardingSpecWithUnreducedDimsRoundTrip) {
  if (version().version_number() < SerDesVersionNumber(1)) {
    GTEST_SKIP() << "Unreduced dims not supported before version 1.";
  }
  auto sharding_spec = ShardingParamShardingSpec::Create(
      ShardingParam({1, 1}, {{0}, {num_shards()}},
                    /*unreduced_axes=*/{0}));

  auto options = std::make_unique<SerializeOptions>(version());
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized,
                          Serialize(*sharding_spec, std::move(options)));
  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding_spec,
      Deserialize<ShardingParamShardingSpec>(serialized, /*options=*/nullptr));

  EXPECT_THAT(out_sharding_spec->num_shards(), sharding_spec->num_shards());
  EXPECT_THAT(out_sharding_spec->sharding_param(),
              sharding_spec->sharding_param());
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumShards, ShardingSpecSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(2, 4)));

}  // namespace
}  // namespace ifrt
}  // namespace xla
