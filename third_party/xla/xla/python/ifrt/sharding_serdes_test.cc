/* Copyright 2023 The OpenXLA Authors.

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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/bind_front.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;

class ShardingSerDesTest : public test_util::ShardingTest {};

TEST_P(ShardingSerDesTest, SingleDeviceShardingRoundTrip) {
  auto sharding = SingleDeviceSharding::Create(
      GetDevices({0}).devices().front(), MemoryKind("abc"));

  TF_ASSERT_OK_AND_ASSIGN(auto serialized, Serialize(*sharding));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<SingleDeviceSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(
                          absl::bind_front(&Client::LookupDevice, client()))));

  EXPECT_THAT(out_sharding->devices(), ElementsAreArray(sharding->devices()));
}

TEST_P(ShardingSerDesTest, OpaqueShardingRoundTrip) {
  auto sharding = OpaqueSharding::Create(GetDevices({0, 1}), MemoryKind("abc"));

  TF_ASSERT_OK_AND_ASSIGN(auto serialized, Serialize(*sharding));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<OpaqueSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(
                          absl::bind_front(&Client::LookupDevice, client()))));

  EXPECT_THAT(out_sharding->devices(), ElementsAreArray(sharding->devices()));
}

TEST_P(ShardingSerDesTest, ConcreteShardingRoundTrip) {
  auto sharding = ConcreteSharding::Create(
      GetDevices({0, 1}), MemoryKind("abc"),
      /*shape=*/Shape({10, 20}),
      /*shard_shapes=*/{Shape({3, 20}), Shape({7, 20})});

  TF_ASSERT_OK_AND_ASSIGN(auto serialized, Serialize(*sharding));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<ConcreteSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(
                          absl::bind_front(&Client::LookupDevice, client()))));

  EXPECT_THAT(out_sharding->devices(), ElementsAreArray(sharding->devices()));
  EXPECT_THAT(out_sharding->shape(), sharding->shape());
  EXPECT_THAT(out_sharding->shard_shapes(),
              ElementsAreArray(sharding->shard_shapes()));
}

TEST_P(ShardingSerDesTest, ConcreteShardingWithDynamicShapeRoundTrip) {
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({10, 20}),
                           BoundedDynamicShapeTag({false, true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape1,
      DynamicShape::Create(Shape({3, 20}),
                           BoundedDynamicShapeTag({false, true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shard_dynamic_shape2,
      DynamicShape::Create(Shape({7, 20}),
                           BoundedDynamicShapeTag({false, true})));
  auto sharding = ConcreteSharding::Create(
      GetDevices({0, 1}), MemoryKind("abc"),
      /*dynamic_shape=*/dynamic_shape,
      /*shard_dynamic_shapes=*/{shard_dynamic_shape1, shard_dynamic_shape2});

  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(*sharding));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<ConcreteSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(
                          absl::bind_front(&Client::LookupDevice, client()))));

  EXPECT_THAT(out_sharding->devices(), ElementsAreArray(sharding->devices()));
  EXPECT_THAT(out_sharding->dynamic_shape(), sharding->dynamic_shape());
  EXPECT_THAT(out_sharding->shard_dynamic_shapes(),
              ElementsAreArray(sharding->shard_dynamic_shapes()));
}

TEST_P(ShardingSerDesTest, ConcreteEvenShardingRoundTrip) {
  auto sharding =
      ConcreteEvenSharding::Create(GetDevices({0, 1}), MemoryKind("abc"),
                                   /*shape=*/Shape({10, 20}),
                                   /*shard_shape=*/Shape({5, 20}));

  TF_ASSERT_OK_AND_ASSIGN(auto serialized, Serialize(*sharding));

  TF_ASSERT_OK_AND_ASSIGN(
      auto out_sharding,
      Deserialize<ConcreteEvenSharding>(
          serialized, std::make_unique<DeserializeShardingOptions>(
                          absl::bind_front(&Client::LookupDevice, client()))));

  EXPECT_THAT(out_sharding->devices(), ElementsAreArray(sharding->devices()));
  EXPECT_THAT(out_sharding->shape(), sharding->shape());
  EXPECT_THAT(out_sharding->shard_shape(), sharding->shard_shape());
}

INSTANTIATE_TEST_SUITE_P(NumDevices, ShardingSerDesTest,
                         testing::Values(test_util::ShardingTestParam{
                             /*num_devices=*/2,
                             /*num_addressable_devices=*/2}));

}  // namespace
}  // namespace ifrt
}  // namespace xla
