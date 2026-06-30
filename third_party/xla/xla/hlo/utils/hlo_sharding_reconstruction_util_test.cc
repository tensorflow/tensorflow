/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_sharding_reconstruction_util.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Key;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

TEST(HloShardingReconstructionUtilTest, FactorManualShardingTiled) {
  HloSharding sharding = HloSharding::IotaTile({2});

  auto literal0 =
      std::make_shared<Literal>(LiteralUtil::CreateR1<float>({1.0f}));
  auto literal1 =
      std::make_shared<Literal>(LiteralUtil::CreateR1<float>({2.0f}));

  std::vector<ShardTensor> shards = {
      {0, literal0},
      {1, literal1},
  };

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0)));
  EXPECT_EQ(info.unshard_sharding, sharding);
}

TEST(HloShardingReconstructionUtilTest, FactorManualShardingPureManual) {
  HloSharding sharding = HloSharding::Manual();

  auto literal0 =
      std::make_shared<Literal>(LiteralUtil::CreateR1<float>({1.0f}));
  auto literal1 =
      std::make_shared<Literal>(LiteralUtil::CreateR1<float>({2.0f}));

  std::vector<ShardTensor> shards = {
      {0, literal0},
      {1, literal1},
  };

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0), Key(1)));
  EXPECT_TRUE(info.has_manual_sharding);
  EXPECT_TRUE(info.unshard_sharding.IsReplicated());
}

TEST(HloShardingReconstructionUtilTest, FactorManualShardingNamedManual) {
  // mesh = (x=2, y=2), manual_axes={x}, tiled_axes={y} sharding dim 0
  Mesh mesh({2, 2}, {"x", "y"});
  NamedSharding ns(
      mesh,
      {NamedSharding::DimensionSharding({AxisRef(1)}, /*is_closed=*/true)},
      /*replicated_axes=*/{}, /*unreduced_axes=*/{},
      /*manual_axes=*/{AxisRef(0)});
  HloSharding sharding(ns);

  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  std::vector<ShardTensor> shards = {
      {0, dummy},
      {1, dummy},
      {2, dummy},
      {3, dummy},
  };

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0), Key(1)));
  EXPECT_THAT(info.manual_shard_groups.at(0),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 0),
                                   Field(&ShardTensor::logical_shard_id, 1)));
  EXPECT_THAT(info.manual_shard_groups.at(1),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 2),
                                   Field(&ShardTensor::logical_shard_id, 3)));

  EXPECT_TRUE(info.has_manual_sharding);
  // unshard_sharding should be NamedSharding without manual axes
  EXPECT_THAT(info.unshard_sharding.named_sharding().manual_axes(), IsEmpty());
  EXPECT_THAT(info.unshard_sharding.named_sharding().dim_shardings(),
              SizeIs(1));
}

TEST(HloShardingReconstructionUtilTest, FactorManualShardingNamedNoManual) {
  // mesh = (x=2, y=2), manual_axes={}, tiled_axes={y} sharding dim 0
  Mesh mesh({2, 2}, {"x", "y"});
  NamedSharding ns(
      mesh,
      {NamedSharding::DimensionSharding({AxisRef(1)}, /*is_closed=*/true)},
      /*replicated_axes=*/{}, /*unreduced_axes=*/{},
      /*manual_axes=*/{});
  HloSharding sharding(ns);

  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  std::vector<ShardTensor> shards = {
      {0, dummy},
      {1, dummy},
      {2, dummy},
      {3, dummy},
  };

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0)));
  EXPECT_THAT(info.manual_shard_groups.at(0), SizeIs(4));
  EXPECT_FALSE(info.has_manual_sharding);
  EXPECT_EQ(info.unshard_sharding, sharding);
}

TEST(HloShardingReconstructionUtilTest,
     FactorManualShardingNamedManualUnknownShard) {
  // mesh = (x=2, y=2), manual_axes={x}, tiled_axes={y} sharding dim 0
  Mesh mesh({2, 2}, {"x", "y"});
  NamedSharding ns(
      mesh,
      {NamedSharding::DimensionSharding({AxisRef(1)}, /*is_closed=*/true)},
      /*replicated_axes=*/{}, /*unreduced_axes=*/{},
      /*manual_axes=*/{AxisRef(0)});
  HloSharding sharding(ns);

  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  // Add a shard with logical_shard_id = 4 which is not in the mesh (size 4 ->
  // IDs 0-3)
  std::vector<ShardTensor> shards = {
      {0, dummy},
      {4, dummy},
  };

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  // Only shard 0 should be processed. Shard 4 should be skipped.
  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0)));
  EXPECT_THAT(info.manual_shard_groups.at(0),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 0)));
}

TEST(HloShardingReconstructionUtilTest, FactorManualShardingSubgroupMixed) {
  // mesh = (x=2, y=2), manual_axes={x}, dim 0 sharded by y
  Mesh mesh({2, 2}, {"x", "y"});
  NamedSharding ns(
      mesh,
      {NamedSharding::DimensionSharding({AxisRef(1)}, /*is_closed=*/true),
       NamedSharding::DimensionSharding({}, /*is_closed=*/true)},
      /*replicated_axes=*/{}, /*unreduced_axes=*/{},
      /*manual_axes=*/{AxisRef(0)});
  HloSharding sharding(ns);

  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  std::vector<ShardTensor> shards;
  for (int i = 0; i < 4; ++i) {
    shards.push_back({i, dummy});
  }

  // Mesh looks like:
  // x=0: y=0 (device 0), y=1 (device 1)  -> manual_id 0
  // x=1: y=0 (device 2), y=1 (device 3)  -> manual_id 1

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0), Key(1)));
  EXPECT_THAT(info.manual_shard_groups.at(0),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 0),
                                   Field(&ShardTensor::logical_shard_id, 1)));
  EXPECT_THAT(info.manual_shard_groups.at(1),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 2),
                                   Field(&ShardTensor::logical_shard_id, 3)));

  EXPECT_THAT(info.unshard_sharding.named_sharding().manual_axes(), IsEmpty());
  EXPECT_THAT(
      info.unshard_sharding.named_sharding().dim_shardings().at(0).axes(),
      ElementsAre(AxisRef(1)));
}

TEST(HloShardingReconstructionUtilTest, UnshardLiteralReplicated) {
  HloSharding sharding = HloSharding::Replicate();
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {2});

  auto lit =
      std::make_shared<Literal>(LiteralUtil::CreateR1<float>({1.0f, 2.0f}));
  std::vector<ShardTensor> shards = {{0, lit}};

  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          UnshardLiteral(shards, sharding, unsharded_shape));

  EXPECT_EQ(result.Get<float>({0}), 1.0f);
  EXPECT_EQ(result.Get<float>({1}), 2.0f);
}

TEST(HloShardingReconstructionUtilTest, UnshardLiteralTiled) {
  HloSharding sharding = HloSharding::IotaTile({2, 2});
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {4, 4});

  auto create_shard = [](float val) {
    auto lit = std::make_shared<Literal>(ShapeUtil::MakeShape(F32, {2, 2}));
    lit->PopulateWithValue(val);
    return lit;
  };

  std::vector<ShardTensor> shards = {
      {0, create_shard(0.0f)},
      {1, create_shard(1.0f)},
      {2, create_shard(2.0f)},
      {3, create_shard(3.0f)},
  };

  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          UnshardLiteral(shards, sharding, unsharded_shape));

  EXPECT_EQ(result.Get<float>({0, 0}), 0.0f);
  EXPECT_EQ(result.Get<float>({0, 2}), 1.0f);
  EXPECT_EQ(result.Get<float>({2, 0}), 2.0f);
  EXPECT_EQ(result.Get<float>({2, 2}), 3.0f);
}

TEST(HloShardingReconstructionUtilTest, FullReconstructionFlow) {
  Mesh mesh({2, 2}, {"x", "y"});
  NamedSharding ns(
      mesh,
      {NamedSharding::DimensionSharding({AxisRef(1)}, /*is_closed=*/true)},
      /*replicated_axes=*/{}, /*unreduced_axes=*/{},
      /*manual_axes=*/{AxisRef(0)});
  HloSharding sharding(ns);

  auto create_shard = [](int64_t id, float val) {
    auto lit = std::make_shared<Literal>(ShapeUtil::MakeShape(F32, {2}));
    lit->PopulateWithValue(val);
    return lit;
  };

  std::vector<ShardTensor> shards;
  shards.push_back({0, create_shard(0, 0.0f)});
  shards.push_back({1, create_shard(1, 1.0f)});
  shards.push_back({2, create_shard(2, 2.0f)});
  shards.push_back({3, create_shard(3, 3.0f)});

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  Shape grouped_unsharded_shape = ShapeUtil::MakeShape(F32, {4});

  TF_ASSERT_OK_AND_ASSIGN(
      Literal group0_lit,
      UnshardLiteral(info.manual_shard_groups.at(0), info.unshard_sharding,
                     grouped_unsharded_shape));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal group1_lit,
      UnshardLiteral(info.manual_shard_groups.at(1), info.unshard_sharding,
                     grouped_unsharded_shape));

  EXPECT_EQ(group0_lit.Get<float>({0}), 0.0f);
  EXPECT_EQ(group0_lit.Get<float>({2}), 1.0f);
  EXPECT_EQ(group1_lit.Get<float>({0}), 2.0f);
  EXPECT_EQ(group1_lit.Get<float>({2}), 3.0f);
}

TEST(HloShardingReconstructionUtilTest, FactorManualShardingSubAxis) {
  // mesh = (x=4), manual_axis={x_sub(pre=1, size=2)},
  // tiled_axis={x_sub(pre=2, size=2)}
  Mesh mesh({4}, {"x"});
  AxisRef manual_x(0, {1, 2});
  AxisRef tiled_x(0, {2, 2});

  NamedSharding ns(
      mesh, {NamedSharding::DimensionSharding({tiled_x}, /*is_closed=*/true)},
      /*replicated_axes=*/{}, /*unreduced_axes=*/{},
      /*manual_axes=*/{manual_x});
  HloSharding sharding(ns);

  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  std::vector<ShardTensor> shards;
  for (int i = 0; i < 4; ++i) {
    shards.push_back({i, dummy});
  }

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0), Key(1)));
  // manual_id 0: x=0 (shard 0), x=2 (shard 2)
  EXPECT_THAT(info.manual_shard_groups.at(0),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 0),
                                   Field(&ShardTensor::logical_shard_id, 2)));
  // manual_id 1: x=1 (shard 1), x=3 (shard 3)
  EXPECT_THAT(info.manual_shard_groups.at(1),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 1),
                                   Field(&ShardTensor::logical_shard_id, 3)));

  EXPECT_THAT(info.unshard_sharding.named_sharding().manual_axes(), IsEmpty());
  EXPECT_THAT(
      info.unshard_sharding.named_sharding().dim_shardings().at(0).axes(),
      ElementsAre(tiled_x));
}

TEST(HloShardingReconstructionUtilTest, FactorManualShardingEmptyShards) {
  HloSharding sharding = HloSharding::IotaTile({2});
  std::vector<ShardTensor> shards;
  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));
  EXPECT_THAT(info.manual_shard_groups, SizeIs(1));
  EXPECT_THAT(info.manual_shard_groups.at(0), IsEmpty());
}

TEST(HloShardingReconstructionUtilTest, FactorManualShardingSubgroupV2) {
  Array<int64_t> tile_assignment({2, 2}, {0, 1, 2, 3});

  HloSharding sharding =
      HloSharding::Subgroup(tile_assignment, {OpSharding::MANUAL});

  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  std::vector<ShardTensor> shards;
  for (int i = 0; i < 4; ++i) {
    shards.push_back({i, dummy});
  }

  TF_ASSERT_OK_AND_ASSIGN(ManualShardingInfo info,
                          FactorManualSharding(shards, sharding));

  EXPECT_THAT(info.manual_shard_groups, UnorderedElementsAre(Key(0), Key(1)));
  EXPECT_THAT(info.manual_shard_groups.at(0),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 0),
                                   Field(&ShardTensor::logical_shard_id, 2)));
  EXPECT_THAT(info.manual_shard_groups.at(1),
              UnorderedElementsAre(Field(&ShardTensor::logical_shard_id, 1),
                                   Field(&ShardTensor::logical_shard_id, 3)));
  EXPECT_TRUE(info.has_manual_sharding);
  EXPECT_EQ(info.unshard_sharding,
            HloSharding::Subgroup(tile_assignment, {OpSharding::REPLICATED}));
}

TEST(HloShardingReconstructionUtilTest, UnshardLiteralEmptyShards) {
  HloSharding sharding = HloSharding::IotaTile({2});
  std::vector<ShardTensor> shards;
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {4});
  auto status = UnshardLiteral(shards, sharding, unsharded_shape);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(HloShardingReconstructionUtilTest, UnshardLiteralManualError) {
  HloSharding sharding = HloSharding::Manual();
  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  std::vector<ShardTensor> shards = {{0, dummy}};
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {2});
  auto status = UnshardLiteral(shards, sharding, unsharded_shape);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              testing::HasSubstr(
                  "One should not call UnshardLiteral on a manual sharding"));
}

TEST(HloShardingReconstructionUtilTest, UnshardLiteralTuple) {
  HloSharding sharding = HloSharding::Tuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {2})}),
      {HloSharding::IotaTile({2})});
  auto dummy = std::make_shared<Literal>(LiteralUtil::CreateR1<float>({0.0f}));
  std::vector<ShardTensor> shards = {{0, dummy}};
  Shape unsharded_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {2})});
  auto status = UnshardLiteral(shards, sharding, unsharded_shape);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(HloShardingReconstructionUtilTest, UnshardLiteralPadded) {
  HloSharding sharding = HloSharding::IotaTile({2});
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {3});

  auto create_shard = [](float val, int size) {
    auto lit = std::make_shared<Literal>(ShapeUtil::MakeShape(F32, {size}));
    lit->PopulateWithValue(val);
    return lit;
  };

  std::vector<ShardTensor> shards = {
      {0, create_shard(1.0f, 2)},
      {1, create_shard(2.0f, 2)},
  };

  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          UnshardLiteral(shards, sharding, unsharded_shape));

  EXPECT_EQ(result.Get<float>({0}), 1.0f);
  EXPECT_EQ(result.Get<float>({1}), 1.0f);
  EXPECT_EQ(result.Get<float>({2}), 2.0f);
}

TEST(HloShardingReconstructionUtilTest, UnshardLiteralManualSubgroup) {
  Array<int64_t> tile_assignment({2, 2}, {0, 1, 2, 3});

  HloSharding sharding =
      HloSharding::Subgroup(tile_assignment, {OpSharding::MANUAL});
  Shape unsharded_shape = ShapeUtil::MakeShape(F32, {4});

  auto create_shard = [](float val) {
    auto lit = std::make_shared<Literal>(ShapeUtil::MakeShape(F32, {2}));
    lit->PopulateWithValue(val);
    return lit;
  };

  std::vector<ShardTensor> shards = {
      {0, create_shard(1.0f)},
      {1, create_shard(2.0f)},
      {2, create_shard(3.0f)},
      {3, create_shard(4.0f)},
  };

  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          UnshardLiteral(shards, sharding, unsharded_shape));

  EXPECT_EQ(result.Get<float>({0}), 1.0f);
  EXPECT_EQ(result.Get<float>({1}), 1.0f);
  EXPECT_EQ(result.Get<float>({2}), 3.0f);
  EXPECT_EQ(result.Get<float>({3}), 3.0f);
}

}  // namespace
}  // namespace xla
