/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/hlo/utils/hlo_sharding_util.h"

#include <optional>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace hlo_sharding_util {
namespace {

TEST(HloShardingUtilTest, TransposeShardingReplicated) {
  EXPECT_EQ(TransposeSharding(HloSharding::Replicate(), {0, 1, 2}),
            HloSharding::Replicate());
}

TEST(HloShardingUtilTest, TransposeShardingTiled) {
  HloSharding input = HloSharding::IotaTile({1, 2, 1, 2});
  HloSharding output =
      HloSharding::Tile(TileAssignment({2, 1, 2, 1}, {2, 2}, {1, 0}));
  EXPECT_EQ(TransposeSharding(input, {3, 0, 1, 2}), output);
}

TEST(HloShardingUtilTest, TransposeShardingWithCollapsedDimsSubgroupManual) {
  HloSharding input =
      HloSharding::Subgroup(TileAssignment({1, 2, 4}), {OpSharding::MANUAL});
  HloSharding output =
      HloSharding::Subgroup(TileAssignment({1, 1, 2, 4}), {OpSharding::MANUAL});
  EXPECT_EQ(TransposeShardingWithCollapsedDims(input, {-1, 2}, {-1, -1, 1}),
            output);
}

TEST(HloShardingUtilTest, ReshapeShardingMaximal) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 2});
  HloSharding sharding = HloSharding::AssignDevice(7);
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledInvalid) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 2});
  HloSharding sharding = HloSharding::IotaTile({1, 2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingTiledMerge) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {20, 7});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 4, 7});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 4, 7});
  HloSharding input_sharding = HloSharding::IotaTile({16, 1});
  HloSharding output_sharding = HloSharding::IotaTile({4, 4, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplitThenMerge) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 4, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 16, 7});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledArbitraryMinorDimensions) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7, 5, 3});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 15, 2, 14});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledTrivialDimensions) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {3, 1, 5, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 1, 7});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 2, 1});
  HloSharding output_sharding = HloSharding::IotaTile({1, 2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTrivialDImensionInsertedToEnd) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {8, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {8, 16, 1});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, NoopReshapeShardingEmptyTile) {
  Shape shape = ShapeUtil::MakeShape(F32, {7, 1, 1});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result = ReshapeSharding(shape, shape, sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingScalar) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeToTileDimension2D_Dim0) {
  HloSharding sharding = HloSharding::IotaTile({2, 2});
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/0, /*dims=*/{0, 1});
  EXPECT_EQ(result.tile_assignment(),
            TileAssignment((absl::Span<const int64_t>){4, 1}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension2D_Dim1) {
  HloSharding sharding = HloSharding::IotaTile({2, 2});
  HloSharding result = ReshapeToTileDimension(
      sharding, /*dim=*/1, /*dims=*/(absl::Span<const int64_t>){0, 1});
  EXPECT_EQ(result.tile_assignment(), TileAssignment({1, 4}, {2, 2}, {1, 0}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Dim0) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2});
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/0, /*dims=*/{0, 1, 2});
  EXPECT_EQ(result.tile_assignment(), TileAssignment({8, 1, 1}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Dim1) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2});
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1, 2});
  EXPECT_EQ(result.tile_assignment(),
            TileAssignment({1, 8, 1}, {2, 2, 2}, {1, 0, 2}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Dim2) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2});
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/2, /*dims=*/{0, 1, 2});
  EXPECT_EQ(result.tile_assignment(),
            TileAssignment({1, 1, 8}, {4, 2}, {1, 0}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension2D_Dim2_Batch1) {
  // Tile sharding in batch dimension, i.e.
  // sharding={devices[2,2,2]<=[8].
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2});
  // Reshape on dimensions {1, 2} only, therefore ignoring batch dimension 0.
  HloSharding result = ReshapeToTileDimension(sharding, /*dim=*/2,
                                              /*dims=*/{1, 2});
  // Expected result is {devices=[2,1,4]0,2,1,3,4,6,5,7}, i.e. the two
  // non-batch dimensions {{0, 1}, {2, 3}} and {{4, 5}, {6, 7}} are individually
  // reshaped to tile dimension 2, i.e. {{0, 2, 1, 3}}, {{4, 6, 5, 7}}.
  EXPECT_EQ(result.tile_assignment().array(),
            Array3D<int64_t>({{{0, 2, 1, 3}}, {{4, 6, 5, 7}}}));
}

TEST(HloShardingUtilTest, PropagateReshapeShardingTiledSplitPartialMatch) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {14, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 7, 4, 4});
  HloSharding input_sharding = HloSharding::IotaTile({4, 8});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({1, 1, 4, 2, 4}, {4, 8}, {1, 0}));
  HloSharding result = PropagateShardingThroughReshape(
      input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, PropagateReshapeShardingTiledMergeSplitPartialMatch) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2, 14, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 2, 7, 4, 4});
  HloSharding input_sharding = HloSharding::IotaTile({2, 2, 4, 8});
  HloSharding output_sharding = HloSharding::PartialTile(
      TileAssignment({4, 1, 1, 4, 2, 4}, {2, 2, 4, 8}, {0, 1, 3, 2}));
  HloSharding result = PropagateShardingThroughReshape(
      input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest,
     PropagateReshapeShardingTiledSplitPartialMatchManual) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {14, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 7, 4, 4});
  HloSharding input_sharding =
      HloSharding::Subgroup(TileAssignment({4, 8, 2}), {OpSharding::MANUAL});
  HloSharding output_sharding = HloSharding::Subgroup(
      TileAssignment({1, 1, 4, 2, 4, 2}, {4, 8, 2}, {1, 0, 2}),
      {OpSharding::REPLICATED, OpSharding::MANUAL});
  HloSharding result = PropagateShardingThroughReshape(
      input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, GetManualSubgroupSharding_ManualOnly) {
  TileAssignment tile_assignment({1, 2, 2});
  std::vector<OpSharding::Type> subgroup_types = {OpSharding::MANUAL};
  // Subgroup sharding {devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
  HloSharding sharding = HloSharding::Subgroup(tile_assignment, subgroup_types);

  GroupedSharding group_sharding = GetManualSubgroupSharding(sharding);

  // Expect group_sharding.sharding to be {devices=[1,2]0,1}
  EXPECT_EQ(group_sharding.sharding.tile_assignment(),
            TileAssignment((absl::Span<const int64_t>){1, 2}));

  // Expect the device groups are: {0, 2} and {1, 3}
  EXPECT_THAT(group_sharding.device_groups[0],
              testing::ElementsAreArray({0, 2}));
  EXPECT_THAT(group_sharding.device_groups[1],
              testing::ElementsAreArray({1, 3}));
}

TEST(HloShardingUtilTest, GetManualSubgroupSharding_ManualAndReplicted) {
  TileAssignment tile_assignment({1, 2, 2, 2});
  std::vector<OpSharding::Type> subgroup_types = {OpSharding::REPLICATED,
                                                  OpSharding::MANUAL};
  // Subgroup sharding
  //  {devices=[1,2,2,2]0,1,2,3,4,5,6,7 last_tile_dims={replicated, manual}}
  HloSharding sharding = HloSharding::Subgroup(tile_assignment, subgroup_types);

  GroupedSharding group_sharding = GetManualSubgroupSharding(sharding);

  EXPECT_EQ(group_sharding.sharding.ToString(),
            "{devices=[1,2,2]<=[4] last_tile_dim_replicate}");

  // Expect the device groups are: {0, 2, 4, 6} and {1, 3, 5, 7}
  EXPECT_THAT(group_sharding.device_groups[0],
              testing::ElementsAreArray({0, 2, 4, 6}));
  EXPECT_THAT(group_sharding.device_groups[1],
              testing::ElementsAreArray({1, 3, 5, 7}));
}

TEST(HloShardingUtilTest, GetManualSubgroupSharding_ReplicatedAndManual) {
  TileAssignment tile_assignment({1, 2, 2, 2});
  std::vector<OpSharding::Type> subgroup_types = {OpSharding::MANUAL,
                                                  OpSharding::REPLICATED};
  // Subgroup sharding
  //  {devices=[1,2,2,2]0,1,2,3,4,5,6,7 last_tile_dims={manual, replicated}}
  HloSharding sharding = HloSharding::Subgroup(tile_assignment, subgroup_types);

  GroupedSharding group_sharding = GetManualSubgroupSharding(sharding);

  EXPECT_EQ(group_sharding.sharding.ToString(),
            "{devices=[1,2,2]<=[4] last_tile_dim_replicate}");

  // Expect the device groups are: {0, 1, 4, 5} and {2, 3, 6, 7}
  EXPECT_THAT(group_sharding.device_groups[0],
              testing::ElementsAreArray({0, 1, 4, 5}));
  EXPECT_THAT(group_sharding.device_groups[1],
              testing::ElementsAreArray({2, 3, 6, 7}));
}

TEST(HloShardingUtilTest, UngroupSharding_ManualOnly) {
  HloSharding sharding = HloSharding::IotaTile({1, 2});
  std::vector<std::vector<int64_t>> device_groups = {{0, 2}, {1, 3}};
  std::vector<int64_t> group_dims = {2};
  std::vector<int64_t> group_dim_sizes = {2};

  auto grouped = GroupedSharding(
      std::move(device_groups),
      std::vector<int64_t>(group_dims.begin(), group_dims.end()),
      std::move(group_dim_sizes), sharding.tile_assignment().num_dimensions(),
      sharding, /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}");
}

TEST(HloShardingUtilTest, UngroupSharding_ReplicatedAndManual) {
  HloSharding sharding = HloSharding::PartialTile(TileAssignment({1, 2, 2}));
  std::vector<std::vector<int64_t>> device_groups = {{0, 2, 4, 6},
                                                     {1, 3, 5, 7}};
  std::vector<int64_t> group_dims = {3};
  std::vector<int64_t> group_dim_sizes = {2};

  auto grouped = GroupedSharding(
      std::move(device_groups),
      std::vector<int64_t>(group_dims.begin(), group_dims.end()),
      std::move(group_dim_sizes),
      sharding.tile_assignment().num_dimensions() - 1, sharding,
      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(
      ungroup_sharding.ToString(),
      "{devices=[1,2,2,2]0,2,1,3,4,6,5,7 last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, UngroupSharding_ManualAndReplicated) {
  HloSharding sharding = HloSharding::PartialTile(TileAssignment({1, 2, 2}));
  std::vector<std::vector<int64_t>> device_groups = {{0, 1, 4, 5},
                                                     {2, 3, 6, 7}};
  std::vector<int64_t> group_dims = {2};
  std::vector<int64_t> group_dim_sizes = {2};

  auto grouped = GroupedSharding(
      std::move(device_groups),
      std::vector<int64_t>(group_dims.begin(), group_dims.end()),
      std::move(group_dim_sizes),
      sharding.tile_assignment().num_dimensions() - 1, sharding,
      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(
      ungroup_sharding.ToString(),
      "{devices=[1,2,2,2]0,1,2,3,4,5,6,7 last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, UngroupSharding_Replicated) {
  HloSharding sharding = HloSharding::Replicate();

  std::vector<int64_t> group_dims = {3};
  std::vector<int64_t> group_dim_sizes = {2};

  std::vector<std::vector<int64_t>> device_groups = {{0, 1}, {2, 3}};

  auto grouped = GroupedSharding(
      std::move(device_groups),
      std::vector<int64_t>(group_dims.begin(), group_dims.end()),
      std::move(group_dim_sizes), 2, sharding,
      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,1,2,2]0,1,2,3 last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, UngroupSharding_Replicated2) {
  HloSharding sharding = HloSharding::Replicate();
  std::vector<int64_t> group_dims = {2};
  std::vector<int64_t> group_dim_sizes = {2};

  std::vector<std::vector<int64_t>> device_groups = {{0, 2}, {1, 3}};

  auto grouped = GroupedSharding(
      std::move(device_groups),
      std::vector<int64_t>(group_dims.begin(), group_dims.end()),
      std::move(group_dim_sizes), 2, sharding,
      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,1,2,2]0,2,1,3 last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, DeviceGroupsDoesNotMatch) {
  HloSharding sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  std::vector<int64_t> group_dims = {2};
  std::vector<int64_t> group_dim_sizes = {2};

  std::vector<std::vector<int64_t>> lhs_device_groups = {{0, 2, 4, 6},
                                                         {1, 3, 5, 7}};
  std::vector<int64_t> lhs_group_dims = {3};

  auto lhs = GroupedSharding(
      std::move(lhs_device_groups),
      std::vector<int64_t>(lhs_group_dims.begin(), lhs_group_dims.end()),
      group_dim_sizes, 2, sharding,
      /*subgroup_manual=*/true);

  std::vector<std::vector<int64_t>> rhs_device_groups = {{0, 1, 4, 5},
                                                         {2, 3, 6, 7}};
  std::vector<int64_t> rhs_group_dims = {2};

  auto rhs = GroupedSharding(
      std::move(rhs_device_groups),
      std::vector<int64_t>(rhs_group_dims.begin(), rhs_group_dims.end()),
      group_dim_sizes, 2, sharding,
      /*subgroup_manual=*/true);

  EXPECT_FALSE(DeviceGroupsAreMatch(lhs, rhs));
}

TEST(HloShardingUtilTest, DeviceGroupsMatch) {
  HloSharding lhs_sharding = HloSharding::Replicate();
  std::vector<int64_t> group_dims = {2};
  std::vector<int64_t> group_dim_sizes = {2};
  std::vector<std::vector<int64_t>> device_groups = {{0, 2}, {1, 3}};

  auto lhs = GroupedSharding(
      device_groups, std::vector<int64_t>(group_dims.begin(), group_dims.end()),
      group_dim_sizes, 2, lhs_sharding,
      /*subgroup_manual=*/true);

  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  auto rhs = GroupedSharding(
      device_groups, std::vector<int64_t>(group_dims.begin(), group_dims.end()),
      group_dim_sizes, 2, rhs_sharding,
      /*subgroup_manual=*/true);

  EXPECT_TRUE(DeviceGroupsAreMatch(lhs, rhs));
}

TEST(HloShardingUtilTest, IsSubShardingTiledReplicated) {
  HloSharding rhs_sharding = HloSharding::Replicate();
  HloSharding lhs_sharding = HloSharding::IotaTile({4, 1});
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingReplicatedTiled) {
  HloSharding rhs_sharding = HloSharding::IotaTile({4, 1});
  HloSharding lhs_sharding = HloSharding::Replicate();
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingTiledPartialReplicated) {
  HloSharding rhs_sharding = HloSharding::Replicate();
  HloSharding lhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingReplicatedTiledPartial) {
  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  HloSharding lhs_sharding = HloSharding::Replicate();
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingPartialTiledTiled) {
  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4, 1});
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingIncompatibleTiled) {
  HloSharding rhs_sharding = HloSharding::IotaTile({4, 1});
  HloSharding lhs_sharding = HloSharding::IotaTile({1, 4});
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingIncompatibleShapeTiledPartialTiled) {
  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4, 1});
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingCompatibleShapeTiledPartialTiled) {
  HloSharding rhs_sharding =
      HloSharding::PartialTile(TileAssignment({2, 1, 2}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4, 1});
  Shape shape = ShapeUtil::MakeShape(F32, {128, 253});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSortOperandShardingMovableRankTwoOneFreeDim) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {8, 128}), 1);
  iota.set_sharding(HloSharding::IotaTile({1, 2}));
  EXPECT_TRUE(IsSortOperandShardingMovable(&iota, 1));
}

TEST(HloShardingUtilTest, IsSortOperandShardingMovableRankTwoNoFreeDims) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {8, 128}), 1);
  iota.set_sharding(HloSharding::IotaTile({2, 2}));
  EXPECT_FALSE(IsSortOperandShardingMovable(&iota, 1));
}

TEST(HloShardingUtilTest, IsSortOperandShardingMovableRankOne) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {1024}), 1);
  iota.set_sharding(
      HloSharding::Tile(TileAssignment(std::initializer_list<int64_t>{2})));
  EXPECT_FALSE(IsSortOperandShardingMovable(&iota, 0));
}

TEST(HloShardingUtilTest, IsSortOperandShardingMovableNoSharding) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {1024}), 1);
  EXPECT_FALSE(IsSortOperandShardingMovable(&iota, 0));
}

TEST(HloShardingUtilTest, IsSortOperandShardingMovableReplicated) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {8, 128}), 1);
  iota.set_sharding(HloSharding::Replicate());
  EXPECT_FALSE(IsSortOperandShardingMovable(&iota, 1));
}

TEST(HloShardingUtilTest, IsSortOperandShardingMovableSortDimUnsharded) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {8, 128}), 1);
  iota.set_sharding(HloSharding::IotaTile({1, 2}));
  EXPECT_FALSE(IsSortOperandShardingMovable(&iota, 0));
}
}  // namespace
}  // namespace hlo_sharding_util
}  // namespace xla
