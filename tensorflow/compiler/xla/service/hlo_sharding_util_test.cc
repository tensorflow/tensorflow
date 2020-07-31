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

#include "tensorflow/compiler/xla/service/hlo_sharding_util.h"

#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace hlo_sharding_util {
namespace {

TEST(HloShardingUtilTest, TransposeShardingReplicated) {
  EXPECT_EQ(TransposeSharding(HloSharding::Replicate(), {0, 1, 2}),
            HloSharding::Replicate());
}

TEST(HloShardingUtilTest, TransposeShardingTiled) {
  HloSharding input = HloSharding::Tile(Array4D<int64>({{{{0, 1}}, {{2, 3}}}}));
  HloSharding output =
      HloSharding::Tile(Array4D<int64>({{{{0}, {2}}}, {{{1}, {3}}}}));
  EXPECT_EQ(TransposeSharding(input, {3, 0, 1, 2}), output);
}

TEST(HloShardingUtilTest, ReshapeShardingMaximal) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 2});
  HloSharding sharding = HloSharding::AssignDevice(7);
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledInvalid) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 2});
  HloSharding sharding = HloSharding::Tile(Array3D<int64>({{{0}, {1}}}));
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingTiledMerge) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {20, 7});
  HloSharding input_sharding =
      HloSharding::Tile(Array3D<int64>({{{0}}, {{1}}}));
  HloSharding output_sharding = HloSharding::Tile(Array2D<int64>({{0}, {1}}));
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 4, 7});
  HloSharding input_sharding = HloSharding::Tile(Array2D<int64>({{0}, {1}}));
  HloSharding output_sharding =
      HloSharding::Tile(Array3D<int64>({{{0}}, {{1}}}));
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 4, 7});
  Array2D<int64> tile(16, 1);
  tile.FillIota(0);
  HloSharding input_sharding = HloSharding::Tile(tile);
  tile.Reshape({4, 4, 1});
  HloSharding output_sharding = HloSharding::Tile(tile);
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplitThenMerge) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 4, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 16, 7});
  HloSharding input_sharding =
      HloSharding::Tile(Array3D<int64>({{{0}}, {{1}}}));
  HloSharding output_sharding =
      HloSharding::Tile(Array3D<int64>({{{0}}, {{1}}}));
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledArbitraryMinorDimensions) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7, 5, 3});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 15, 2, 14});
  Array<int64> sharding_array({2, 1, 1, 1});
  sharding_array(0, 0, 0, 0) = 0;
  sharding_array(1, 0, 0, 0) = 1;
  HloSharding sharding = HloSharding::Tile(sharding_array);
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledTrivialDimensions) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {3, 1, 5, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 1, 7});
  HloSharding input_sharding =
      HloSharding::Tile(Array4D<int64>({{{{0}, {1}}}}));
  HloSharding output_sharding =
      HloSharding::Tile(Array4D<int64>({{{{0}}, {{1}}}}));
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTrivialDImensionInsertedToEnd) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {8, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {8, 16, 1});
  HloSharding input_sharding = HloSharding::Tile(Array2D<int64>({{0}, {1}}));
  HloSharding output_sharding =
      HloSharding::Tile(Array3D<int64>({{{0}}, {{1}}}));
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, NoopReshapeShardingEmptyTile) {
  Shape shape = ShapeUtil::MakeShape(F32, {7, 1, 1});
  HloSharding sharding = HloSharding::Tile(Array3D<int64>({{{0}, {1}}}));
  absl::optional<HloSharding> result = ReshapeSharding(shape, shape, sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingScalar) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {});
  HloSharding sharding = HloSharding::Tile(Array3D<int64>({{{0}, {1}}}));
  absl::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeToTileDimension2D_Dim0) {
  HloSharding sharding = HloSharding::Tile(Array2D<int64>({{0, 1}, {2, 3}}));
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/0, /*dims=*/{0, 1});
  EXPECT_EQ(result.tile_assignment(), Array2D<int64>({{0}, {1}, {2}, {3}}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension2D_Dim1) {
  HloSharding sharding = HloSharding::Tile(Array2D<int64>({{0, 1}, {2, 3}}));
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1});
  EXPECT_EQ(result.tile_assignment(), Array2D<int64>({{0, 2, 1, 3}}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Dim0) {
  HloSharding sharding =
      HloSharding::Tile(Array3D<int64>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}));
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/0, /*dims=*/{0, 1, 2});
  EXPECT_EQ(
      result.tile_assignment(),
      Array3D<int64>({{{0}}, {{1}}, {{2}}, {{3}}, {{4}}, {{5}}, {{6}}, {{7}}}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Dim1) {
  HloSharding sharding =
      HloSharding::Tile(Array3D<int64>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}));
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1, 2});
  EXPECT_EQ(result.tile_assignment(),
            Array3D<int64>({{{0}, {1}, {4}, {5}, {2}, {3}, {6}, {7}}}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Dim2) {
  HloSharding sharding =
      HloSharding::Tile(Array3D<int64>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}));
  HloSharding result =
      ReshapeToTileDimension(sharding, /*dim=*/2, /*dims=*/{0, 1, 2});
  EXPECT_EQ(result.tile_assignment(),
            Array3D<int64>({{{0, 2, 4, 6, 1, 3, 5, 7}}}));
}

TEST(HloShardingUtilTest, ReshapeToTileDimension2D_Dim2_Batch1) {
  // Tile sharding in batch dimension, i.e.
  // sharding={devices[2,2,2]0,1,2,3,4,5,6,7,8}.
  HloSharding sharding =
      HloSharding::Tile(Array3D<int64>({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}}));
  // Reshape on dimensions {1, 2} only, therefore ignoring batch dimension 0.
  HloSharding result = ReshapeToTileDimension(sharding, /*dim=*/2,
                                              /*dims=*/{1, 2});
  // Expected result is {devices=[2,1,4]0,2,1,3,4,6,5,7}, i.e. the two
  // non-batch dimensions {{0, 1}, {2, 3}} and {{4, 5}, {6, 7}} are individually
  // reshaped to tile dimension 2, i.e. {{0, 2, 1, 3}}, {{4, 6, 5, 7}}.
  EXPECT_EQ(result.tile_assignment(),
            Array3D<int64>({{{0, 2, 1, 3}}, {{4, 6, 5, 7}}}));
}

}  // namespace
}  // namespace hlo_sharding_util
}  // namespace xla
