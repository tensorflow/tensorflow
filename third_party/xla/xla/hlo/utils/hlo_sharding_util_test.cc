/* Copyright 2020 The OpenXLA Authors.

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

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace hlo_sharding_util {
namespace {

TEST(HloShardingUtilTest, MergeShardingIfCompatible1) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}, {16, 8}, {1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 4, 2, 4}, {4, 4, 8}, {0, 2, 1})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible2) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 2, 4, 16}, {16, 8}, {1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 2, 4, 4}, {4, 4, 8}, {0, 2, 1})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible3) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({4, 2, 1, 16}, {16, 8}, {1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({1, 1, 4, 32}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 2, 4, 4}, {16, 8}, {1, 0})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible4) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}, {16, 8}, {1, 0}));
  HloSharding dst =
      HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}, {4, 32}, {1, 0}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 4, 2, 4}, {4, 32}, {1, 0})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible5) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}, {16, 8}, {1, 0}));
  HloSharding dst =
      HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}, {32, 4}, {1, 0}));
  EXPECT_FALSE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible6) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}));
  EXPECT_FALSE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible7) {
  HloSharding to_merge = HloSharding::PartialTile(
      TileAssignment({2, 1, 2, 2}, {2, 2, 2}, {2, 1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({1, 2, 1, 4}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
  EXPECT_EQ(dst,
            HloSharding::Tile(TileAssignment({2, 2, 2}, {2, 2, 2}, {2, 0, 1})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible8) {
  HloSharding to_merge = HloSharding::PartialTile(TileAssignment({2, 1, 4}));
  HloSharding dst =
      HloSharding::PartialTile(TileAssignment({1, 4, 2}, {2, 2, 2}, {2, 1, 0}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
  EXPECT_EQ(dst,
            HloSharding::Tile(TileAssignment({2, 4}, {2, 2, 2}, {0, 2, 1})));
}

TEST(HloShardingUtilTest, TransposeShardingReplicated) {
  EXPECT_EQ(TransposeSharding(HloSharding::Replicate(), {0, 1, 2}),
            HloSharding::Replicate());
}

TEST(HloShardingUtilTest, TransposeShardingTiled) {
  HloSharding input = HloSharding::IotaTile({1, 2, 1, 2});
  HloSharding output = HloSharding::IotaTile({2, 1, 2, 1}, {2, 2}, {1, 0});
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

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {36});
  Shape output_shape = ShapeUtil::MakeShape(F32, {6, 6});
  HloSharding input_sharding = HloSharding::IotaTile({4});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 1, 2}));
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

TEST(HloShardingUtilTest, ReshapeShardingTrivialDimensionInsertedToEnd) {
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

TEST(HloShardingUtilTest, ReshapeShardingSuffixShapeSizeOne1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {64, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {64, 1});
  HloSharding input_sharding = HloSharding::IotaTile({4, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({4, 1});

  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);

  result = ReshapeSharding(output_shape, input_shape, output_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), input_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingSuffixShapeSizeOne2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {64, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {64, 1});
  HloSharding input_sharding = HloSharding::IotaTile({4, 2, 8});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({4, 2, 8}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingSuffixShapeSizeOne3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {64, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {64, 1, 1});
  HloSharding input_sharding = HloSharding::IotaTile({4, 2});
  HloSharding output_sharding = HloSharding::IotaTile({4, 2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingPrefixShapeSizeOne1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 64});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 64});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 4});
  HloSharding output_sharding = HloSharding::IotaTile({1, 4});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);

  result = ReshapeSharding(output_shape, input_shape, output_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), input_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingPrefixShapeSizeOne2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 64});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 64});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);

  result = ReshapeSharding(output_shape, input_shape, output_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), input_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {6, 2, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 3, 5});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 5});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5, 7, 11});
  Shape output_shape = ShapeUtil::MakeShape(F32, {10, 21, 11});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1, 1, 13});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 13});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 10});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 5});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose4) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5, 7, 11, 13, 17, 19});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 2, 55, 91, 19, 17});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 5, 1, 1, 13, 1, 1});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({1, 1, 5, 1, 1, 1, 13}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), output_sharding);
}

TEST(HloShardingUtilTest, ReshapeToTileDimension2D) {
  // The two sharding in the vector are the same. They will be processed in
  // different branches in ReshapeToTileDimension.
  std::vector<HloSharding> shardings = {HloSharding::IotaTile({2, 2}),
                                        HloSharding::Tile({{0, 1}, {2, 3}})};

  for (const HloSharding& sharding : shardings) {
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/0, /*dims=*/{0, 1})
                  .tile_assignment(),
              TileAssignment((absl::Span<const int64_t>){4, 1}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1})
                  .tile_assignment(),
              TileAssignment({1, 4}, {2, 2}, {1, 0}));
  }
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Case1) {
  std::vector<HloSharding> shardings = {
      HloSharding::IotaTile({2, 2, 2}),
      HloSharding::Tile({{{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}})};

  for (const HloSharding& sharding : shardings) {
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/0, /*dims=*/{0, 1, 2})
                  .tile_assignment(),
              TileAssignment({8, 1, 1}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1, 2})
                  .tile_assignment(),
              TileAssignment({1, 8, 1}, {2, 2, 2}, {1, 0, 2}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/2, /*dims=*/{0, 1, 2})
                  .tile_assignment(),
              TileAssignment({1, 1, 8}, {4, 2}, {1, 0}));

    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/2,
                                     /*dims=*/{1, 2})
                  .tile_assignment(),
              TileAssignment({2, 1, 4}, {2, 2, 2}, {0, 2, 1}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/0,
                                     /*dims=*/{0, 2})
                  .tile_assignment(),
              TileAssignment({4, 2, 1}, {2, 2, 2}, {1, 0, 2}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/2,
                                     /*dims=*/{0, 2})
                  .tile_assignment(),
              TileAssignment({1, 2, 4}, {2, 2, 2}, {1, 2, 0}));
  }
}

TEST(HloShardingUtilTest, ReshapeToTileDimension3D_Case2) {
  // The input sharding has a complicated device list.
  std::vector<HloSharding> shardings = {
      HloSharding::IotaTile({2, 2, 2}, {4, 2}, {1, 0}),
      HloSharding::Tile({{{0, 2}, {4, 6}}, {{1, 3}, {5, 7}}})};
  for (const HloSharding& sharding : shardings) {
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/0, /*dims=*/{0, 1, 2})
                  .tile_assignment(),
              TileAssignment({8, 1, 1}, {4, 2}, {1, 0}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1, 2})
                  .tile_assignment(),
              TileAssignment({1, 8, 1}, {2, 2, 2}, {0, 2, 1}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/2, /*dims=*/{0, 1, 2})
                  .tile_assignment(),
              TileAssignment({1, 1, 8}, {2, 4}, {1, 0}));
  }
}

TEST(HloShardingUtilTest, ReshapeToTileDimension4D) {
  HloSharding sharding1 = HloSharding::IotaTile({2, 3, 5, 7});
  HloSharding sharding2 =
      HloSharding::Tile(sharding1.tile_assignment().array());
  std::vector<HloSharding> shardings = {sharding1, sharding2};

  for (const HloSharding& sharding : shardings) {
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1})
                  .tile_assignment(),
              TileAssignment({1, 6, 5, 7}, {2, 3, 5, 7}, {2, 3, 1, 0}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{1, 2})
                  .tile_assignment(),
              TileAssignment({2, 15, 1, 7}, {2, 3, 5, 7}, {0, 3, 1, 2}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{1, 3})
                  .tile_assignment(),
              TileAssignment({2, 21, 5, 1}, {2, 3, 5, 7}, {0, 2, 1, 3}));

    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1, 2})
                  .tile_assignment(),
              TileAssignment({1, 30, 1, 7}, {2, 3, 5, 7}, {3, 1, 0, 2}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1, 3})
                  .tile_assignment(),
              TileAssignment({1, 42, 5, 1}, {2, 3, 5, 7}, {2, 1, 0, 3}));
    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{1, 2, 3})
                  .tile_assignment(),
              TileAssignment({2, 105, 1, 1}, {2, 3, 5, 7}, {0, 1, 2, 3}));

    EXPECT_EQ(ReshapeToTileDimension(sharding, /*dim=*/1, /*dims=*/{0, 1, 2, 3})
                  .tile_assignment(),
              TileAssignment({1, 210, 1, 1}, {2, 3, 5, 7}, {1, 0, 2, 3}));
  }
}

TEST(HloShardingUtilTest, PropagateReshapeShardingTranspose1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {6, 4});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 2, 3, 2});
  HloSharding input_sharding = HloSharding::IotaTile({6, 1});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 1, 1, 1, 3}));
  HloSharding result = PropagateShardingThroughReshape(
      input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, PropagateReshapeShardingTranspose2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {6, 4});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 6});
  HloSharding input_sharding = HloSharding::IotaTile({6, 1});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 1, 3}));
  HloSharding result = PropagateShardingThroughReshape(
      input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, PropagateReshapeShardingTranspose3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 6, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 2, 2, 5, 3});
  HloSharding input_sharding = HloSharding::IotaTile({2, 6, 1});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 1, 2, 1, 1, 3}));
  HloSharding result = PropagateShardingThroughReshape(
      input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
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

TEST(HloShardingUtilTest, MergeManualSubgroupSharding) {
  TileAssignment tile_assignment({16, 4});
  std::vector<OpSharding::Type> subgroup_types = {OpSharding::MANUAL,
                                                  OpSharding::REPLICATED};
  // Subgroup sharding
  //  {devices=[16,4]<=[64] last_tile_dims={manual, replicated}}
  HloSharding dst = HloSharding::Subgroup(tile_assignment, subgroup_types);
  HloSharding to_merge = dst;
  EXPECT_FALSE(MergeShardingIfCompatible(to_merge, dst.NumTiles() + 1, &dst));
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
              ::testing::ElementsAreArray({0, 2}));
  EXPECT_THAT(group_sharding.device_groups[1],
              ::testing::ElementsAreArray({1, 3}));
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
              ::testing::ElementsAreArray({0, 2, 4, 6}));
  EXPECT_THAT(group_sharding.device_groups[1],
              ::testing::ElementsAreArray({1, 3, 5, 7}));
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
              ::testing::ElementsAreArray({0, 1, 4, 5}));
  EXPECT_THAT(group_sharding.device_groups[1],
              ::testing::ElementsAreArray({2, 3, 6, 7}));
}

TEST(HloShardingUtilTest, UngroupSharding_ManualOnly) {
  HloSharding sharding = HloSharding::IotaTile({1, 2});
  std::vector<std::vector<int64_t>> device_groups = {{0, 2}, {1, 3}};
  DimensionVector group_dims = {2};
  DimensionVector group_dim_sizes = {2};

  auto grouped = GroupedSharding(
      std::move(device_groups), std::move(group_dims),
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
  DimensionVector group_dims = {3};
  DimensionVector group_dim_sizes = {2};

  auto grouped =
      GroupedSharding(std::move(device_groups), std::move(group_dims),
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
  DimensionVector group_dims = {2};
  DimensionVector group_dim_sizes = {2};

  auto grouped =
      GroupedSharding(std::move(device_groups), std::move(group_dims),
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

  DimensionVector group_dims = {3};
  DimensionVector group_dim_sizes = {2};

  std::vector<std::vector<int64_t>> device_groups = {{0, 1}, {2, 3}};

  auto grouped =
      GroupedSharding(std::move(device_groups), std::move(group_dims),
                      std::move(group_dim_sizes), 2, sharding,
                      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,1,2,2]0,1,2,3 last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, UngroupSharding_Replicated2) {
  HloSharding sharding = HloSharding::Replicate();
  DimensionVector group_dims = {2};
  DimensionVector group_dim_sizes = {2};

  std::vector<std::vector<int64_t>> device_groups = {{0, 2}, {1, 3}};

  auto grouped =
      GroupedSharding(std::move(device_groups), std::move(group_dims),
                      std::move(group_dim_sizes), 2, sharding,
                      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,1,2,2]0,2,1,3 last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, GroupedAndUngroupedReplicatedSharding) {
  GroupedSharding group_sharding = GetGroupedReplicatedSharding(
      /*num_groups=*/3, /*num_tiles=*/12, /*data_rank=*/2);
  EXPECT_EQ(UngroupSharding(group_sharding), HloSharding::Replicate());
}

TEST(HloShardingUtilTest, GroupedAndUngroupedIotaSharding) {
  std::vector<std::vector<int64_t>> device_groups = {{0, 1, 2, 3, 4, 5},
                                                     {6, 7, 8, 9, 10, 11}};
  GroupedSharding group_sharding = GroupedSharding(
      device_groups, /*group_dims=*/{0}, /*group_dim_sizes=*/{2},
      /*data_rank=*/2, HloSharding::IotaTile({1, 2, 3}, {2, 3}, {1, 0}));
  EXPECT_EQ(UngroupSharding(group_sharding),
            HloSharding::IotaTile({2, 2, 3}, {2, 2, 3}, {0, 2, 1}));
}

TEST(HloShardingUtilTest, GroupedAndUngroupedShardingWithUnsortedGroupDims) {
  HloSharding sharding = HloSharding::IotaTile({4, 3, 5, 7});
  GroupedSharding group_sharding =
      GroupShardingOnDims(sharding, {2, 0}, {1, 2});
  EXPECT_EQ(group_sharding.sharding, HloSharding::IotaTile({2, 3, 1, 7}));
  EXPECT_EQ(UngroupSharding(group_sharding), sharding);
}

TEST(HloShardingUtilTest, UngroupShardingWithUnsortedGroupDims) {
  GroupedSharding group_sharding({{0}, {1}, {2}, {3}}, {1, 0}, {2, 2}, 4,
                                 HloSharding::Replicate());
  EXPECT_EQ(UngroupSharding(group_sharding),
            HloSharding::IotaTile({2, 2, 1, 1}, {2, 2}, {1, 0}));
}

TEST(HloShardingUtilTest, DeviceGroupsDoesNotMatch) {
  HloSharding sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  DimensionVector group_dim_sizes = {2};

  std::vector<std::vector<int64_t>> lhs_device_groups = {{0, 2, 4, 6},
                                                         {1, 3, 5, 7}};
  DimensionVector lhs_group_dims = {3};

  auto lhs =
      GroupedSharding(std::move(lhs_device_groups), std::move(lhs_group_dims),
                      group_dim_sizes, 2, sharding,
                      /*subgroup_manual=*/true);

  std::vector<std::vector<int64_t>> rhs_device_groups = {{0, 1, 4, 5},
                                                         {2, 3, 6, 7}};
  DimensionVector rhs_group_dims = {2};

  auto rhs =
      GroupedSharding(std::move(rhs_device_groups), std::move(rhs_group_dims),
                      group_dim_sizes, 2, sharding,
                      /*subgroup_manual=*/true);

  EXPECT_FALSE(DeviceGroupsAreMatch(lhs, rhs));
}

TEST(HloShardingUtilTest, DeviceGroupsMatch) {
  HloSharding lhs_sharding = HloSharding::Replicate();
  DimensionVector group_dims = {2};
  DimensionVector group_dim_sizes = {2};
  std::vector<std::vector<int64_t>> device_groups = {{0, 2}, {1, 3}};

  auto lhs = GroupedSharding(
      device_groups, DimensionVector(group_dims.begin(), group_dims.end()),
      group_dim_sizes, 2, lhs_sharding,
      /*subgroup_manual=*/true);

  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  auto rhs = GroupedSharding(
      device_groups, DimensionVector(group_dims.begin(), group_dims.end()),
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

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingNoShortcut) {
  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4});
  std::vector<int64_t> success = {1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20};
  std::vector<int64_t> fail = {2, 5, 6, 9, 10, 13, 14, 17, 18};
  for (int64_t i : success) {
    Shape shape = ShapeUtil::MakeShape(F32, {i});
    EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
  }
  for (int64_t i : fail) {
    Shape shape = ShapeUtil::MakeShape(F32, {i});
    EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
  }
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut1) {
  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4});
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut2) {
  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  Array<int64_t> lhs_array({4});
  lhs_array.SetValues({1, 0, 2, 3});
  HloSharding lhs_sharding = HloSharding::Tile(lhs_array);
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut3) {
  HloSharding rhs_sharding = HloSharding::PartialTile(
      TileAssignment((absl::Span<const int64_t>){2, 2}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4}, {2, 2}, {1, 0});
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut4) {
  HloSharding rhs_sharding =
      HloSharding::PartialTile(TileAssignment({2, 2}, {2, 2}, {1, 0}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4}, {2, 2}, {1, 0});
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut5) {
  HloSharding rhs_sharding =
      HloSharding::PartialTile(TileAssignment({2, 3, 5, 7}));
  HloSharding lhs_sharding_1 =
      HloSharding::IotaTile({2, 21, 5}, {2, 3, 5, 7}, {0, 1, 3, 2});
  HloSharding lhs_sharding_2 =
      HloSharding::IotaTile({2, 21, 5}, {2, 3, 5, 7}, {0, 2, 3, 1});
  HloSharding lhs_sharding_3 = HloSharding::IotaTile({2, 21, 5});
  std::vector<Shape> shapes = {ShapeUtil::MakeShape(F32, {10, 42, 10}),
                               ShapeUtil::MakeShape(F32, {11, 41, 11})};
  for (const auto& shape : shapes) {
    EXPECT_TRUE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_1, rhs_sharding));
    EXPECT_FALSE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_2, rhs_sharding));
    EXPECT_FALSE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_3, rhs_sharding));
  }
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut6) {
  HloSharding rhs_sharding =
      HloSharding::PartialTile(TileAssignment({2, 3, 5, 7 * 11 * 13}));
  HloSharding lhs_sharding_1 = HloSharding::PartialTile(TileAssignment(
      {2 * 7, 3, 5 * 11, 13}, {2, 3, 5, 7, 11, 13}, {0, 3, 1, 2, 4, 5}));
  HloSharding lhs_sharding_2 = HloSharding::PartialTile(TileAssignment(
      {2 * 7, 3, 5 * 11, 13}, {2, 3, 5, 11, 7, 13}, {0, 4, 1, 2, 3, 5}));
  HloSharding lhs_sharding_3 = HloSharding::PartialTile(TileAssignment(
      {2 * 7, 3, 5 * 11, 13}, {2, 3, 5, 13, 7, 11}, {0, 4, 1, 2, 5, 3}));
  HloSharding lhs_sharding_4 = HloSharding::PartialTile(TileAssignment(
      {2 * 7, 3, 5 * 11, 13}, {2, 3, 5, 7, 13, 11}, {0, 3, 1, 2, 5, 4}));
  HloSharding lhs_sharding_5 =
      HloSharding::PartialTile(TileAssignment({2 * 7, 3, 5 * 11, 13}));
  std::vector<Shape> shapes = {
      ShapeUtil::MakeShape(F32, {2 * 7, 9, 5 * 11}),
      ShapeUtil::MakeShape(F32, {2 * 7 - 1, 4, 5 * 11 - 1})};
  for (const auto& shape : shapes) {
    EXPECT_TRUE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_1, rhs_sharding));
    EXPECT_TRUE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_2, rhs_sharding));
    EXPECT_TRUE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_3, rhs_sharding));
    EXPECT_TRUE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_4, rhs_sharding));
    EXPECT_FALSE(
        IsSubTilingOrEqualSharding(shape, lhs_sharding_5, rhs_sharding));
  }
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut7) {
  HloSharding rhs_sharding =
      HloSharding::PartialTile(TileAssignment({1, 2, 1, 3, 5 * 7 * 11}));
  HloSharding lhs_sharding = HloSharding::PartialTile(
      TileAssignment({5, 2, 7, 3, 11}, {2, 3, 5, 7, 11}, {2, 0, 3, 1, 4}));
  std::vector<Shape> shapes = {ShapeUtil::MakeShape(F32, {5, 2, 7, 3}),
                               ShapeUtil::MakeShape(F32, {2, 2, 9, 3})};
  for (const auto& shape : shapes) {
    EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
  }
}

TEST(HloShardingUtilTest, IsSortOperandShardingMovableRankTwoOneFreeDim) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {8, 128}), 1);
  iota.set_sharding(HloSharding::IotaTile({1, 2}));
  EXPECT_TRUE(IsSortOperandShardingMovable(&iota, 1));
}

TEST(HloShardingUtilTest,
     IsSortOperandShardingMovableRankTwoOneFreeDimOfSize1) {
  HloIotaInstruction iota(ShapeUtil::MakeShape(F32, {1, 128}), 1);
  iota.set_sharding(HloSharding::IotaTile({1, 2}));
  EXPECT_FALSE(IsSortOperandShardingMovable(&iota, 1));
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

TEST(HloShardingUtilTest, TileShape) {
  HloSharding sharding = HloSharding::Tile(TileAssignment({4, 1}));
  Shape shape_0 = ShapeUtil::MakeShape(F32, {80, 128});
  auto tile_shape_0 = hlo_sharding_util::TileShape(sharding, shape_0);
  auto expected_shape_0 = ShapeUtil::MakeShape(F32, {20, 128});
  EXPECT_EQ(tile_shape_0, expected_shape_0);
  Shape shape_1 = ShapeUtil::MakeShape(F32, {40, 128});
  auto tile_shape_1 = hlo_sharding_util::TileShape(sharding, shape_1);
  auto expected_shape_1 = ShapeUtil::MakeShape(F32, {10, 128});
  EXPECT_EQ(tile_shape_1, expected_shape_1);
  const Shape tuple = ShapeUtil::MakeTupleShape({tile_shape_0, tile_shape_1});
  EXPECT_EQ(hlo_sharding_util::TileShape(sharding, tuple),
            ShapeUtil::MakeTupleShape({expected_shape_0, expected_shape_1}));
}

TEST(HloShardingUtilTest, UntileShape) {
  HloSharding sharding = HloSharding::Tile(TileAssignment({4, 1}));
  Shape shape_0 = ShapeUtil::MakeShape(F32, {80, 128});
  auto tile_shape_0 = hlo_sharding_util::UntileShape(sharding, shape_0);
  auto expected_shape_0 = ShapeUtil::MakeShape(F32, {320, 128});
  EXPECT_EQ(tile_shape_0, expected_shape_0);
  Shape shape_1 = ShapeUtil::MakeShape(F32, {40, 128});
  auto tile_shape_1 = hlo_sharding_util::UntileShape(sharding, shape_1);
  auto expected_shape_1 = ShapeUtil::MakeShape(F32, {160, 128});
  EXPECT_EQ(tile_shape_1, expected_shape_1);
  const Shape tuple = ShapeUtil::MakeTupleShape({tile_shape_0, tile_shape_1});
  EXPECT_EQ(hlo_sharding_util::UntileShape(sharding, tuple),
            ShapeUtil::MakeTupleShape({expected_shape_0, expected_shape_1}));
}

using HloShardingUtilTestWithHlo = HloTestBase;

TEST_F(HloShardingUtilTestWithHlo, InferDotOperandShardingTest) {
  absl::string_view hlo_string = R"(
    HloModule module

    ENTRY %main.7 {
      %p0 = bf16[32,64,128,512] parameter(0), sharding={devices=[8,1,1,4]<=[32]}
      %p1 = bf16[32,64,256,512] parameter(1), sharding={devices=[1,1,1,2,16]<=[8,2,2]T(1,0,2) last_tile_dim_replicate}
      ROOT %dot.3 = bf16[32,64,128,256] dot(%p0, %p1), lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={3}, sharding={devices=[2,2,2,2,2]<=[32] last_tile_dim_replicate}
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* dot = module->entry_computation()->root_instruction();
  auto dnums = dot_as_convolution_util::ParseDotGeneralFromDot(dot);

  bool consider_other_operand = true;
  bool may_combine_partial_sharding = false;
  EXPECT_EQ(InferDotOperandSharding(dot, 0, dnums, consider_other_operand,
                                    may_combine_partial_sharding),
            HloSharding::PartialTile(TileAssignment({2, 2, 2, 1, 4})));
  EXPECT_EQ(InferDotOperandSharding(dot, 1, dnums, consider_other_operand,
                                    may_combine_partial_sharding),
            HloSharding::IotaTile({8, 1, 1, 4}));

  consider_other_operand = true;
  may_combine_partial_sharding = true;
  EXPECT_EQ(InferDotOperandSharding(dot, 0, dnums, consider_other_operand,
                                    may_combine_partial_sharding),
            HloSharding::PartialTile(TileAssignment({2, 2, 2, 2, 2})));
  EXPECT_EQ(InferDotOperandSharding(dot, 1, dnums, consider_other_operand,
                                    may_combine_partial_sharding),
            HloSharding::IotaTile({8, 1, 1, 4}));

  consider_other_operand = false;
  for (bool may_combine_partial_sharding : {false, true}) {
    EXPECT_EQ(InferDotOperandSharding(dot, 0, dnums, consider_other_operand,
                                      may_combine_partial_sharding),
              HloSharding::PartialTile(TileAssignment({2, 2, 2, 1, 4})));
    EXPECT_EQ(InferDotOperandSharding(dot, 1, dnums, consider_other_operand,
                                      may_combine_partial_sharding),
              HloSharding::PartialTile(TileAssignment(
                  {2, 2, 2, 1, 4}, {2, 2, 2, 2, 2}, {0, 1, 3, 2, 4})));
  }
}

}  // namespace
}  // namespace hlo_sharding_util
}  // namespace xla
