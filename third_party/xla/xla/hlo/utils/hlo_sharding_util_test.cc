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
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/call_graph.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace hlo_sharding_util {
namespace {

TEST(HloShardingUtilTest, MergeShardingIfCompatible1) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}, {16, 8}, {1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 4, 2, 4}, {4, 4, 8}, {0, 2, 1})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible2) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 2, 4, 16}, {16, 8}, {1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 2, 4, 4}, {4, 4, 8}, {0, 2, 1})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible3) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({4, 2, 1, 16}, {16, 8}, {1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({1, 1, 4, 32}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 2, 4, 4}, {16, 8}, {1, 0})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible4) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}, {16, 8}, {1, 0}));
  HloSharding dst =
      HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}, {4, 32}, {1, 0}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, &dst));
  EXPECT_EQ(dst, HloSharding::PartialTile(
                     TileAssignment({4, 4, 2, 4}, {4, 32}, {1, 0})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible5) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}, {16, 8}, {1, 0}));
  HloSharding dst =
      HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}, {32, 4}, {1, 0}));
  EXPECT_FALSE(MergeShardingIfCompatible(to_merge, &dst));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible6) {
  HloSharding to_merge =
      HloSharding::PartialTile(TileAssignment({1, 4, 2, 16}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({4, 1, 1, 32}));
  EXPECT_FALSE(MergeShardingIfCompatible(to_merge, &dst));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible7) {
  HloSharding to_merge = HloSharding::PartialTile(
      TileAssignment({2, 1, 2, 2}, {2, 2, 2}, {2, 1, 0}));
  HloSharding dst = HloSharding::PartialTile(TileAssignment({1, 2, 1, 4}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, &dst));
  EXPECT_EQ(dst,
            HloSharding::Tile(TileAssignment({2, 2, 2}, {2, 2, 2}, {2, 0, 1})));
}

TEST(HloShardingUtilTest, MergeShardingIfCompatible8) {
  HloSharding to_merge = HloSharding::PartialTile(TileAssignment({2, 1, 4}));
  HloSharding dst =
      HloSharding::PartialTile(TileAssignment({1, 4, 2}, {2, 2, 2}, {2, 1, 0}));
  EXPECT_TRUE(MergeShardingIfCompatible(to_merge, &dst));
  EXPECT_EQ(dst,
            HloSharding::Tile(TileAssignment({2, 4}, {2, 2, 2}, {0, 2, 1})));
}

TEST(HloShardingUtilTest, MoveAndMergeShardingTilesPartialTile) {
  HloSharding sharding =
      HloSharding::PartialTile(TileAssignment({2, 3, 5, 7, 11}));

  EXPECT_EQ(MoveAndMergeShardingTiles(sharding, 1, 3),
            HloSharding::PartialTile(TileAssignment(
                {2, 1, 5, 7 * 3, 11}, {2, 3, 5, 7, 11}, {0, 2, 3, 1, 4})));

  EXPECT_EQ(MoveAndMergeShardingTiles(sharding, 3, 1),
            HloSharding::PartialTile(TileAssignment(
                {2, 3 * 7, 5, 1, 11}, {2, 3, 5, 7, 11}, {0, 1, 3, 2, 4})));

  {
    Mesh mesh({2, 3, 5, 7, 11}, {"a", "b", "c", "d", "e"});
    NamedSharding input =
        test_utils::FromAxisNames(mesh, {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}});
    HloSharding sharding(input);

    EXPECT_EQ(
        MoveAndMergeShardingTiles(sharding, 1, 3).named_sharding(),
        test_utils::FromAxisNames(mesh, {{"a"}, {}, {"c"}, {"d", "b"}, {"e"}}));

    EXPECT_EQ(
        MoveAndMergeShardingTiles(sharding, 3, 1).named_sharding(),
        test_utils::FromAxisNames(mesh, {{"a"}, {"b", "d"}, {"c"}, {}, {"e"}}));
  }
}

TEST(HloShardingUtilTest, MoveAndMergeShardingTilesSubGroup) {
  HloSharding sharding =
      HloSharding::Subgroup(TileAssignment({2, 3, 5, 7, 11}),
                            {OpSharding::MANUAL, OpSharding::REPLICATED});

  EXPECT_EQ(
      MoveAndMergeShardingTiles(sharding, 0, 2),
      HloSharding::Subgroup(TileAssignment({1, 3, 5 * 2, 7, 11},
                                           {2, 3, 5, 7, 11}, {1, 2, 0, 3, 4}),
                            {OpSharding::MANUAL, OpSharding::REPLICATED}));

  EXPECT_EQ(
      MoveAndMergeShardingTiles(sharding, 2, 0),
      HloSharding::Subgroup(TileAssignment({2 * 5, 3, 1, 7, 11},
                                           {2, 3, 5, 7, 11}, {0, 2, 1, 3, 4}),
                            {OpSharding::MANUAL, OpSharding::REPLICATED}));

  {
    Mesh mesh({2, 3, 5, 7, 11}, {"a", "b", "c", "d", "e"});
    NamedSharding input = test_utils::FromAxisNames(
        mesh, {{"a"}, {"b"}, {"c"}}, /*replicated_axes=*/{"e"},
        /*unreduced_axes=*/{}, /*manual_axes=*/{"d"});
    HloSharding sharding(input);

    EXPECT_EQ(MoveAndMergeShardingTiles(sharding, 0, 2).named_sharding(),
              test_utils::FromAxisNames(mesh, {{}, {"b"}, {"c", "a"}},
                                        /*replicated_axes=*/{"e"},
                                        /*unreduced_axes=*/{},
                                        /*manual_axes=*/{"d"}));
    EXPECT_EQ(MoveAndMergeShardingTiles(sharding, 2, 0).named_sharding(),
              test_utils::FromAxisNames(mesh, {{"a", "c"}, {"b"}, {}},
                                        /*replicated_axes=*/{"e"},
                                        /*unreduced_axes=*/{},
                                        /*manual_axes=*/{"d"}));
  }
}

TEST(HloShardingUtilTest, MoveAndMergeShardingTilesNamedSharding) {
  Mesh mesh({4, 2}, {"x", "y"});
  NamedSharding input =
      test_utils::FromAxisNames(mesh, {{"x:(1)2"}, {"y", "x:(2)2"}});
  HloSharding sharding(input);

  EXPECT_EQ(MoveAndMergeShardingTiles(sharding, 1, 0).named_sharding(),
            test_utils::FromAxisNames(mesh, {{"x:(1)2", "y", "x:(2)2"}, {}}));
}

TEST(HloShardingUtilTest, MoveAndMergeShardingTilesNamedShardingSubAxes) {
  Mesh mesh({4}, {"x"});
  NamedSharding input =
      test_utils::FromAxisNames(mesh, {{"x:(1)2"}, {"x:(2)2"}});
  HloSharding sharding(input);

  EXPECT_EQ(MoveAndMergeShardingTiles(sharding, 1, 0).named_sharding(),
            test_utils::FromAxisNames(mesh, {{"x"}, {}}));
}

TEST(HloShardingUtilTest, MoveAndMergeShardingTilesErrorCases) {
  HloSharding sharding = HloSharding::IotaTile({2, 2});

  EXPECT_DEATH(MoveAndMergeShardingTiles(sharding, 0, 0),
               "source_dim != target_dim");
  EXPECT_DEATH(MoveAndMergeShardingTiles(sharding, 2, 0),
               "source_dim < sharding.TiledDataRank()");
  EXPECT_DEATH(MoveAndMergeShardingTiles(sharding, 0, 2),
               "target_dim < sharding.TiledDataRank()");
  EXPECT_DEATH(MoveAndMergeShardingTiles(HloSharding::Replicate(), 0, 1),
               "sharding.IsTiled()");

  Mesh mesh({2, 2}, {"x", "y"});
  NamedSharding input = test_utils::FromAxisNames(mesh, {{"x"}, {"y"}});
  HloSharding named_sharding(input);

  EXPECT_DEATH(MoveAndMergeShardingTiles(named_sharding, 0, 0),
               "source_dim != target_dim");
  EXPECT_DEATH(MoveAndMergeShardingTiles(named_sharding, 2, 0),
               "source_dim < sharding.TiledDataRank()");
  EXPECT_DEATH(MoveAndMergeShardingTiles(named_sharding, 0, 2),
               "target_dim < sharding.TiledDataRank()");
}

TEST(HloShardingUtilTest, MergeShardingDimension) {
  EXPECT_EQ(MergeShardingDimension(HloSharding::IotaTile({2, 2}), 0),
            HloSharding::IotaTile({4}));

  {
    Mesh mesh({2, 2}, {"x", "y"});

    HloSharding result = MergeShardingDimension(
        HloSharding(test_utils::FromAxisNames(mesh, {{"x"}, {"y"}})), 0);

    EXPECT_EQ(result.named_sharding(),
              test_utils::FromAxisNames(mesh, {{"x", "y"}}));
  }
}

TEST(HloShardingUtilTest, MergeShardingDimensionMultiAxis) {
  EXPECT_EQ(MergeShardingDimension(HloSharding::IotaTile({2, 2, 2}), 1),
            HloSharding::IotaTile({2, 4}));

  {
    Mesh mesh({2, 2, 2}, {"x", "y", "z"});

    HloSharding result = MergeShardingDimension(
        HloSharding(test_utils::FromAxisNames(mesh, {{"x"}, {"y"}, {"z"}})), 1);

    EXPECT_EQ(result.named_sharding(),
              test_utils::FromAxisNames(mesh, {{"x"}, {"y", "z"}}));
  }
}

TEST(HloShardingUtilTest, MergeShardingDimensionWithEmpty) {
  EXPECT_EQ(MergeShardingDimension(HloSharding::IotaTile({2, 1}), 0),
            HloSharding::IotaTile({2}));
  EXPECT_EQ(MergeShardingDimension(HloSharding::IotaTile({1, 2}), 0),
            HloSharding::IotaTile({2}));

  {
    Mesh mesh({2}, {"x"});

    HloSharding result = MergeShardingDimension(
        HloSharding(test_utils::FromAxisNames(mesh, {{"x"}, {}})), 0);

    EXPECT_EQ(result.named_sharding(),
              test_utils::FromAxisNames(mesh, {{"x"}}));
  }
}

TEST(HloShardingUtilTest, MergeShardingDimensionWithSubAxesKeepSubAxis) {
  // 'x':(1)2 + 'x':(2)2 = 'x'(1)4.
  Mesh mesh({8}, {"x"});
  NamedSharding input =
      test_utils::FromAxisNames(mesh, {{"x:(1)2"}, {"x:(2)2"}});

  HloSharding sharding(input);
  HloSharding merged = MergeShardingDimension(sharding, 0);

  NamedSharding expected_ns = test_utils::FromAxisNames(mesh, {{"x:(1)4"}});
  EXPECT_EQ(merged.named_sharding(), expected_ns);
}

TEST(HloShardingUtilTest, MergeShardingDimensionWithSubAxesBecomeFullAxis) {
  // 'x':(1)2 + 'x':(2)2 = 'x'.
  Mesh mesh({4}, {"x"});
  NamedSharding input =
      test_utils::FromAxisNames(mesh, {{"x:(1)2"}, {"x:(2)2"}});

  HloSharding sharding(input);
  HloSharding merged = MergeShardingDimension(sharding, 0);

  NamedSharding expected_ns = test_utils::FromAxisNames(mesh, {{"x"}});
  EXPECT_EQ(merged.named_sharding(), expected_ns);
}

TEST(HloShardingUtilTest, MergeShardingDimensionWithSubAxesSuccess) {
  // {'x':(1)2} + {y, 'x':(2)2} = {'x':(1)2, y, 'x':(2)2}
  Mesh mesh({4, 2}, {"x", "y"});
  NamedSharding input =
      test_utils::FromAxisNames(mesh, {{"x:(1)2"}, {"y", "x:(2)2"}});

  HloSharding sharding(input);
  HloSharding merged = MergeShardingDimension(sharding, 0);

  EXPECT_EQ(merged.named_sharding(),
            test_utils::FromAxisNames(mesh, {{"x:(1)2", "y", "x:(2)2"}}));
}

TEST(HloShardingUtilTest, MergeShardingDimensionMergesSubAxesAtBoundary) {
  // ["x":(4)2, "y":(1)4, "x":(1)2] + ["x":(2)2, "y":(4)4] =
  // ["x":(4)2, "y":(1)4, "x":(1)4, "y":(4)4]
  Mesh mesh({16, 16}, {"x", "y"});
  NamedSharding input = test_utils::FromAxisNames(
      mesh, {{"x:(4)2", "y:(1)4", "x:(1)2"}, {"x:(2)2", "y:(4)4"}});

  HloSharding sharding(input);
  HloSharding merged = MergeShardingDimension(sharding, 0);

  EXPECT_EQ(merged.named_sharding(),
            test_utils::FromAxisNames(
                mesh, {{"x:(4)2", "y:(1)4", "x:(1)4", "y:(4)4"}}));
}

TEST(HloShardingUtilTest, SplitShardingDimension) {
  EXPECT_EQ(SplitShardingDimension(HloSharding::IotaTile({8}), 0, 2),
            HloSharding::IotaTile({2, 4}));

  {
    Mesh mesh({2, 4}, {"x", "y"});

    HloSharding result = SplitShardingDimension(
        HloSharding(test_utils::FromAxisNames(mesh, {{"x", "y"}})), 0, 2);

    EXPECT_EQ(result.named_sharding(),
              test_utils::FromAxisNames(mesh, {{"x"}, {"y"}}));
  }

  {
    Mesh mesh({2, 2, 2}, {"x", "y", "z"});

    HloSharding result = SplitShardingDimension(
        HloSharding(test_utils::FromAxisNames(mesh, {{"x", "y", "z"}})), 0, 2);

    EXPECT_EQ(result.named_sharding(),
              test_utils::FromAxisNames(mesh, {{"x"}, {"y", "z"}}));
  }
}

TEST(HloShardingUtilTest, SplitShardingDimensionIntoSubAxes) {
  EXPECT_EQ(SplitShardingDimension(HloSharding::IotaTile({8}), 0, 4),
            HloSharding::IotaTile({4, 2}));

  {
    Mesh mesh({8}, {"x"});

    HloSharding result = SplitShardingDimension(
        HloSharding(test_utils::FromAxisNames(mesh, {{"x"}})), 0, 4);

    EXPECT_EQ(result.named_sharding(),
              test_utils::FromAxisNames(mesh, {{"x:(1)4"}, {"x:(4)2"}}));
  }
}

TEST(HloShardingUtilTest, SplitShardingDimensionNamedShardingConstraints) {
  // Tiled sharding can split, Named sharding cannot (because 3 is not
  // compatible with x=2).
  EXPECT_EQ(SplitShardingDimension(HloSharding::IotaTile({6}), 0, 3),
            HloSharding::IotaTile({3, 2}));
  {
    Mesh mesh({2, 3}, {"x", "y"});

    // 3 is not a valid split for x=2, y=3.
    EXPECT_DEATH(
        SplitShardingDimension(
            HloSharding(test_utils::FromAxisNames(mesh, {{"x", "y"}})), 0, 3),
        "Could not slice dimension 0 with size 3");
  }
}

TEST(HloShardingUtilTest, TransposeShardingReplicated) {
  EXPECT_EQ(TransposeSharding(HloSharding::Replicate(), {0, 1, 2}),
            HloSharding::Replicate());

  EXPECT_EQ(
      TransposeSharding(HloSharding::Replicate({}, /*use_named_sharding=*/true),
                        {0, 1, 2}),
      HloSharding::Replicate({}, /*use_named_sharding=*/true));
}

TEST(HloShardingUtilTest, TransposeShardingTiled) {
  HloSharding input = HloSharding::IotaTile({1, 2, 1, 2});
  HloSharding output = HloSharding::IotaTile({2, 1, 2, 1}, {2, 2}, {1, 0});
  EXPECT_EQ(TransposeSharding(input, {3, 0, 1, 2}), output);

  {
    Mesh mesh({2, 2}, {"a", "b"});
    NamedSharding input =
        test_utils::FromAxisNames(mesh, {{}, {"a"}, {}, {"b"}});
    NamedSharding output =
        test_utils::FromAxisNames(mesh, {{"b"}, {}, {"a"}, {}});
    EXPECT_EQ(
        TransposeSharding(HloSharding(input), {3, 2, 1, 0}).named_sharding(),
        output);
  }
}

TEST(HloShardingUtilTest, TransposeShardingWithCollapsedDimsSubgroupManual) {
  HloSharding input =
      HloSharding::Subgroup(TileAssignment({1, 2, 4}), {OpSharding::MANUAL});
  HloSharding output =
      HloSharding::Subgroup(TileAssignment({1, 1, 2, 4}), {OpSharding::MANUAL});
  EXPECT_EQ(TransposeShardingWithCollapsedDims(input, {-1, 2}, {-1, -1, 1}),
            output);

  {
    Mesh mesh({1, 2, 4}, {"a", "b", "c"});
    NamedSharding input = test_utils::FromAxisNames(mesh, {{"a"}, {"b"}});
    NamedSharding output = test_utils::FromAxisNames(mesh, {{}, {}, {"b"}});
    EXPECT_EQ(TransposeShardingWithCollapsedDims(HloSharding(input), {-1, 2},
                                                 {-1, -1, 1})
                  ->named_sharding(),
              output);
  }
}

TEST(HloShardingUtilTest, ReshapeShardingDimensionSizeOnePartitioned1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 2, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 16});
  HloSharding input_sharding = HloSharding::IotaTile({3, 2, 2});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 2, 3}, {3, 2, 2}, {1, 2, 0}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingDimensionSizeOnePartitioned2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 1, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 16});
  HloSharding input_sharding = HloSharding::IotaTile({2, 3, 2});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 2, 3}, {2, 3, 2}, {0, 2, 1}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingDimensionSizeOnePartitioned3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 1, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {32});
  HloSharding input_sharding = HloSharding::IotaTile({2, 3, 2});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({4, 3}, {2, 3, 2}, {0, 2, 1}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingDimensionSizeOnePartitioned4) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 32});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 16});
  HloSharding input_sharding = HloSharding::IotaTile({3, 4});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 2, 3}, {3, 4}, {1, 0}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingDimensionSizeOnePartitioned5) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 32});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 1, 2, 16});
  HloSharding input_sharding = HloSharding::IotaTile({2, 3, 4});
  HloSharding output_sharding = HloSharding::IotaTile({2, 3, 2, 2});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingMaximal) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 2});
  HloSharding sharding = HloSharding::AssignDevice(7);
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_EQ(result, sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledInvalid) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 2});
  HloSharding sharding = HloSharding::IotaTile({1, 2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  ASSERT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingTiledMerge) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 5, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {20, 7});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 4, 7});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 4, 7});
  HloSharding input_sharding = HloSharding::IotaTile({16, 1});
  HloSharding output_sharding = HloSharding::IotaTile({4, 4, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplit3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {36});
  Shape output_shape = ShapeUtil::MakeShape(F32, {6, 6});
  HloSharding input_sharding = HloSharding::IotaTile({4});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({2, 1, 2}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledSplitThenMerge) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 4, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 16, 7});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledArbitraryMinorDimensions) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {16, 7, 5, 3});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 15, 2, 14});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_EQ(result, sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTiledTrivialDimensions) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {3, 1, 5, 7});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 5, 1, 7});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 2, 1});
  HloSharding output_sharding = HloSharding::IotaTile({1, 2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTrivialDimensionInsertedToEnd) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {8, 16});
  Shape output_shape = ShapeUtil::MakeShape(F32, {8, 16, 1});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, NoopReshapeShardingEmptyTile) {
  Shape shape = ShapeUtil::MakeShape(F32, {7, 1, 1});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result = ReshapeSharding(shape, shape, sharding);
  EXPECT_EQ(result, sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingScalar) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  ASSERT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingSuffixShapeSizeOne1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {64, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {64, 1});
  HloSharding input_sharding = HloSharding::IotaTile({4, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({4, 1});

  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);

  result = ReshapeSharding(output_shape, input_shape, output_sharding);
  EXPECT_EQ(result, input_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingSuffixShapeSizeOne2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {64, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {64, 1});
  HloSharding input_sharding = HloSharding::IotaTile({4, 2, 8});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({4, 2, 8}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingSuffixShapeSizeOne3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {64, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {64, 1, 1});
  HloSharding input_sharding = HloSharding::IotaTile({4, 2});
  HloSharding output_sharding = HloSharding::IotaTile({4, 2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingSuffixShapeSizeOne4) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {4, 2, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 2});
  HloSharding input_sharding = HloSharding::IotaTile({4, 2, 4});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({4, 2, 4}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingPrefixShapeSizeOne1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 64});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 64});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 4});
  HloSharding output_sharding = HloSharding::IotaTile({1, 4});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);

  result = ReshapeSharding(output_shape, input_shape, output_sharding);
  EXPECT_EQ(result, input_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingPrefixShapeSizeOne2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 1, 64});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 64});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);

  result = ReshapeSharding(output_shape, input_shape, output_sharding);
  EXPECT_EQ(result, input_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {6, 2, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4, 3, 5});
  HloSharding sharding = HloSharding::IotaTile({2, 1, 5});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, sharding);
  EXPECT_EQ(result, sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5, 7, 11});
  Shape output_shape = ShapeUtil::MakeShape(F32, {10, 21, 11});
  HloSharding input_sharding = HloSharding::IotaTile({2, 1, 1, 1, 13});
  HloSharding output_sharding = HloSharding::IotaTile({2, 1, 13});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  ASSERT_TRUE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose3) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 10});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 5});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  ASSERT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingTranspose4) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3, 5, 7, 11, 13, 17, 19});
  Shape output_shape = ShapeUtil::MakeShape(F32, {3, 2, 55, 91, 19, 17});
  HloSharding input_sharding = HloSharding::IotaTile({1, 1, 5, 1, 1, 13, 1, 1});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({1, 1, 5, 1, 1, 1, 13}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
}

TEST(HloShardingUtilTest, ReshapeShardingWithPadding1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {4});
  Shape output_shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloSharding input_sharding = HloSharding::IotaTile({8});
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  ASSERT_FALSE(result.has_value());
}

TEST(HloShardingUtilTest, ReshapeShardingWithPadding2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape output_shape = ShapeUtil::MakeShape(F32, {4});
  HloSharding input_sharding = HloSharding::IotaTile({2, 4});
  HloSharding output_sharding =
      HloSharding::PartialTile(TileAssignment({4, 2}));
  std::optional<HloSharding> result =
      ReshapeSharding(input_shape, output_shape, input_sharding);
  EXPECT_EQ(result, output_sharding);
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

TEST(HloShardingUtilTest, PropagateShardingAlongDimsAndReplicateOthers1) {
  HloSharding source_sharding = HloSharding::IotaTile({2, 3, 5, 7, 11});
  std::vector<int64_t> source_dims = {2, 4, 1};
  std::vector<int64_t> target_dims = {2, 1, 3};
  int64_t target_shape_rank = 5;
  HloSharding target_sharding = PropagateShardingAlongDimsAndReplicateOthers(
      source_sharding, source_dims, target_dims, target_shape_rank);
  HloSharding expected = HloSharding::PartialTile(
      TileAssignment({1, 11, 5, 3, 1, 14}, {2, 3, 5, 7, 11}, {4, 2, 1, 0, 3}));
  EXPECT_EQ(target_sharding, expected);

  {
    Mesh mesh({2, 3, 5, 7, 11}, {"a", "b", "c", "d", "e"});
    NamedSharding source_sharding =
        test_utils::FromAxisNames(mesh, {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}});
    HloSharding target_sharding = PropagateShardingAlongDimsAndReplicateOthers(
        HloSharding(source_sharding), source_dims, target_dims,
        target_shape_rank);
    NamedSharding expected =
        test_utils::FromAxisNames(mesh, {{}, {"e"}, {"c"}, {"b"}, {}});
    EXPECT_EQ(target_sharding.named_sharding(), expected);
  }
}

TEST(HloShardingUtilTest, PropagateShardingAlongDimsAndReplicateOthers2) {
  HloSharding source_sharding = HloSharding::IotaTile({2, 3, 5, 7, 11});
  std::vector<int64_t> source_dims = {0, 2, 4};
  std::vector<int64_t> target_dims = {0, 1, 2};
  int64_t target_shape_rank = 3;
  HloSharding target_sharding = PropagateShardingAlongDimsAndReplicateOthers(
      source_sharding, source_dims, target_dims, target_shape_rank);
  HloSharding expected = HloSharding::PartialTile(
      TileAssignment({2, 5, 11, 21}, {2, 3, 5, 7, 11}, {0, 2, 4, 1, 3}));
  EXPECT_EQ(target_sharding, expected);

  {
    Mesh mesh({2, 3, 5, 7, 11}, {"a", "b", "c", "d", "e"});
    NamedSharding source_sharding =
        test_utils::FromAxisNames(mesh, {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}});
    HloSharding target_sharding = PropagateShardingAlongDimsAndReplicateOthers(
        HloSharding(source_sharding), source_dims, target_dims,
        target_shape_rank);
    NamedSharding expected =
        test_utils::FromAxisNames(mesh, {{"a"}, {"c"}, {"e"}});
    EXPECT_EQ(target_sharding.named_sharding(), expected);
  }
}

TEST(HloShardingUtilTest, PropagateShardingAlongDimsAndReplicateOthers3) {
  HloSharding source_sharding = HloSharding::IotaTile({2, 3, 5, 7, 11});
  std::vector<int64_t> source_dims = {4, 3, 1};
  std::vector<int64_t> target_dims = {0, 1, 3};
  int64_t target_shape_rank = 4;
  HloSharding target_sharding = PropagateShardingAlongDimsAndReplicateOthers(
      source_sharding, source_dims, target_dims, target_shape_rank);
  HloSharding expected = HloSharding::PartialTile(
      TileAssignment({11, 7, 1, 3, 10}, {2, 3, 5, 7, 11}, {4, 3, 1, 0, 2}));
  EXPECT_EQ(target_sharding, expected);

  {
    Mesh mesh({2, 3, 5, 7, 11}, {"a", "b", "c", "d", "e"});
    NamedSharding source_sharding =
        test_utils::FromAxisNames(mesh, {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}});
    HloSharding target_sharding = PropagateShardingAlongDimsAndReplicateOthers(
        HloSharding(source_sharding), source_dims, target_dims,
        target_shape_rank);
    NamedSharding expected =
        test_utils::FromAxisNames(mesh, {{"e"}, {"d"}, {}, {"b"}});
    EXPECT_EQ(target_sharding.named_sharding(), expected);
  }
}

TEST(HloShardingUtilTest, PropagateShardingAlongDimsAndReplicateOthers4) {
  Mesh mesh({2, 3, 5, 7, 11, 13}, {"a", "b", "c", "d", "e", "f"});
  NamedSharding source_sharding = test_utils::FromAxisNames(
      mesh, {{"a"}, {"c", "b"}, {}, {"d"}, {}}, {},
      /*unreduced_axes=*/{"e"}, /*manual_axes=*/{"f"});
  std::vector<int64_t> source_dims = {2, 1, 3};
  std::vector<int64_t> target_dims = {0, 3, 1};
  int64_t target_shape_rank = 4;
  HloSharding target_sharding = PropagateShardingAlongDimsAndReplicateOthers(
      HloSharding(source_sharding), source_dims, target_dims,
      target_shape_rank);
  NamedSharding expected = test_utils::FromAxisNames(
      mesh, {{}, {"d"}, {}, {"c", "b"}}, {},
      /*unreduced_axes=*/{"e"}, /*manual_axes=*/{"f"});
  EXPECT_EQ(target_sharding.named_sharding(), expected);
}

TEST(HloShardingUtilTest, PartiallyReplicateTiledShardingOnDims) {
  Mesh mesh({2, 3, 5, 7, 11}, {"a", "b", "c", "d", "e"});
  NamedSharding source_sharding =
      test_utils::FromAxisNames(mesh, {{"a"}, {"b"}, {"c"}, {"d"}, {"e"}});
  std::vector<int64_t> dims_to_replicate = {3, 1};
  HloSharding target_sharding = PartiallyReplicateTiledShardingOnDims(
      HloSharding(source_sharding), dims_to_replicate);
  NamedSharding expected =
      test_utils::FromAxisNames(mesh, {{"a"}, {}, {"c"}, {}, {"e"}});
  EXPECT_EQ(target_sharding.named_sharding(), expected);
}

TEST(HloShardingUtilTest, ReplicateAllDataDims) {
  Mesh mesh({2, 3, 5, 7, 11, 13}, {"a", "b", "c", "d", "e", "f"});
  NamedSharding source_sharding = test_utils::FromAxisNames(
      mesh, {{"a"}, {}, {"c"}, {}, {"e"}}, /*replicated_axes=*/{"d"},
      /*unreduced_axes=*/{"b"}, /*manual_axes=*/{"f"});
  HloSharding target_sharding =
      ReplicateAllDataDims(HloSharding(source_sharding), 3);
  NamedSharding expected =
      test_utils::FromAxisNames(mesh, {{}, {}, {}}, {"d"}, {"b"},
                                /*manual_axes=*/{"f"});
  EXPECT_EQ(target_sharding.named_sharding(), expected);
}

TEST(HloShardingUtilTest, RemoveShapeDimensions) {
  Mesh mesh({2, 3, 5, 7, 11}, {"a", "b", "c", "d", "e"});
  NamedSharding source_sharding =
      test_utils::FromAxisNames(mesh, {{"a"}, {}, {"c"}, {}, {"e"}});
  std::vector<int64_t> dims_to_remove = {1, 3};
  HloSharding target_sharding =
      RemoveShapeDimensions(HloSharding(source_sharding), dims_to_remove);
  NamedSharding expected =
      test_utils::FromAxisNames(mesh, {{"a"}, {"c"}, {"e"}});
  EXPECT_EQ(target_sharding.named_sharding(), expected);
}

TEST(HloShardingUtilTest, MergeManualSubgroupSharding) {
  TileAssignment tile_assignment({16, 4});
  std::vector<OpSharding::Type> subgroup_types = {OpSharding::MANUAL,
                                                  OpSharding::REPLICATED};
  // Subgroup sharding
  //  {devices=[16,4]<=[64] last_tile_dims={manual, replicated}}
  HloSharding dst = HloSharding::Subgroup(tile_assignment, subgroup_types);
  HloSharding to_merge = dst;
  EXPECT_FALSE(MergeShardingIfCompatible(to_merge, &dst));
}

TEST(HloShardingUtilTest, GetManualSubgroupSharding_ManualOnly) {
  TileAssignment tile_assignment({1, 2, 2});
  std::vector<OpSharding::Type> subgroup_types = {OpSharding::MANUAL};
  // Subgroup sharding {devices=[1,2,2]0,1,2,3 last_tile_dims={manual}}
  HloSharding sharding = HloSharding::Subgroup(tile_assignment, subgroup_types);

  GroupedSharding group_sharding = GetManualSubgroupSharding(sharding);

  // Expect group_sharding.sharding to be {devices=[1,2]0,1}
  EXPECT_EQ(group_sharding.sharding.tile_assignment(), TileAssignment({1, 2}));

  // Expect the device groups are: {0, 2} and {1, 3}
  EXPECT_EQ(group_sharding.device_groups.ToString(),
            "devices=[2,2]<=[2,2]T(1,0)");
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
  EXPECT_EQ(group_sharding.device_groups.ToString(),
            "devices=[2,4]<=[4,2]T(1,0)");
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
  EXPECT_EQ(group_sharding.device_groups.ToString(),
            "devices=[2,4]<=[2,2,2]T(1,0,2)");
}

TEST(HloShardingUtilTest, UngroupSharding_ManualOnly) {
  HloSharding sharding = HloSharding::IotaTile({1, 2});
  DeviceGroupTileAssignment device_groups(2, 2, {2, 2}, {1, 0});
  DimensionVector group_dims = {2};
  DimensionVector group_dim_sizes = {2};

  auto grouped =
      GroupedSharding(std::move(device_groups), std::move(group_dims),
                      std::move(group_dim_sizes), sharding.num_dimensions(),
                      sharding, /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,2,2]<=[4] last_tile_dims={manual}}");
}

TEST(HloShardingUtilTest, UngroupSharding_ReplicatedAndManual) {
  HloSharding sharding = HloSharding::PartialTile(TileAssignment({1, 2, 2}));
  DeviceGroupTileAssignment device_groups(2, 4, {2, 2, 2}, {2, 0, 1});
  DimensionVector group_dims = {3};
  DimensionVector group_dim_sizes = {2};

  auto grouped = GroupedSharding(
      std::move(device_groups), std::move(group_dims),
      std::move(group_dim_sizes), sharding.num_dimensions() - 1, sharding,
      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,2,2,2]<=[2,2,2]T(0,2,1) last_tile_dims={manual, "
            "replicated}}");
}

TEST(HloShardingUtilTest, UngroupSharding_ManualAndReplicated) {
  HloSharding sharding = HloSharding::PartialTile(TileAssignment({1, 2, 2}));
  DeviceGroupTileAssignment device_groups(2, 4, {2, 2, 2}, {1, 0, 2});
  DimensionVector group_dims = {2};
  DimensionVector group_dim_sizes = {2};

  auto grouped = GroupedSharding(
      std::move(device_groups), std::move(group_dims),
      std::move(group_dim_sizes), sharding.num_dimensions() - 1, sharding,
      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,2,2,2]<=[8] last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, UngroupSharding_Replicated) {
  HloSharding sharding = HloSharding::Replicate();

  DimensionVector group_dims = {3};
  DimensionVector group_dim_sizes = {2};

  DeviceGroupTileAssignment device_groups(2, 2);
  auto grouped =
      GroupedSharding(std::move(device_groups), std::move(group_dims),
                      std::move(group_dim_sizes), 2, sharding,
                      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(ungroup_sharding.ToString(),
            "{devices=[1,1,2,2]<=[4] last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, UngroupSharding_Replicated2) {
  HloSharding sharding = HloSharding::Replicate();
  DimensionVector group_dims = {2};
  DimensionVector group_dim_sizes = {2};

  DeviceGroupTileAssignment device_groups(2, 2, {2, 2}, {1, 0});

  auto grouped =
      GroupedSharding(std::move(device_groups), std::move(group_dims),
                      std::move(group_dim_sizes), 2, sharding,
                      /*subgroup_manual=*/true);

  HloSharding ungroup_sharding = UngroupSharding(grouped);
  VLOG(1) << "ungroup_sharding: " << ungroup_sharding.ToString();

  EXPECT_EQ(
      ungroup_sharding.ToString(),
      "{devices=[1,1,2,2]<=[2,2]T(1,0) last_tile_dims={manual, replicated}}");
}

TEST(HloShardingUtilTest, GroupedAndUngroupedReplicatedSharding) {
  GroupedSharding group_sharding = GetGroupedReplicatedSharding(
      /*num_groups=*/3, /*num_tiles=*/12, /*data_rank=*/2);
  EXPECT_EQ(UngroupSharding(group_sharding), HloSharding::Replicate());
}

TEST(HloShardingUtilTest, GroupedAndUngroupedIotaSharding) {
  DeviceGroupTileAssignment device_groups(2, 6);
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
  DeviceGroupTileAssignment device_groups(4, 1);
  GroupedSharding group_sharding(device_groups, {1, 0}, {2, 2}, 4,
                                 HloSharding::Replicate());
  EXPECT_EQ(UngroupSharding(group_sharding),
            HloSharding::IotaTile({2, 2, 1, 1}, {2, 2}, {1, 0}));
}

TEST(HloShardingUtilTest, DeviceGroupsDoesNotMatch) {
  HloSharding sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
  DimensionVector group_dim_sizes = {2};

  DeviceGroupTileAssignment lhs_device_groups(2, 4, {2, 2, 2}, {2, 0, 1});
  DimensionVector lhs_group_dims = {3};

  auto lhs =
      GroupedSharding(std::move(lhs_device_groups), std::move(lhs_group_dims),
                      group_dim_sizes, 2, sharding,
                      /*subgroup_manual=*/true);

  DeviceGroupTileAssignment rhs_device_groups(2, 4, {2, 2, 2}, {1, 0, 2});
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
  DeviceGroupTileAssignment device_groups(2, 2, {2, 2}, {1, 0});

  auto lhs = GroupedSharding(
      device_groups, DimensionVector(group_dims.begin(), group_dims.end()),
      group_dim_sizes, 2, lhs_sharding,
      /*subgroup_manual=*/true);

  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
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
  HloSharding lhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingReplicatedTiledPartial) {
  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
  HloSharding lhs_sharding = HloSharding::Replicate();
  Shape shape = ShapeUtil::MakeShape(F32, {129, 253});
  EXPECT_FALSE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubShardingPartialTiledTiled) {
  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
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
  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
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
  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
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
  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
  HloSharding lhs_sharding = HloSharding::IotaTile({4});
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut2) {
  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
  Array<int64_t> lhs_array({4});
  lhs_array.SetValues({1, 0, 2, 3});
  HloSharding lhs_sharding = HloSharding::Tile(lhs_array);
  Shape shape = ShapeUtil::MakeShape(F32, {8});
  EXPECT_TRUE(IsSubTilingOrEqualSharding(shape, lhs_sharding, rhs_sharding));
}

TEST(HloShardingUtilTest, IsSubTilingOrEqualShardingShortcut3) {
  HloSharding rhs_sharding = HloSharding::PartialTile(TileAssignment({2, 2}));
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

TEST(HloShardingUtilTest, GetFirstTargetDimToMoveShardingTiles1) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 8, 128, 128});
  HloSharding sharding = HloSharding::IotaTile({8, 1, 2, 16});
  EXPECT_FALSE(
      GetFirstTargetDimToMoveShardingTiles(shape, sharding, 0).has_value());
  EXPECT_FALSE(
      GetFirstTargetDimToMoveShardingTiles(shape, sharding, 1).has_value());
  EXPECT_EQ(GetFirstTargetDimToMoveShardingTiles(shape, sharding, 2), 1);
  EXPECT_EQ(GetFirstTargetDimToMoveShardingTiles(shape, sharding, 3), 2);
}

TEST(HloShardingUtilTest, GetFirstTargetDimToMoveShardingTiles2) {
  Shape shape = ShapeUtil::MakeShape(F32, {4, 8, 128, 128});
  HloSharding sharding = HloSharding::IotaTile({2, 2, 4, 16});
  EXPECT_EQ(GetFirstTargetDimToMoveShardingTiles(shape, sharding, 0), 1);
  EXPECT_EQ(GetFirstTargetDimToMoveShardingTiles(shape, sharding, 1), 0);
  EXPECT_EQ(GetFirstTargetDimToMoveShardingTiles(shape, sharding, 2), 1);
  EXPECT_EQ(GetFirstTargetDimToMoveShardingTiles(shape, sharding, 3), 2);
}

TEST(HloShardingUtilTest, GetFirstTargetDimToMoveShardingTiles3) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 128});
  HloSharding sharding = HloSharding::IotaTile({1, 2});
  EXPECT_FALSE(
      GetFirstTargetDimToMoveShardingTiles(shape, sharding, 0).has_value());
  EXPECT_FALSE(
      GetFirstTargetDimToMoveShardingTiles(shape, sharding, 1).has_value());
}

TEST(HloShardingUtilTest, GetFirstTargetDimToMoveShardingTilesRankOne) {
  Shape shape = ShapeUtil::MakeShape(F32, {1024});
  HloSharding sharding =
      HloSharding::Tile(TileAssignment(std::initializer_list<int64_t>{2}));
  EXPECT_FALSE(
      GetFirstTargetDimToMoveShardingTiles(shape, sharding, 0).has_value());
}

TEST(HloShardingUtilTest, GetFirstTargetDimToMoveShardingTilesReplicated) {
  Shape shape = ShapeUtil::MakeShape(F32, {8, 128});
  HloSharding sharding = HloSharding::Replicate();
  EXPECT_FALSE(
      GetFirstTargetDimToMoveShardingTiles(shape, sharding, 0).has_value());
  EXPECT_FALSE(
      GetFirstTargetDimToMoveShardingTiles(shape, sharding, 1).has_value());
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

TEST(HloShardingUtilTest, TileShapeWithUnreducedSharding) {
  HloSharding sharding = HloSharding::Unreduced();
  Shape shape = ShapeUtil::MakeShape(F32, {6, 6});
  EXPECT_EQ(hlo_sharding_util::TileShape(sharding, shape), shape);
}

TEST(HloShardingUtilTest, TileShapeWithMixedUnreducedSubgroupSharding) {
  Shape shape = ShapeUtil::MakeShape(F32, {6, 6});
  Mesh mesh({2, 2}, {"a", "b"});

  NamedSharding ns_1 = test_utils::FromAxisNames(mesh, {{}, {}}, {},
                                                 /*unreduced_axes=*/{"b"});
  HloSharding sharding_1 = HloSharding::V3ToV2Sharding(ns_1);
  EXPECT_EQ(sharding_1, HloSharding::Subgroup(
                            TileAssignment({2, 2}, {2, 2}, {1, 0}),
                            {OpSharding::UNREDUCED, OpSharding::REPLICATED}));
  EXPECT_EQ(hlo_sharding_util::TileShape(sharding_1, shape),
            ShapeUtil::MakeShape(F32, {6, 6}));

  NamedSharding ns_2 = test_utils::FromAxisNames(mesh, {{"a"}, {}}, {},
                                                 /*unreduced_axes=*/{"b"});
  HloSharding sharding_2 = HloSharding::V3ToV2Sharding(ns_2);
  EXPECT_EQ(sharding_2, HloSharding::Subgroup(TileAssignment({2, 1, 2}),
                                              {OpSharding::UNREDUCED}));
  EXPECT_EQ(hlo_sharding_util::TileShape(sharding_2, shape),
            ShapeUtil::MakeShape(F32, {3, 6}));
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

using HloShardingUtilTestWithHlo = HloHardwareIndependentTestBase;

TEST_F(HloShardingUtilTestWithHlo, InferDotOperandShardingTest1) {
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

TEST_F(HloShardingUtilTestWithHlo, InferDotOperandShardingTest2) {
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

  const HloSharding& lhs_sharding = dot->operand(0)->sharding();
  const HloSharding& rhs_sharding = dot->operand(1)->sharding();
  const HloSharding& dot_sharding = dot->sharding();

  bool may_combine_partial_sharding = true;
  for (int64_t i = 0; i < 2; ++i) {
    EXPECT_EQ(InferDotOperandSharding(nullptr, nullptr, i, dnums, true,
                                      may_combine_partial_sharding),
              HloSharding::Replicate());
  }

  // If the other_operand_sharding is missing (nullptr), we only infer the
  // result from the result.
  for (int64_t i = 0; i < 2; ++i) {
    EXPECT_EQ(InferDotOperandSharding(&dot_sharding, nullptr, i, dnums, true,
                                      may_combine_partial_sharding),
              InferDotOperandSharding(dot, i, dnums, false,
                                      may_combine_partial_sharding));
  }

  EXPECT_EQ(InferDotOperandSharding(nullptr, &rhs_sharding, 0, dnums, true,
                                    may_combine_partial_sharding),
            rhs_sharding);
  EXPECT_EQ(InferDotOperandSharding(nullptr, &lhs_sharding, 1, dnums, true,
                                    may_combine_partial_sharding),
            lhs_sharding);

  EXPECT_EQ(InferDotOperandSharding(nullptr, &rhs_sharding, 0, dnums, false,
                                    may_combine_partial_sharding),
            HloSharding::Replicate());
  EXPECT_EQ(InferDotOperandSharding(nullptr, &lhs_sharding, 1, dnums, false,
                                    may_combine_partial_sharding),
            HloSharding::Replicate());
}

TEST_F(HloShardingUtilTestWithHlo, MultipleCallSitesForIota) {
  absl::string_view hlo_string = R"(
    HloModule module

    call_computation {
      %param = (s32[4096,4096]) parameter(0)
      ROOT %gte = s32[4096,4096] get-tuple-element(%param), index=0
    }

    ENTRY %main {
      %iota = s32[4096,4096] iota(), iota_dimension=0
      %tuple = (s32[4096,4096]) tuple(%iota)
      %call.0 = s32[4096,4096] call(%tuple), to_apply=call_computation
      %call.1 = s32[4096,4096] call(%tuple), to_apply=call_computation
      ROOT %add = s32[4096,4096] add(%call.0, %call.1)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // TODO(b/260601110): Actually recognize the iota.
  auto call_graph = CallGraph::Build(module.get());
  EXPECT_EQ(
      GetDimensionForIota(module->GetComputationWithName("call_computation")
                              ->root_instruction(),
                          *call_graph),
      std::nullopt);
}

}  // namespace
}  // namespace hlo_sharding_util
}  // namespace xla
