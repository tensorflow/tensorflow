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

#include "xla/hlo/ir/tile_assignment.h"

#include <memory>
#include <vector>

#include "absl/hash/hash.h"
#include "xla/array3d.h"
#include "xla/test.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

std::vector<int64_t> ToVectorUsingEach(const TileAssignment& tile) {
  std::vector<int64_t> result;
  result.reserve(tile.num_elements());
  tile.Each([&](absl::Span<const int64_t> index, int64_t device) {
    result.push_back(device);
  });
  return result;
}

// TODO(b/281892190): Replicated sharding should not really have a tile
// assignment like this, but initially TileAssignment is supposed to closely
// resemble xla::Array's behavior, and HloSharding assign a 0-sized dimension
// rank 1 xla::Array for replicated shardings.
TEST(TileAssignmentTest, Replicated) {
  TileAssignment tile;
  EXPECT_EQ(tile.num_dimensions(), 1);
  EXPECT_EQ(tile.dim(0), 0);
}

TEST(TileAssignmentTest, Maximal) {
  TileAssignment tile(5);
  EXPECT_EQ(tile.num_dimensions(), 1);
  EXPECT_EQ(tile.dim(0), 1);
  EXPECT_EQ(tile(0), 5);
  EXPECT_EQ(tile({0}), 5);
  EXPECT_FALSE(tile.iota());
  EXPECT_TRUE(tile.UsesDevice(5));
  EXPECT_EQ(tile.first(), 5);
  EXPECT_FALSE(tile.UsesDevice(0));
  EXPECT_THAT(ToVectorUsingEach(tile), ElementsAre(5));
}

TEST(TileAssignmentTest, V1V2Equivalence) {
  Array3D<int64_t> array(
      {{{0, 8, 4, 12}, {1, 9, 5, 13}}, {{2, 10, 6, 14}, {3, 11, 7, 15}}});
  TileAssignment v1(std::make_shared<const Array<int64_t>>(array));
  TileAssignment v2({2, 2, 4}, {2, 2, 4}, {2, 1, 0});
  EXPECT_EQ(v1, v2);
  EXPECT_EQ(v2, v1);
  EXPECT_EQ(v1.first(), 0);
  EXPECT_EQ(v2.first(), 0);
  EXPECT_NE(v1.iota().has_value(), v2.iota().has_value());
  EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
}

TEST(TileAssignmentTest, CopyConstruction) {
  TileAssignment tile({2, 2, 4}, {2, 2, 4}, {2, 1, 0});
  TileAssignment copied(tile);
  EXPECT_EQ(tile, copied);
  EXPECT_EQ(tile.iota().has_value(), copied.iota().has_value());
  EXPECT_EQ(absl::HashOf(tile), absl::HashOf(copied));
}

TEST(TileAssignmentTest, CopyAssignment) {
  TileAssignment tile({2, 2, 4}, {2, 2, 4}, {2, 1, 0});
  TileAssignment copied = tile;
  EXPECT_EQ(tile, copied);
  EXPECT_EQ(tile.iota().has_value(), copied.iota().has_value());
  EXPECT_EQ(absl::HashOf(tile), absl::HashOf(copied));
}

class FormattedTileAssignmentTest : public ::testing::TestWithParam<bool> {
 protected:
  bool ShouldConvertToV1() { return GetParam(); }
};

TEST_P(FormattedTileAssignmentTest, TrivialIotaTile) {
  TileAssignment tile({4, 4, 2});
  EXPECT_EQ(tile.ToString(), "devices=[4,4,2]<=[32]");
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  EXPECT_EQ(tile, TileAssignment({4, 4, 2}));
  EXPECT_EQ(tile.num_dimensions(), 3);
  EXPECT_EQ(tile.dim(0), 4);
  EXPECT_EQ(tile.dim(1), 4);
  EXPECT_EQ(tile.dim(2), 2);
  EXPECT_EQ(tile(0, 0, 0), 0);
  EXPECT_EQ(tile({3, 2, 1}), 29);
  EXPECT_EQ(tile.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(tile.UsesDevice(0));
  EXPECT_TRUE(tile.UsesDevice(31));
  EXPECT_FALSE(tile.UsesDevice(32));
  EXPECT_THAT(
      ToVectorUsingEach(tile),
      ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31));
}

TEST_P(FormattedTileAssignmentTest, TransposedIotaTile) {
  TileAssignment tile({4, 4, 2}, {2, 4, 4}, {2, 1, 0});
  EXPECT_EQ(tile.ToString(), "devices=[4,4,2]<=[2,4,4]T(2,1,0)");
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  EXPECT_EQ(tile, TileAssignment({4, 4, 2}, {2, 4, 4}, {2, 1, 0}));
  EXPECT_EQ(tile.num_dimensions(), 3);
  EXPECT_EQ(tile.dim(0), 4);
  EXPECT_EQ(tile.dim(1), 4);
  EXPECT_EQ(tile.dim(2), 2);
  EXPECT_EQ(tile(0, 0, 0), 0);
  EXPECT_EQ(tile({3, 2, 1}), 27);
  EXPECT_EQ(tile.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(tile.UsesDevice(0));
  EXPECT_TRUE(tile.UsesDevice(31));
  EXPECT_FALSE(tile.UsesDevice(32));
  EXPECT_THAT(
      ToVectorUsingEach(tile),
      ElementsAre(0, 16, 4, 20, 8, 24, 12, 28, 1, 17, 5, 21, 9, 25, 13, 29, 2,
                  18, 6, 22, 10, 26, 14, 30, 3, 19, 7, 23, 11, 27, 15, 31));
}

TEST_P(FormattedTileAssignmentTest, NonCanonicalTransposedIotaTile) {
  TileAssignment tile({4, 8}, {2, 4, 4}, {1, 2, 0});
  EXPECT_EQ(tile.ToString(), "devices=[4,8]<=[2,16]T(1,0)");
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  EXPECT_EQ(tile, TileAssignment({4, 8}, {2, 16}, {1, 0}));
  EXPECT_EQ(tile.num_dimensions(), 2);
  EXPECT_EQ(tile.dim(0), 4);
  EXPECT_EQ(tile.dim(1), 8);
  EXPECT_EQ(tile(0, 0), 0);
  EXPECT_EQ(tile({3, 2}), 13);
  EXPECT_EQ(tile.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(tile.UsesDevice(0));
  EXPECT_TRUE(tile.UsesDevice(31));
  EXPECT_FALSE(tile.UsesDevice(32));
  EXPECT_THAT(
      ToVectorUsingEach(tile),
      ElementsAre(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24,
                  9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31));
}

TEST_P(FormattedTileAssignmentTest, ReshapeTrivalIotaTile) {
  TileAssignment tile({4, 4, 2});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment reshaped = tile.Reshape({2, 8, 2});
  EXPECT_NE(reshaped, tile);
  EXPECT_EQ(reshaped, TileAssignment({2, 8, 2}));
  EXPECT_EQ(reshaped.num_dimensions(), 3);
  EXPECT_EQ(reshaped.dim(0), 2);
  EXPECT_EQ(reshaped.dim(1), 8);
  EXPECT_EQ(reshaped.dim(2), 2);
  EXPECT_EQ(reshaped(0, 0, 0), 0);
  EXPECT_EQ(reshaped({1, 3, 1}), 23);
  EXPECT_EQ(reshaped.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(reshaped.UsesDevice(0));
  EXPECT_TRUE(reshaped.UsesDevice(31));
  EXPECT_FALSE(reshaped.UsesDevice(32));
  EXPECT_THAT(
      ToVectorUsingEach(reshaped),
      ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31));
}

TEST_P(FormattedTileAssignmentTest, ReshapeTransposedIotaTile) {
  TileAssignment tile({4, 4, 2}, {2, 4, 4}, {2, 1, 0});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment reshaped = tile.Reshape({2, 2, 4, 2});
  EXPECT_NE(reshaped, tile);
  EXPECT_EQ(reshaped, TileAssignment({2, 2, 4, 2}, {2, 4, 4}, {2, 1, 0}));
  EXPECT_EQ(reshaped.num_dimensions(), 4);
  EXPECT_EQ(reshaped.dim(0), 2);
  EXPECT_EQ(reshaped.dim(1), 2);
  EXPECT_EQ(reshaped.dim(2), 4);
  EXPECT_EQ(reshaped.dim(3), 2);
  EXPECT_EQ(reshaped(0, 0, 0, 0), 0);
  EXPECT_EQ(reshaped({1, 1, 2, 1}), 27);
  EXPECT_EQ(reshaped.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(reshaped.UsesDevice(0));
  EXPECT_TRUE(reshaped.UsesDevice(31));
  EXPECT_FALSE(reshaped.UsesDevice(32));
  EXPECT_THAT(
      ToVectorUsingEach(reshaped),
      ElementsAre(0, 16, 4, 20, 8, 24, 12, 28, 1, 17, 5, 21, 9, 25, 13, 29, 2,
                  18, 6, 22, 10, 26, 14, 30, 3, 19, 7, 23, 11, 27, 15, 31));
}

TEST_P(FormattedTileAssignmentTest, TransposeTrivalIotaTile) {
  TileAssignment tile({4, 4, 2});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({2, 0, 1});
  EXPECT_NE(xposed, tile);
  EXPECT_EQ(xposed, TileAssignment({2, 4, 4}, {16, 2}, {1, 0}));
  EXPECT_EQ(xposed.num_dimensions(), 3);
  EXPECT_EQ(xposed.dim(0), 2);
  EXPECT_EQ(xposed.dim(1), 4);
  EXPECT_EQ(xposed.dim(2), 4);
  EXPECT_EQ(xposed(0, 0, 0), 0);
  EXPECT_EQ(xposed({1, 3, 1}), 27);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(31));
  EXPECT_FALSE(xposed.UsesDevice(32));
  EXPECT_THAT(
      ToVectorUsingEach(xposed),
      ElementsAre(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1,
                  3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31));
}

TEST_P(FormattedTileAssignmentTest, TransposeTransposedIotaTile) {
  TileAssignment tile({4, 4, 2}, {2, 4, 4}, {2, 1, 0});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({0, 2, 1});
  EXPECT_NE(xposed, tile);
  EXPECT_EQ(xposed, TileAssignment({4, 2, 4}, {8, 4}, {1, 0}));
  EXPECT_EQ(xposed.num_dimensions(), 3);
  EXPECT_EQ(xposed.dim(0), 4);
  EXPECT_EQ(xposed.dim(1), 2);
  EXPECT_EQ(xposed.dim(2), 4);
  EXPECT_EQ(xposed(0, 0, 0), 0);
  EXPECT_EQ(xposed({3, 0, 3}), 15);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(31));
  EXPECT_FALSE(xposed.UsesDevice(32));
  EXPECT_THAT(
      ToVectorUsingEach(xposed),
      ElementsAre(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2,
                  6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15, 19, 23, 27, 31));
}

TEST_P(FormattedTileAssignmentTest, TransposeIotaTileWithDegernateDims) {
  TileAssignment tile({4, 4, 1}, {4, 4}, {1, 0});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({1, 2, 0});
  EXPECT_NE(xposed, tile);
  EXPECT_EQ(xposed, TileAssignment({4, 1, 4}));
  EXPECT_EQ(xposed.num_dimensions(), 3);
  EXPECT_EQ(xposed.dim(0), 4);
  EXPECT_EQ(xposed.dim(1), 1);
  EXPECT_EQ(xposed.dim(2), 4);
  EXPECT_EQ(xposed(0, 0, 0), 0);
  EXPECT_EQ(xposed({2, 0, 3}), 11);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(15));
  EXPECT_FALSE(xposed.UsesDevice(16));
  EXPECT_THAT(
      ToVectorUsingEach(xposed),
      ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
}

TEST_P(FormattedTileAssignmentTest, TransposeNoopIotaTile) {
  TileAssignment tile({4, 4}, {4, 4}, {1, 0});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({0, 1});
  EXPECT_EQ(xposed, tile);
  EXPECT_EQ(xposed.num_dimensions(), 2);
  EXPECT_EQ(xposed.dim(0), 4);
  EXPECT_EQ(xposed.dim(1), 4);
  EXPECT_EQ(xposed(0, 0), 0);
  EXPECT_EQ(xposed({2, 3}), 14);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(15));
  EXPECT_FALSE(xposed.UsesDevice(16));
  EXPECT_THAT(
      ToVectorUsingEach(xposed),
      ElementsAre(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
}

TEST_P(FormattedTileAssignmentTest, TransposeNoopIotaTileWithDegernateDims) {
  TileAssignment tile({1, 4, 1, 1, 4, 1}, {4, 4}, {1, 0});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({1, 5, 0, 4, 3, 2});
  EXPECT_NE(xposed, tile);
  EXPECT_EQ(xposed.num_dimensions(), 6);
  EXPECT_EQ(xposed.dim(0), 4);
  EXPECT_EQ(xposed.dim(1), 1);
  EXPECT_EQ(xposed.dim(2), 1);
  EXPECT_EQ(xposed.dim(3), 4);
  EXPECT_EQ(xposed.dim(4), 1);
  EXPECT_EQ(xposed.dim(5), 1);
  EXPECT_EQ(xposed(0, 0, 0, 0, 0, 0), 0);
  EXPECT_EQ(xposed({2, 0, 0, 3, 0, 0}), 14);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(15));
  EXPECT_FALSE(xposed.UsesDevice(16));
  EXPECT_THAT(
      ToVectorUsingEach(xposed),
      ElementsAre(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
}

INSTANTIATE_TEST_SUITE_P(All, FormattedTileAssignmentTest, ::testing::Bool());

}  // namespace
}  // namespace xla
