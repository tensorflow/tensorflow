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
#include "xla/hlo/testlib/test.h"

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

TEST(IotaTileAssignmentTest, TransposeCase1) {
  IotaTileAssignment tile =
      IotaTileAssignment::Create({4, 5, 24}, {15, 4, 8}, {1, 0, 2});
  auto transposed = tile.Transpose({0, 2, 1});
  EXPECT_EQ(transposed.has_value(), true);
  EXPECT_EQ(transposed->ToString(), "[4,24,5]<=[5,3,4,8]T(2,1,3,0)");
}

TEST(IotaTileAssignmentTest, TransposeCase2) {
  IotaTileAssignment tile = IotaTileAssignment::Create(
      {1, 32, 82, 1, 1, 128}, {1312, 32, 8}, {1, 0, 2});
  auto transposed = tile.Transpose({1, 3, 4, 5, 0, 2});
  EXPECT_EQ(transposed.has_value(), true);
  EXPECT_EQ(transposed->ToString(),
            "[32,1,1,128,1,82]<=[82,16,32,8]T(2,1,3,0)");
}

TEST(IotaTileAssignmentTest, TransposeCase3) {
  IotaTileAssignment tile =
      IotaTileAssignment::Create({1, 32, 82, 128}, {1312, 32, 8}, {1, 0, 2});
  auto transposed = tile.Transpose({0, 1, 3, 2});
  EXPECT_EQ(transposed.has_value(), true);
  EXPECT_EQ(transposed->ToString(), "[1,32,128,82]<=[82,16,32,8]T(2,1,3,0)");
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

TEST_P(FormattedTileAssignmentTest,
       TransposeIotaTileSplittingCanonicalizedReshapeDims) {
  TileAssignment tile({8, 2, 16}, {16, 16}, {1, 0});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({0, 2, 1});
  EXPECT_NE(xposed, tile);
  EXPECT_EQ(xposed, TileAssignment({8, 16, 2}, {16, 8, 2}, {1, 0, 2}));
  EXPECT_EQ(xposed.num_dimensions(), 3);
  EXPECT_EQ(xposed.dim(0), 8);
  EXPECT_EQ(xposed.dim(1), 16);
  EXPECT_EQ(xposed.dim(2), 2);
  EXPECT_EQ(xposed(0, 0, 0), 0);
  EXPECT_EQ(xposed({2, 7, 1}), 117);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(255));
  EXPECT_FALSE(xposed.UsesDevice(256));
  EXPECT_THAT(
      ToVectorUsingEach(xposed),
      ElementsAre(
          0, 1, 16, 17, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97, 112, 113, 128,
          129, 144, 145, 160, 161, 176, 177, 192, 193, 208, 209, 224, 225, 240,
          241, 2, 3, 18, 19, 34, 35, 50, 51, 66, 67, 82, 83, 98, 99, 114, 115,
          130, 131, 146, 147, 162, 163, 178, 179, 194, 195, 210, 211, 226, 227,
          242, 243, 4, 5, 20, 21, 36, 37, 52, 53, 68, 69, 84, 85, 100, 101, 116,
          117, 132, 133, 148, 149, 164, 165, 180, 181, 196, 197, 212, 213, 228,
          229, 244, 245, 6, 7, 22, 23, 38, 39, 54, 55, 70, 71, 86, 87, 102, 103,
          118, 119, 134, 135, 150, 151, 166, 167, 182, 183, 198, 199, 214, 215,
          230, 231, 246, 247, 8, 9, 24, 25, 40, 41, 56, 57, 72, 73, 88, 89, 104,
          105, 120, 121, 136, 137, 152, 153, 168, 169, 184, 185, 200, 201, 216,
          217, 232, 233, 248, 249, 10, 11, 26, 27, 42, 43, 58, 59, 74, 75, 90,
          91, 106, 107, 122, 123, 138, 139, 154, 155, 170, 171, 186, 187, 202,
          203, 218, 219, 234, 235, 250, 251, 12, 13, 28, 29, 44, 45, 60, 61, 76,
          77, 92, 93, 108, 109, 124, 125, 140, 141, 156, 157, 172, 173, 188,
          189, 204, 205, 220, 221, 236, 237, 252, 253, 14, 15, 30, 31, 46, 47,
          62, 63, 78, 79, 94, 95, 110, 111, 126, 127, 142, 143, 158, 159, 174,
          175, 190, 191, 206, 207, 222, 223, 238, 239, 254, 255));
}

TEST_P(FormattedTileAssignmentTest,
       TransposeIotaTileSplittingBothCanonicalizedReshapeDimsAndTileDims) {
  TileAssignment tile({14, 3, 5}, {6, 5, 7}, {2, 0, 1});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({1, 0, 2});
  EXPECT_NE(xposed, tile);
  EXPECT_EQ(xposed, TileAssignment({3, 14, 5}, {2, 3, 5, 7}, {1, 3, 0, 2}));
  EXPECT_EQ(xposed.num_dimensions(), 3);
  EXPECT_EQ(xposed.dim(0), 3);
  EXPECT_EQ(xposed.dim(1), 14);
  EXPECT_EQ(xposed.dim(2), 5);
  EXPECT_EQ(xposed(0, 0, 0), 0);
  EXPECT_EQ(xposed({2, 11, 3}), 201);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(209));
  EXPECT_FALSE(xposed.UsesDevice(210));
  EXPECT_THAT(
      ToVectorUsingEach(xposed),
      ElementsAre(
          0, 7, 14, 21, 28, 105, 112, 119, 126, 133, 1, 8, 15, 22, 29, 106, 113,
          120, 127, 134, 2, 9, 16, 23, 30, 107, 114, 121, 128, 135, 3, 10, 17,
          24, 31, 108, 115, 122, 129, 136, 4, 11, 18, 25, 32, 109, 116, 123,
          130, 137, 5, 12, 19, 26, 33, 110, 117, 124, 131, 138, 6, 13, 20, 27,
          34, 111, 118, 125, 132, 139, 35, 42, 49, 56, 63, 140, 147, 154, 161,
          168, 36, 43, 50, 57, 64, 141, 148, 155, 162, 169, 37, 44, 51, 58, 65,
          142, 149, 156, 163, 170, 38, 45, 52, 59, 66, 143, 150, 157, 164, 171,
          39, 46, 53, 60, 67, 144, 151, 158, 165, 172, 40, 47, 54, 61, 68, 145,
          152, 159, 166, 173, 41, 48, 55, 62, 69, 146, 153, 160, 167, 174, 70,
          77, 84, 91, 98, 175, 182, 189, 196, 203, 71, 78, 85, 92, 99, 176, 183,
          190, 197, 204, 72, 79, 86, 93, 100, 177, 184, 191, 198, 205, 73, 80,
          87, 94, 101, 178, 185, 192, 199, 206, 74, 81, 88, 95, 102, 179, 186,
          193, 200, 207, 75, 82, 89, 96, 103, 180, 187, 194, 201, 208, 76, 83,
          90, 97, 104, 181, 188, 195, 202, 209));
}

TEST_P(FormattedTileAssignmentTest,
       TransposeIotaTileGroupingCanonicalizedReshapeDims) {
  TileAssignment tile({1, 4, 16}, {4, 4, 4}, {1, 0, 2});
  if (ShouldConvertToV1()) {
    tile = TileAssignment(tile.shared_array());
  }
  TileAssignment xposed = tile.Transpose({2, 0, 1});
  EXPECT_NE(xposed, tile);
  EXPECT_EQ(xposed, TileAssignment({16, 1, 4}, {4, 4, 4}, {0, 2, 1}));
  EXPECT_EQ(xposed.num_dimensions(), 3);
  EXPECT_EQ(xposed.dim(0), 16);
  EXPECT_EQ(xposed.dim(1), 1);
  EXPECT_EQ(xposed.dim(2), 4);
  EXPECT_EQ(xposed(0, 0, 0), 0);
  EXPECT_EQ(xposed({7, 0, 3}), 31);
  EXPECT_EQ(xposed.iota().has_value(), !ShouldConvertToV1());
  EXPECT_TRUE(xposed.UsesDevice(0));
  EXPECT_TRUE(xposed.UsesDevice(63));
  EXPECT_FALSE(xposed.UsesDevice(64));
  EXPECT_THAT(ToVectorUsingEach(xposed),
              ElementsAre(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                          16, 20, 24, 28, 17, 21, 25, 29, 18, 22, 26, 30, 19,
                          23, 27, 31, 32, 36, 40, 44, 33, 37, 41, 45, 34, 38,
                          42, 46, 35, 39, 43, 47, 48, 52, 56, 60, 49, 53, 57,
                          61, 50, 54, 58, 62, 51, 55, 59, 63));
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
