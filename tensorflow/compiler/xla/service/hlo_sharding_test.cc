/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/hlo_sharding.h"

#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace {

Array<int64> MakeArray(tensorflow::gtl::ArraySlice<int64> dimensions,
                       tensorflow::gtl::ArraySlice<int64> contents) {
  Array<int64> a(dimensions);
  std::copy(contents.begin(), contents.end(), a.begin());
  return a;
}

class HloShardingTest : public HloTestBase {};

TEST_F(HloShardingTest, Replicate) {
  Shape tile_shape = ShapeUtil::MakeShape(U32, {4});
  HloSharding sharding = HloSharding::Replicate();
  EXPECT_TRUE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_TRUE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(65535));

  HloSharding other = HloSharding::Replicate();
  EXPECT_EQ(other, sharding);

  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4}),
                                 /*num_devices=*/2));
  EXPECT_IS_NOT_OK(sharding.UniqueDevice());
}

TEST_F(HloShardingTest, DevicePlacement) {
  HloSharding sharding = HloSharding::AssignDevice(5);
  EXPECT_FALSE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_FALSE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(5));
  EXPECT_EQ(5, sharding.UniqueDevice().ValueOrDie());

  HloSharding other = HloSharding::Replicate();
  EXPECT_NE(other, sharding);

  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4}),
                                 /*num_devices=*/6));
  EXPECT_IS_NOT_OK(
      sharding.Validate(ShapeUtil::MakeShape(U32, {4}), /*num_devices=*/5));
}

TEST_F(HloShardingTest, Tile) {
  {
    // Test should fail because of a duplicate tile assignment.
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 0, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {4, 6}),
                                       /*num_devices=*/4));
  }

  {
    // Test should pass.
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 1, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4, 6}),
                                       /*num_devices=*/2));
  }

  {
    // Test should fail due to the tile being larger than the input space.
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 1, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {2, 2}),
                                       /*num_devices=*/4));
  }

  {
    // Test should fail due to the tile not dividing the input space into 4
    // sections (even with padding).
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 1, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {6, 3}),
                                       /*num_devices=*/4));
  }

  {
    // Test should pass.
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 3, 2, 1}));
    EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {3, 5}),
                                   /*num_devices=*/5));

    EXPECT_EQ(0, sharding.DeviceForTileIndex({0, 0}));
    EXPECT_EQ(3, sharding.DeviceForTileIndex({0, 1}));
    EXPECT_EQ(2, sharding.DeviceForTileIndex({1, 0}));
    EXPECT_EQ(1, sharding.DeviceForTileIndex({1, 1}));

    EXPECT_EQ(sharding.TileOffsetForDevice(0), (std::vector<int64>{0, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(3), (std::vector<int64>{0, 3}));
    EXPECT_EQ(sharding.TileOffsetForDevice(2), (std::vector<int64>{2, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(1), (std::vector<int64>{2, 3}));

    EXPECT_IS_NOT_OK(sharding.UniqueDevice());
  }
}

TEST_F(HloShardingTest, Hash) {
  auto hash_compare_equal = [](const HloSharding& a, const HloSharding& b) {
    if (a.Hash() != b.Hash()) {
      return false;
    }
    return a == b;
  };

  {
    HloSharding sharding1 = HloSharding::Replicate();
    HloSharding sharding2 = HloSharding::Replicate();
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    HloSharding sharding1 = HloSharding::AssignDevice(1);
    HloSharding sharding2 = HloSharding::AssignDevice(1);
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    HloSharding sharding1 = HloSharding::AssignDevice(1);
    HloSharding sharding2 = HloSharding::AssignDevice(2);
    EXPECT_FALSE(hash_compare_equal(sharding1, sharding2));
  }

  {
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding1 =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 3, 2, 1}));
    HloSharding sharding2 = HloSharding::Tile(ShapeUtil::MakeShape(U32, {2, 3}),
                                              MakeArray({2, 2}, {0, 3, 2, 1}));
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding1 =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 3, 2, 1}));
    HloSharding sharding2 = HloSharding::Tile(ShapeUtil::MakeShape(U32, {2, 3}),
                                              MakeArray({2, 2}, {0, 3, 2, 1}));
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    Shape tile_shape = ShapeUtil::MakeShape(U32, {2, 3});
    HloSharding sharding1 =
        HloSharding::Tile(tile_shape, MakeArray({2, 2}, {0, 3, 2, 1}));
    HloSharding sharding2 = HloSharding::Tile(ShapeUtil::MakeShape(U32, {2, 3}),
                                              MakeArray({2, 2}, {0, 3, 1, 2}));
    EXPECT_FALSE(hash_compare_equal(sharding1, sharding2));
  }
}

}  // namespace
}  // namespace xla
