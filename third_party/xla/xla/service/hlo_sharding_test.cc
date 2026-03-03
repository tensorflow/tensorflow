/* Copyright 2017 The OpenXLA Authors.

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

#include <cstdint>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/array3d.h"
#include "xla/array4d.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

OpMetadata GetMetadata(const std::string& op_name) {
  OpMetadata metadata;
  metadata.set_op_name(op_name);
  return metadata;
}

std::vector<OpMetadata> SingleMetadata() { return {GetMetadata("a")}; }

std::vector<OpMetadata> ListMetadata() {
  return {GetMetadata("b"), GetMetadata("c")};
}

class HloShardingTest : public HloHardwareIndependentTestBase {};

// TODO(b/456418464): Parameterize `HloShardingTest` itself after supporting
// NamedSharding in all methods.
class HloShardingRepresentationTest
    : public HloShardingTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(HloShardingRepresentationTest, Replicate) {
  bool use_named_sharding = GetParam();
  HloSharding sharding = HloSharding::Replicate({}, use_named_sharding);
  EXPECT_EQ(sharding.UseNamedShardingLeaf(), use_named_sharding);
  EXPECT_TRUE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_TRUE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(65535));

  HloSharding other = HloSharding::Replicate({}, use_named_sharding);
  EXPECT_EQ(other, sharding);
  // Shardings are compared regardless of representation.
  EXPECT_EQ(HloSharding::Replicate(),
            HloSharding::Replicate({}, /*use_named_sharding=*/true));

  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4}),
                                 /*num_devices=*/2));
  EXPECT_FALSE(sharding.HasUniqueDevice());
}

TEST_P(HloShardingRepresentationTest, DevicePlacement) {
  bool use_named_sharding = GetParam();
  HloSharding sharding = HloSharding::AssignDevice(5, {}, use_named_sharding);
  EXPECT_EQ(sharding.UseNamedShardingLeaf(), use_named_sharding);
  EXPECT_FALSE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_FALSE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(5));
  EXPECT_EQ(5, sharding.GetUniqueDevice());

  HloSharding other = HloSharding::Replicate({}, use_named_sharding);
  EXPECT_NE(other, sharding);
  // Shardings are compared regardless of representation.
  EXPECT_EQ(HloSharding::AssignDevice(5),
            HloSharding::AssignDevice(5, {}, /*use_named_sharding=*/true));

  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4}),
                                 /*num_devices=*/6));
  EXPECT_IS_NOT_OK(
      sharding.Validate(ShapeUtil::MakeShape(U32, {4}), /*num_devices=*/5));

  ShapeTree<HloSharding> shape_tree =
      sharding.GetAsShapeTree(ShapeUtil::MakeShape(U32, {4}));
  EXPECT_EQ(shape_tree.element({}), sharding);
  EXPECT_TRUE(shape_tree.IsLeaf({}));
}

TEST_F(HloShardingTest, ProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<OpSharding>(R"pb(
    type: TUPLE
    tuple_shardings {
      type: OTHER
      tile_assignment_devices: 0
      tile_assignment_devices: 1
      tile_assignment_dimensions: 1
      tile_assignment_dimensions: 2
      metadata { op_name: "a" }
      metadata { op_name: "b" }
    }
    tuple_shardings {
      type: REPLICATED
      metadata { op_name: "c" }
    }
    tuple_shardings { type: MANUAL }
  )pb");
  HloSharding sharding = HloSharding::FromProto(proto).value();
  EXPECT_THAT(sharding.ToProto(), EqualsProto(proto));
}

TEST_F(HloShardingTest, IotaProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<OpSharding>(R"pb(
    type: TUPLE
    tuple_shardings {
      type: OTHER
      tile_assignment_dimensions: 6
      tile_assignment_dimensions: 1
      iota_reshape_dims: 3
      iota_reshape_dims: 2
      iota_transpose_perm: 1
      iota_transpose_perm: 0
      metadata { op_name: "a" }
      metadata { op_name: "b" }
    }
    tuple_shardings {
      type: REPLICATED
      metadata { op_name: "c" }
    }
    tuple_shardings { type: MANUAL }
  )pb");
  HloSharding sharding = HloSharding::FromProto(proto).value();
  EXPECT_THAT(sharding.ToProto(), EqualsProto(proto));
}

TEST_F(HloShardingTest, NamedShardingTupleProtoRoundTrip) {
  auto proto = ParseTextProtoOrDie<OpSharding>(R"pb(
    type: TUPLE
    tuple_shardings {
      named_sharding {
        mesh {
          axes { name: "a" size: 4 }
          axes { name: "b" size: 4 }
        }
        dim_shardings {
          axes {
            mesh_axis_index: 0
            sub_axis_info { pre_size: 1 size: 2 }
          }
          is_closed: true
        }
        dim_shardings {
          axes { mesh_axis_index: 1 }
          is_closed: false
        }
      }
    }
    tuple_shardings {
      named_sharding {
        mesh {
          axes { name: "a" size: 4 }
          axes { name: "b" size: 4 }
        }
        dim_shardings {
          axes {
            mesh_axis_index: 0
            sub_axis_info { pre_size: 2 size: 2 }
          }
          is_closed: true
        }
        dim_shardings {
          axes { mesh_axis_index: 1 }
          is_closed: false
        }
      }
    }
  )pb");

  HloSharding sharding = HloSharding::FromProto(proto).value();

  EXPECT_THAT(sharding.ToProto(), EqualsProto(proto));
}

using TileTest = HloShardingTest;
TEST_F(TileTest, FailsWithDuplicateDeviceInTileAssignment) {
  HloSharding sharding =
      HloSharding::Tile(Array<int64_t>({2, 2}, {0, 0, 2, 3}));
  EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {4, 6}),
                                     /*num_devices=*/4));
  // No need to test NamedSharding here as constructing Mesh with duplicate
  // device ids is not valid.
}

TEST_F(TileTest, FailsWhenDeviceUsedGreaterThanNumDevices) {
  HloSharding sharding =
      HloSharding::Tile(Array<int64_t>({2, 2}, {0, 1, 2, 3}));
  EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4, 6}),
                                     /*num_devices=*/2));
  {
    Mesh mesh({2, 2}, {"x", "y"});
    NamedSharding named_sharding =
        test_utils::FromAxisNames(mesh, {{"x"}, {"y"}});
    HloSharding sharding(named_sharding);

    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {4, 6}),
                                       /*num_devices=*/2));
  }
}

TEST_F(TileTest, FailsWhenNotAllDevicesArePresentInTileAssignment) {
  HloSharding sharding =
      HloSharding::Tile(Array<int64_t>({2, 2}, {0, 1, 2, 3}));
  EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4, 6}),
                                     /*num_devices=*/5));
  {
    Mesh mesh({2, 2}, {"x", "y"});
    NamedSharding named_sharding =
        test_utils::FromAxisNames(mesh, {{"x"}, {"y"}});
    HloSharding sharding(named_sharding);

    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {4, 6}),
                                       /*num_devices=*/5));
  }
}

TEST_F(TileTest, PassesValidationAndMatchesTileInfo) {
  Shape shape = ShapeUtil::MakeShape(U32, {4, 5});
  HloSharding sharding =
      HloSharding::Tile(Array<int64_t>({2, 2}, {0, 3, 2, 1}));
  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {3, 5}),
                                 /*num_devices=*/4));
  Mesh mesh(Array<int64_t>({2, 2}, {0, 3, 2, 1}), {"x", "y"});
  HloSharding named_sharding(test_utils::FromAxisNames(mesh, {{"x"}, {"y"}}));

  for (const HloSharding& sharding : {sharding, named_sharding}) {
    EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {3, 5}),
                                   /*num_devices=*/4));

    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 0),
              (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 3),
              (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 2),
              (std::vector<int64_t>{2, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 1),
              (std::vector<int64_t>{2, 3}));

    EXPECT_EQ(sharding.TileLimitForDevice(shape, 0),
              (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(sharding.TileLimitForDevice(shape, 3),
              (std::vector<int64_t>{2, 5}));
    EXPECT_EQ(sharding.TileLimitForDevice(shape, 2),
              (std::vector<int64_t>{4, 3}));
    EXPECT_EQ(sharding.TileLimitForDevice(shape, 1),
              (std::vector<int64_t>{4, 5}));

    EXPECT_FALSE(sharding.HasUniqueDevice());

    // {device_index, tile_offest, tile_limit}.
    std::vector<std::tuple<int, std::vector<int64_t>, std::vector<int64_t>>>
        tiles;
    TF_ASSERT_OK(sharding.EachTile(
        shape.dimensions(),
        [&tiles](int device_index, absl::Span<const int64_t> tile_offset,
                 absl::Span<const int64_t> tile_limit) {
          std::vector<int64_t> offset(tile_offset.begin(), tile_offset.end());
          std::vector<int64_t> limit(tile_limit.begin(), tile_limit.end());
          tiles.emplace_back(device_index, std::move(offset), std::move(limit));
        }));
    EXPECT_THAT(tiles, ::testing::UnorderedElementsAre(
                           std::make_tuple(0, std::vector<int64_t>{0, 0},
                                           std::vector<int64_t>{2, 3}),
                           std::make_tuple(1, std::vector<int64_t>{2, 3},
                                           std::vector<int64_t>{4, 5}),
                           std::make_tuple(2, std::vector<int64_t>{2, 0},
                                           std::vector<int64_t>{4, 3}),
                           std::make_tuple(3, std::vector<int64_t>{0, 3},
                                           std::vector<int64_t>{2, 5})));
  }
}

struct HloShardingComparisonTestParam {
  // Shardings to compare.
  HloSharding sharding1;
  HloSharding sharding2;
  // Shape to apply the shardings on.
  Shape shape;
  // If both shardings should compare equal.
  bool is_equivalent;
};

class HloShardingComparisonTest
    : public HloShardingTest,
      public ::testing::WithParamInterface<HloShardingComparisonTestParam> {};

TEST_P(HloShardingComparisonTest, TileEquivalence) {
  const HloSharding& sharding1 = GetParam().sharding1;
  const HloSharding& sharding2 = GetParam().sharding2;
  const Shape& shape = GetParam().shape;
  const bool is_equivalent = GetParam().is_equivalent;
  const int num_devices = sharding1.num_devices();

  EXPECT_IS_OK(sharding1.Validate(shape, num_devices));
  EXPECT_IS_OK(sharding2.Validate(shape, num_devices));

  auto get_tile_info = [&](const HloSharding& sharding, int64_t device) {
    return std::make_tuple(sharding.TileIndexForDevice(device),
                           sharding.TileOffsetForDevice(shape, device),
                           sharding.TileLimitForDevice(shape, device),
                           sharding.TileShape(shape, device));
  };

  bool tiles_equivalent = true;
  for (int i = 0; i < num_devices; ++i) {
    if (get_tile_info(sharding1, i) != get_tile_info(sharding2, i)) {
      tiles_equivalent = false;
      break;
    }
  }
  EXPECT_EQ(tiles_equivalent, is_equivalent)
      << sharding1 << " vs " << sharding2;
}

INSTANTIATE_TEST_SUITE_P(TileEquivalence, HloShardingComparisonTest, [] {
  const Mesh mesh_a2b2({2, 2}, {"a", "b"});
  const Mesh mesh_a2b3({2, 3}, {"a", "b"});
  return ::testing::Values(
      HloShardingComparisonTestParam{
          HloSharding::IotaTile({2, 2}),
          HloSharding(test_utils::FromAxisNames(mesh_a2b2, {{"a"}, {"b"}})),
          ShapeUtil::MakeShape(U32, {2, 3}), true},
      HloShardingComparisonTestParam{
          HloSharding::IotaTile({2, 2}),
          HloSharding(test_utils::FromAxisNames(mesh_a2b2, {{"b"}, {"a"}})),
          ShapeUtil::MakeShape(U32, {2, 3}), false},
      HloShardingComparisonTestParam{
          HloSharding::IotaTile({2, 2}, {2, 2}, {1, 0}),
          HloSharding(test_utils::FromAxisNames(mesh_a2b2, {{"b"}, {"a"}})),
          ShapeUtil::MakeShape(U32, {2, 3}), true},
      HloShardingComparisonTestParam{
          HloSharding::IotaTile({6}),
          HloSharding(test_utils::FromAxisNames(mesh_a2b3, {{"a", "b"}})),
          ShapeUtil::MakeShape(U32, {13}), true},
      HloShardingComparisonTestParam{
          HloSharding::IotaTile({6}),
          HloSharding(test_utils::FromAxisNames(mesh_a2b3, {{"b", "a"}})),
          ShapeUtil::MakeShape(U32, {13}), false},
      HloShardingComparisonTestParam{
          HloSharding::IotaTile({6}, {2, 3}, {1, 0}),
          HloSharding(test_utils::FromAxisNames(mesh_a2b3, {{"b", "a"}})),
          ShapeUtil::MakeShape(U32, {13}), true});
}());

TEST_F(HloShardingTest, EachTile) {
  auto validate = [](const Shape& shape,
                     const HloSharding& sharding) -> absl::Status {
    return sharding.EachTile(
        shape.dimensions(),
        [&shape, &sharding](int device_index,
                            absl::Span<const int64_t> tile_offset,
                            absl::Span<const int64_t> tile_limit) {
          EXPECT_EQ(tile_offset,
                    sharding.TileOffsetForDevice(shape, device_index));
          EXPECT_EQ(tile_limit,
                    sharding.TileLimitForDevice(shape, device_index));
        });
  };
  {
    // 6-way sharded along axis 0, 1-way sharded along axis 1.
    HloSharding sharding = HloSharding::Tile(TileAssignment({6, 1}));
    Shape shape = ShapeUtil::MakeShape(U32, {12, 20});
    TF_EXPECT_OK(validate(shape, sharding));

    {
      Mesh mesh({6}, {"x"});
      NamedSharding named_sharding =
          test_utils::FromAxisNames(mesh, {{"x"}, {}});

      TF_EXPECT_OK(validate(shape, HloSharding(named_sharding)));
    }
  }
  {
    // 6-way sharded along axis 0, 1-way sharded along axis 1.
    HloSharding sharding = HloSharding::Tile(TileAssignment({6, 1}));
    Shape shape = ShapeUtil::MakeShape(U32, {11, 20});
    TF_EXPECT_OK(validate(shape, sharding));

    {
      Mesh mesh({6}, {"x"});
      NamedSharding named_sharding =
          test_utils::FromAxisNames(mesh, {{"x"}, {}});

      TF_EXPECT_OK(validate(shape, HloSharding(named_sharding)));
    }
  }
  {
    // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
    // replicated by 3 times.
    HloSharding sharding = HloSharding::PartialTile(TileAssignment({2, 1, 3}));
    Shape shape = ShapeUtil::MakeShape(U32, {10, 20});
    TF_EXPECT_OK(validate(shape, sharding));

    {
      Mesh mesh({2, 3}, {"x", "y"});
      NamedSharding named_sharding =
          test_utils::FromAxisNames(mesh, {{"x"}, {}});

      TF_EXPECT_OK(validate(shape, HloSharding(named_sharding)));
    }
  }
  {
    // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
    // replicated by 3 times.
    HloSharding sharding = HloSharding::Subgroup(TileAssignment({2, 1, 3}),
                                                 {OpSharding::REPLICATED});
    Shape shape = ShapeUtil::MakeShape(U32, {10, 20});
    TF_EXPECT_OK(validate(shape, sharding));

    {
      Mesh mesh({2, 3}, {"x", "y"});
      NamedSharding named_sharding = test_utils::FromAxisNames(
          mesh, {{"x"}, {}}, /*replicated_axes=*/{"y"});

      TF_EXPECT_OK(validate(shape, HloSharding(named_sharding)));
    }
  }
}

TEST_F(HloShardingTest, V1V2TileEquivalence) {
  {
    HloSharding v1 = HloSharding::Tile(Array<int64_t>({2, 2}, {0, 1, 2, 3}));
    HloSharding v2 = HloSharding::IotaTile({2, 2});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
  {
    HloSharding v1 = HloSharding::Tile(Array<int64_t>({2, 2}, {0, 2, 1, 3}));
    HloSharding v2 = HloSharding::IotaTile({2, 2}, {2, 2}, {1, 0});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
  {
    HloSharding v1 =
        HloSharding::Tile(Array<int64_t>({2, 2, 2}, {0, 2, 4, 6, 1, 3, 5, 7}));
    HloSharding v2 = HloSharding::IotaTile({2, 2, 2}, {2, 2, 2}, {2, 0, 1});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
}

TEST_F(HloShardingTest, V1V2PartialTileEquivalence) {
  {
    HloSharding v1 =
        HloSharding::PartialTile(Array<int64_t>({2, 2}, {0, 1, 2, 3}));
    HloSharding v2 = HloSharding::PartialTile(TileAssignment({2, 2}));
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
  {
    HloSharding v1 =
        HloSharding::PartialTile(Array<int64_t>({2, 2}, {0, 2, 1, 3}));
    HloSharding v2 =
        HloSharding::PartialTile(TileAssignment({2, 2}, {2, 2}, {1, 0}));
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
  {
    HloSharding v1 = HloSharding::PartialTile(
        Array<int64_t>({2, 2, 2}, {0, 2, 4, 6, 1, 3, 5, 7}));
    HloSharding v2 = HloSharding::PartialTile(
        TileAssignment({2, 2, 2}, {2, 2, 2}, {2, 0, 1}));
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
}

TEST_F(HloShardingTest, V1V2SubgroupEquivalence) {
  {
    HloSharding v1 =
        HloSharding::Subgroup(Array<int64_t>({2, 2}, {0, 1, 2, 3}),
                              {OpSharding::MANUAL, OpSharding::REPLICATED});
    HloSharding v2 = HloSharding::Subgroup(
        TileAssignment({2, 2}), {OpSharding::MANUAL, OpSharding::REPLICATED});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
  {
    HloSharding v1 =
        HloSharding::Subgroup(Array<int64_t>({2, 2}, {0, 2, 1, 3}),
                              {OpSharding::MANUAL, OpSharding::REPLICATED});
    HloSharding v2 =
        HloSharding::Subgroup(TileAssignment({2, 2}, {2, 2}, {1, 0}),
                              {OpSharding::MANUAL, OpSharding::REPLICATED});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
  {
    HloSharding v1 = HloSharding::Subgroup(
        Array<int64_t>({2, 2, 2}, {0, 2, 4, 6, 1, 3, 5, 7}),
        {OpSharding::MANUAL, OpSharding::REPLICATED});
    HloSharding v2 =
        HloSharding::Subgroup(TileAssignment({2, 2, 2}, {2, 2, 2}, {2, 0, 1}),
                              {OpSharding::MANUAL, OpSharding::REPLICATED});
    EXPECT_EQ(v1, v2);
    EXPECT_EQ(absl::HashOf(v1), absl::HashOf(v2));
  }
}

// Tests that empty tuple is supported.
TEST_P(HloShardingRepresentationTest, EmptySingleTuple) {
  bool use_named_sharding = GetParam();
  HloSharding sharding = HloSharding::SingleTuple(
      ShapeUtil::MakeTupleShape({}),
      HloSharding::AssignDevice(0, {}, use_named_sharding));
  EXPECT_TRUE(sharding.ExtractSingleSharding());
  EXPECT_EQ(sharding.ExtractSingleSharding()->UseNamedShardingLeaf(),
            use_named_sharding);
}

// Tests that empty tuple is not a shard group.
TEST_P(HloShardingRepresentationTest, EmptySingleTupleIsNotShardGroup) {
  bool use_named_sharding = GetParam();
  HloSharding sharding = HloSharding::SingleTuple(
      ShapeUtil::MakeTupleShape({}),
      HloSharding::AssignDevice(0, {}, use_named_sharding));
  EXPECT_FALSE(sharding.IsShardGroup());
  EXPECT_FALSE(sharding.IsShardAs());
  EXPECT_FALSE(sharding.IsShardLike());
}

INSTANTIATE_TEST_SUITE_P(HloShardingRepresentationTest,
                         HloShardingRepresentationTest,
                         ::testing::Values(false, true));

TEST_F(HloShardingTest, NestedTuple) {
  // nested_tuple_shape = (f32[], (f32[3]), f32[4, 6])
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(F32, {}),
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3})}),
      ShapeUtil::MakeShape(F32, {4, 6}),
  });

  HloSharding tiled_sharding = HloSharding::Tile(Array<int64_t>({{0, 1}}));
  OpSharding proto;
  proto.set_type(OpSharding::TUPLE);
  *proto.add_tuple_shardings() = HloSharding::Replicate().ToProto();
  *proto.add_tuple_shardings() = HloSharding::AssignDevice(0).ToProto();
  *proto.add_tuple_shardings() = tiled_sharding.ToProto();
  HloSharding tuple_sharding = HloSharding::FromProto(proto).value();

  ShapeTree<HloSharding> shape_tree =
      tuple_sharding.GetAsShapeTree(nested_tuple_shape);
  EXPECT_EQ(shape_tree.element({0}), HloSharding::Replicate());
  EXPECT_EQ(shape_tree.element({1, 0}), HloSharding::AssignDevice(0));
  EXPECT_EQ(shape_tree.element({2}), tiled_sharding);

  EXPECT_IS_OK(tuple_sharding.Validate(nested_tuple_shape, /*num_devices=*/2));
  // Test should fail because tuple element count does not match.
  EXPECT_IS_NOT_OK(tuple_sharding.Validate(ShapeUtil::MakeTupleShape({}),
                                           /*num_devices=*/5));
  // Test should fail because the input type is not a tuple.
  EXPECT_IS_NOT_OK(tuple_sharding.Validate(ShapeUtil::MakeShape(F32, {}),
                                           /*num_devices=*/5));
}

TEST_F(HloShardingTest, DeviceAssignmentTiledSharding) {
  TileAssignment ta({2, 4}, {4, 2}, {1, 0});
  HloSharding sharding = HloSharding::Tile(ta);

  EXPECT_EQ(sharding.device_assignment(), ta);
}
TEST_F(HloShardingTest, DeviceAssignmentNamedSharding) {
  TileAssignment ta({2, 4}, {4, 2}, {1, 0});
  Mesh mesh(ta, {"a", "b"});
  HloSharding hlo_sharding_from_named(
      test_utils::FromAxisNames(mesh, {{"a"}, {"b"}}));

  EXPECT_EQ(hlo_sharding_from_named.device_assignment(), ta);
}

TEST_F(HloShardingTest, NormalizeTrivialSubgroupToManual) {
  HloSharding sharding =
      HloSharding::Subgroup(Array<int64_t>({1, 2, 1}, {0, 1}),
                            {OpSharding::MANUAL, OpSharding::REPLICATED});
  EXPECT_TRUE(sharding.IsManual());
}

TEST_F(HloShardingTest, Hash) {
  auto hash_compare_equal = [](const HloSharding& a, const HloSharding& b) {
    if (absl::HashOf(a) != absl::HashOf(b)) {
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
    HloSharding sharding1 =
        HloSharding::Tile(Array<int64_t>({2, 2}, {0, 3, 2, 1}));
    HloSharding sharding2 =
        HloSharding::Tile(Array<int64_t>({2, 2}, {0, 3, 2, 1}));
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    HloSharding sharding1 = HloSharding::IotaTile({3, 4});
    HloSharding sharding2 = HloSharding::Tile(
        Array<int64_t>({3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  HloSharding default_sharding = HloSharding::Replicate();
  {
    ShapeTree<HloSharding> shape_tree(ShapeUtil::MakeTupleShape({}),
                                      default_sharding);
    HloSharding sharding1 = HloSharding::Replicate();
    HloSharding sharding2 = HloSharding::Tuple(shape_tree);
    EXPECT_FALSE(hash_compare_equal(sharding1, sharding2));
  }

  {
    ShapeTree<HloSharding> shape_tree(ShapeUtil::MakeTupleShape({}),
                                      default_sharding);
    HloSharding sharding1 = HloSharding::Tuple(shape_tree);
    HloSharding sharding2 = HloSharding::Tuple(shape_tree);
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }

  {
    ShapeTree<HloSharding> shape_tree1(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree1.mutable_element({0}) = HloSharding::Replicate();
    ShapeTree<HloSharding> shape_tree2(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree2.mutable_element({0}) = HloSharding::AssignDevice(0);
    HloSharding sharding1 = HloSharding::Tuple(shape_tree1);
    HloSharding sharding2 = HloSharding::Tuple(shape_tree2);
    EXPECT_FALSE(hash_compare_equal(sharding1, sharding2));
  }

  {
    ShapeTree<HloSharding> shape_tree1(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree1.mutable_element({0}) = HloSharding::AssignDevice(0);
    ShapeTree<HloSharding> shape_tree2(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4})}),
        default_sharding);
    *shape_tree2.mutable_element({0}) = HloSharding::AssignDevice(0);
    HloSharding sharding1 = HloSharding::Tuple(shape_tree1);
    HloSharding sharding2 = HloSharding::Tuple(shape_tree2);
    EXPECT_TRUE(hash_compare_equal(sharding1, sharding2));
  }
}

using ShardingWithMetadataParamType =
    std::tuple<std::vector<OpMetadata>, std::string>;

TEST_F(HloShardingTest, ToStringReplicatedTest) {
  HloSharding sharding = HloSharding::Replicate({});
  EXPECT_EQ(sharding.ToString(), "{replicated}");
}

class HloReplicateShardingWithMetadataTest
    : public ::testing::TestWithParam<ShardingWithMetadataParamType> {};

TEST_P(HloReplicateShardingWithMetadataTest, ToStringTest) {
  HloSharding sharding = HloSharding::Replicate(std::get<0>(GetParam()));
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false), "{replicated}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ToString, HloReplicateShardingWithMetadataTest,
    ::testing::Values(
        std::make_tuple(std::vector<OpMetadata>(), "{replicated}"),
        std::make_tuple(SingleMetadata(),
                        "{replicated metadata={op_name=\"a\"}}"),
        std::make_tuple(
            ListMetadata(),
            "{replicated metadata={{op_name=\"b\"}, {op_name=\"c\"}}}")));

TEST_F(HloShardingTest, ToStringAssignDeviceTest) {
  HloSharding sharding = HloSharding::AssignDevice(7);
  EXPECT_EQ(sharding.ToString(), "{maximal device=7}");
}

class HloAssignDeviceShardingWithMetadataTest
    : public ::testing::TestWithParam<ShardingWithMetadataParamType> {};

TEST_P(HloAssignDeviceShardingWithMetadataTest, ToStringTest) {
  HloSharding sharding = HloSharding::AssignDevice(7, std::get<0>(GetParam()));
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false),
            "{maximal device=7}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ToString, HloAssignDeviceShardingWithMetadataTest,
    ::testing::Values(
        std::make_tuple(std::vector<OpMetadata>(), "{maximal device=7}"),
        std::make_tuple(SingleMetadata(),
                        "{maximal device=7 metadata={op_name=\"a\"}}"),
        std::make_tuple(
            ListMetadata(),
            "{maximal device=7 metadata={{op_name=\"b\"}, {op_name=\"c\"}}}")));

TEST_F(HloShardingTest, ToStringTiledTest) {
  HloSharding sharding =
      HloSharding::Tile(Array3D<int64_t>({{{2, 3}}, {{5, 7}}}));
  EXPECT_EQ(sharding.ToString(), "{devices=[2,1,2]2,3,5,7}");
}

TEST_F(HloShardingTest, ToStringIotaTiledTest) {
  HloSharding sharding = HloSharding::IotaTile({3, 4}, {2, 2, 3}, {2, 1, 0});
  EXPECT_EQ(sharding.ToString(), "{devices=[3,4]<=[2,2,3]T(2,1,0)}");
}

class HloTiledShardingWithMetadataTest
    : public ::testing::TestWithParam<ShardingWithMetadataParamType> {};

TEST_P(HloTiledShardingWithMetadataTest, ToStringTest) {
  HloSharding sharding = HloSharding::Tile(
      Array3D<int64_t>({{{2, 3}}, {{5, 7}}}), std::get<0>(GetParam()));
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false),
            "{devices=[2,1,2]2,3,5,7}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    ToString, HloTiledShardingWithMetadataTest,
    ::testing::Values(
        std::make_tuple(std::vector<OpMetadata>(), "{devices=[2,1,2]2,3,5,7}"),
        std::make_tuple(SingleMetadata(),
                        "{devices=[2,1,2]2,3,5,7 metadata={op_name=\"a\"}}"),
        std::make_tuple(ListMetadata(),
                        "{devices=[2,1,2]2,3,5,7 metadata={{op_name=\"b\"}, "
                        "{op_name=\"c\"}}}")));

TEST_F(HloShardingTest, ToStringTupleTest) {
  HloSharding sharding = HloSharding::Tuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                 ShapeUtil::MakeShape(U32, {7, 25}),
                                 ShapeUtil::MakeShape(S32, {9, 11})}),
      {HloSharding::Replicate(), HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
       HloSharding::AssignDevice(3)});
  EXPECT_EQ(sharding.ToString(),
            "{{replicated}, {devices=[1,2]3,5}, {maximal device=3}}");
}

TEST_F(HloShardingTest, ToStringTupleWithMetadataTest) {
  auto metadata = SingleMetadata();
  HloSharding sharding = HloSharding::Tuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                 ShapeUtil::MakeShape(U32, {7, 25}),
                                 ShapeUtil::MakeShape(S32, {9, 11})}),
      {HloSharding::Replicate({GetMetadata("d")}),
       HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
       HloSharding::AssignDevice(3, {GetMetadata("e")})});
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/false),
            "{{replicated}, {devices=[1,2]3,5}, {maximal device=3}}");
  EXPECT_EQ(sharding.ToString(/*include_metadata=*/true),
            "{{replicated metadata={op_name=\"d\"}}, {devices=[1,2]3,5}, "
            "{maximal device=3 metadata={op_name=\"e\"}}}");
}

TEST_F(HloShardingTest, ToStringWithNamedShardingTest) {
  Mesh mesh({2, 4}, {"a", "b"});
  HloSharding sharding(test_utils::FromAxisNames(mesh, {{"a"}, {"b"}}));
  EXPECT_EQ(sharding.ToString(), "{mesh[a=2,b=4], [{a}, {b}]}");

  HloSharding sharding_with_metadata(test_utils::FromAxisNames(
      mesh, {{"a"}, {"b"}}, {}, {}, {}, ListMetadata()));
  EXPECT_EQ(sharding_with_metadata.ToString(/*include_metadata=*/true),
            "{mesh[a=2,b=4], [{a}, {b}], metadata={{op_name=\"b\"}, "
            "{op_name=\"c\"}}}");

  HloSharding tuple_sharding(HloSharding::Tuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                 ShapeUtil::MakeShape(U32, {7, 25}),
                                 ShapeUtil::MakeShape(S32, {9, 11})}),
      {sharding, sharding, sharding_with_metadata}));
  EXPECT_EQ(tuple_sharding.ToString(/*include_metadata=*/true),
            "{{mesh[a=2,b=4], [{a}, {b}]}, {mesh[a=2,b=4], [{a}, {b}]}, "
            "{mesh[a=2,b=4], [{a}, {b}], metadata={{op_name=\"b\"}, "
            "{op_name=\"c\"}}}}");
}

TEST_F(HloShardingTest, OstreamTest) {
  HloSharding sharding =
      HloSharding::Tile(Array4D<int64_t>({{{{0, 1}, {2, 3}}}}));
  std::ostringstream oss;
  oss << sharding;
  EXPECT_EQ(oss.str(), "{devices=[1,1,2,2]0,1,2,3}");
}

class HloParseShardingWithMetadataTest
    : public ::testing::TestWithParam<std::vector<OpMetadata>> {};

TEST_P(HloParseShardingWithMetadataTest, ParseHloString) {
  auto check = [](const HloSharding& sharding) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto parsed_sharding,
        ParseSharding(sharding.ToString(/*include_metadata=*/true)));
    EXPECT_EQ(sharding, parsed_sharding);
  };
  check(HloSharding::Replicate(GetParam()));
  check(HloSharding::AssignDevice(2, GetParam()));
  check(HloSharding::Tile(Array4D<int64_t>({{{{0}, {1}}}}), GetParam()));
  // Empty tuple. One sharding is required for empty tuples, as we need to be
  // able to assign sharding to them, even though they have no leaves.
  check(HloSharding::Tuple(ShapeUtil::MakeTupleShape({}),
                           {HloSharding::Replicate(GetParam())}));
  {
    // Non-nested tuple.
    auto tuple_shape =
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 1, 5, 7}),
                                   ShapeUtil::MakeShape(F32, {3, 5, 7}),
                                   ShapeUtil::MakeShape(F32, {3, 7})});
    check(HloSharding::Tuple(
        tuple_shape,
        {HloSharding::Tile(Array4D<int64_t>({{{{0}, {1}}}})),
         HloSharding::Replicate(GetParam()), HloSharding::AssignDevice(1)}));
  }
  {
    // Nested tuple.
    auto tuple_shape = ShapeUtil::MakeTupleShape(
        {ShapeUtil::MakeShape(F32, {3, 1, 5, 7}),
         ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5, 7}),
                                    ShapeUtil::MakeShape(F32, {3, 7})})});
    std::vector<HloSharding> leaf_shardings = {
        HloSharding::Tile(Array4D<int64_t>({{{{0}, {1}}}})),
        HloSharding::Replicate(), HloSharding::AssignDevice(1, GetParam())};
    ShapeTree<HloSharding> sharding_tree(tuple_shape, HloSharding::Replicate());
    // Assign leaf_shardings to sharding_tree leaves.
    auto it = leaf_shardings.begin();
    for (auto& index_to_sharding : sharding_tree.leaves()) {
      index_to_sharding.second = *it++;
    }
    check(HloSharding::Tuple(sharding_tree));
  }
}

INSTANTIATE_TEST_SUITE_P(ParseHloString, HloParseShardingWithMetadataTest,
                         ::testing::Values(std::vector<OpMetadata>(),
                                           SingleMetadata(), ListMetadata()));

TEST_F(HloShardingTest, WithMetadataNoOverwrite) {
  {
    HloSharding sharding = HloSharding::Replicate();
    auto sharding_new_metadata =
        sharding.WithMetadata(SingleMetadata(), /*overwrite=*/false);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_THAT(sharding_new_metadata.metadata().front(),
                EqualsProto(SingleMetadata().front()));
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/false);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_THAT(sharding_new_metadata.metadata().front(),
                EqualsProto(sharding.metadata().front()));
  }

  {
    HloSharding sharding = HloSharding::Tuple(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                   ShapeUtil::MakeShape(U32, {7, 25}),
                                   ShapeUtil::MakeShape(S32, {9, 11})}),
        {HloSharding::Replicate(SingleMetadata()),
         HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
         HloSharding::AssignDevice(3, SingleMetadata())});
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/false);
    EXPECT_TRUE(sharding_new_metadata.metadata().empty());
    ASSERT_TRUE(sharding_new_metadata.IsTuple());
    ASSERT_EQ(sharding_new_metadata.tuple_elements().size(), 3);

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[0].metadata().size(), 1);
    EXPECT_THAT(sharding_new_metadata.tuple_elements()[0].metadata().front(),
                EqualsProto(SingleMetadata().front()));

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[1].metadata().size(), 2);
    for (int i = 0; i < 2; ++i) {
      EXPECT_THAT(sharding_new_metadata.tuple_elements()[1].metadata()[i],
                  EqualsProto(ListMetadata()[i]));
    }

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[2].metadata().size(), 1);
    EXPECT_THAT(sharding_new_metadata.tuple_elements()[2].metadata().front(),
                EqualsProto(SingleMetadata().front()));
  }
}

TEST_F(HloShardingTest, WithMetadataOverwrite) {
  {
    HloSharding sharding = HloSharding::Replicate();
    auto sharding_new_metadata =
        sharding.WithMetadata(SingleMetadata(), /*overwrite=*/true);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_THAT(sharding_new_metadata.metadata().front(),
                EqualsProto(SingleMetadata().front()));
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/true);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 2);
    for (int i = 0; i < 2; ++i) {
      EXPECT_THAT(sharding_new_metadata.metadata()[i],
                  EqualsProto(ListMetadata()[i]));
    }
  }

  {
    HloSharding sharding = HloSharding::Tuple(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                   ShapeUtil::MakeShape(U32, {7, 25}),
                                   ShapeUtil::MakeShape(S32, {9, 11})}),
        {HloSharding::Replicate(SingleMetadata()),
         HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
         HloSharding::AssignDevice(3, SingleMetadata())});
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/true);
    EXPECT_TRUE(sharding_new_metadata.metadata().empty());
    ASSERT_TRUE(sharding_new_metadata.IsTuple());
    ASSERT_EQ(sharding_new_metadata.tuple_elements().size(), 3);

    for (const auto& sub_sharding : sharding_new_metadata.tuple_elements()) {
      ASSERT_EQ(sub_sharding.metadata().size(), 2);
      for (int i = 0; i < 2; ++i) {
        EXPECT_THAT(sub_sharding.metadata()[i], EqualsProto(ListMetadata()[i]));
      }
    }
  }
}

TEST_F(HloShardingTest, WithoutMetadata) {
  {
    HloSharding sharding = HloSharding::Replicate();
    auto sharding_no_metadata = sharding.WithoutMetadata();
    EXPECT_TRUE(sharding_no_metadata.metadata().empty());
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_no_metadata = sharding.WithoutMetadata();
    EXPECT_TRUE(sharding_no_metadata.metadata().empty());
  }

  {
    HloSharding sharding = HloSharding::Tuple(
        ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                   ShapeUtil::MakeShape(U32, {7, 25}),
                                   ShapeUtil::MakeShape(S32, {9, 11})}),
        {HloSharding::Replicate(SingleMetadata()),
         HloSharding::Tile(Array2D<int64_t>({{3, 5}})),
         HloSharding::AssignDevice(3, ListMetadata())});
    auto sharding_no_metadata = sharding.WithoutMetadata();
    EXPECT_TRUE(sharding_no_metadata.metadata().empty());
    ASSERT_TRUE(sharding_no_metadata.IsTuple());
    EXPECT_EQ(sharding_no_metadata.tuple_elements().size(), 3);
    for (const auto& sub_sharding : sharding_no_metadata.tuple_elements()) {
      EXPECT_TRUE(sub_sharding.metadata().empty());
    }
  }
}

TEST(V3ToV2Sharding, Replicated) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});
  NamedSharding ns(mesh);
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns), HloSharding::Replicate());
}

TEST(V3ToV2Sharding, Maximal) {
  Mesh mesh(5);
  NamedSharding ns(mesh);
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns), HloSharding::AssignDevice(5));
}

TEST(V3ToV2Sharding, SimpleIotaTile) {
  Mesh mesh({16}, {"a"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{"a"}, {}});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns), HloSharding::IotaTile({16, 1}));
}

TEST(V3ToV2Sharding, MeshWithIotaReshapeTranspose) {
  Mesh mesh(TileAssignment({2, 4}, {4, 2}, {1, 0}), {"a", "b"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{"b"}, {"a"}});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns), HloSharding::IotaTile({4, 2}));
}

TEST(V3ToV2Sharding, ResultantMeshWithIotaReshapeTranspose) {
  Mesh mesh(TileAssignment({2, 4}, {4, 2}, {1, 0}), {"a", "b"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{"a"}, {"b"}});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::IotaTile({2, 4}, {4, 2}, {1, 0}));
}

TEST(V3ToV2Sharding, MeshWithDeviceList) {
  Mesh mesh(Array<int64_t>({2, 2, 2}, {0, 2, 4, 6, 1, 3, 5, 7}),
            {"a", "b", "c"});
  NamedSharding ns =
      test_utils::FromAxisNames(mesh, {{"c"}, {"a", "b"}}, {}, {});
  EXPECT_EQ(
      HloSharding::V3ToV2Sharding(ns),
      HloSharding::Tile(Array<int64_t>({2, 4}, {0, 4, 1, 5, 2, 6, 3, 7})));
}

TEST(V3ToV2Sharding, PartialTile) {
  Mesh mesh({2, 4, 4}, {"a", "b", "c"});
  NamedSharding ns =
      test_utils::FromAxisNames(mesh, {{}, {"a"}}, /*replicated_axes=*/{"c"});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::PartialTile(TileAssignment({1, 2, 16})));
}

TEST(V3ToV2Sharding, TransposedIota1) {
  Mesh mesh({2, 4, 4}, {"a", "b", "c"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{"c"}, {"a", "b"}});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::IotaTile({4, 8}, {8, 4}, {1, 0}));
}

TEST(V3ToV2Sharding, TransposedIota2) {
  Mesh mesh({2, 4, 4}, {"a", "b", "c"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{"b"}, {"a", "c"}});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::IotaTile({4, 8}, {2, 4, 4}, {1, 0, 2}));
}

TEST(V3ToV2Sharding, PartialWithTranspose) {
  Mesh mesh({2, 4, 4}, {"a", "b", "c"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{}, {"a", "c"}},
                                               /*replicated_axes=*/{"b"});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::PartialTile(
                TileAssignment({1, 8, 4}, {2, 4, 4}, {0, 2, 1})));
}

TEST(V3ToV2Sharding, Unreduced) {
  Mesh mesh({2, 2}, {"a", "b"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{"a"}, {}}, {},
                                               /*unreduced_axes=*/{"b"});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::Subgroup(TileAssignment({2, 1, 2}),
                                  {OpSharding::UNREDUCED}));
}

TEST(V3ToV2Sharding, Manual) {
  Mesh mesh({2, 2}, {"a", "b"});
  NamedSharding ns = test_utils::FromAxisNames(mesh, {{"a"}, {}}, {}, {},
                                               /*manual_axes=*/{"b"});
  EXPECT_EQ(
      HloSharding::V3ToV2Sharding(ns),
      HloSharding::Subgroup(TileAssignment({2, 1, 2}), {OpSharding::MANUAL}));
}

TEST(V3ToV2Sharding, MultipleSubgroups) {
  Mesh mesh({2, 3, 4, 5}, {"a", "b", "c", "d"});
  NamedSharding ns =
      test_utils::FromAxisNames(mesh, {{"a"}}, {"d"}, {"c"}, {"b"});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::Subgroup(TileAssignment({2, 3, 4, 5}),
                                  {OpSharding::MANUAL, OpSharding::UNREDUCED,
                                   OpSharding::REPLICATED}));
}

class V3ToV2ShardingSplitAxesTest : public ::testing::Test {
 protected:
  Mesh mesh_{{16, 4}, {"a", "b"}};
  AxisRef a12_{0, {1, 2}};
  AxisRef a24_{0, {2, 4}};
  AxisRef a82_{0, {8, 2}};
  AxisRef b_{1};
};

TEST_F(V3ToV2ShardingSplitAxesTest, SplitAxes1) {
  NamedSharding::DimensionSharding ds({b_, a12_}, /*is_closed=*/true);
  NamedSharding ns(mesh_, {ds});
  EXPECT_EQ(
      HloSharding::V3ToV2Sharding(ns),
      HloSharding::PartialTile(TileAssignment({8, 8}, {2, 8, 4}, {2, 0, 1})));
}

TEST_F(V3ToV2ShardingSplitAxesTest, SplitAxes2) {
  NamedSharding::DimensionSharding ds({b_, a24_}, /*is_closed=*/true);
  NamedSharding ns(mesh_, {ds});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::PartialTile(
                TileAssignment({16, 4}, {2, 4, 2, 4}, {3, 1, 0, 2})));
}

TEST_F(V3ToV2ShardingSplitAxesTest, SplitAxes3) {
  NamedSharding::DimensionSharding ds_a({a12_, a82_}, /*is_closed=*/true);
  NamedSharding::DimensionSharding ds_b({a24_, b_}, /*is_closed=*/true);
  NamedSharding ns(mesh_, {ds_a, ds_b});
  EXPECT_EQ(HloSharding::V3ToV2Sharding(ns),
            HloSharding::IotaTile({4, 16}, {2, 4, 2, 4}, {0, 2, 1, 3}));
}

TEST_F(V3ToV2ShardingSplitAxesTest, AllSubgroupTypesWithSplitAxes) {
  NamedSharding::DimensionSharding ds_empty;
  NamedSharding ns(mesh_, {ds_empty, ds_empty}, {a24_}, {a12_}, {a82_, b_});
  EXPECT_EQ(
      HloSharding::V3ToV2Sharding(ns),
      HloSharding::Subgroup(
          TileAssignment({8, 2, 4}, {2, 4, 2, 4}, {2, 3, 0, 1}),
          {OpSharding::MANUAL, OpSharding::UNREDUCED, OpSharding::REPLICATED}));
}

TEST_F(HloShardingTest, ToNamedShardingTuple) {
  HloSharding sharding = HloSharding::Tuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                 ShapeUtil::MakeShape(F32, {2, 3})}),
      {HloSharding::Replicate(), HloSharding::IotaTile({2, 3})});

  HloSharding named_sharding = HloSharding::ToV3Sharding(sharding);

  ASSERT_TRUE(named_sharding.IsTuple());
  ASSERT_EQ(named_sharding.tuple_elements().size(), 2);
  EXPECT_TRUE(named_sharding.tuple_elements()[0].UseNamedShardingLeaf());
  EXPECT_EQ(named_sharding.tuple_elements()[0].named_sharding(),
            NamedSharding::Replicate());
  EXPECT_TRUE(named_sharding.tuple_elements()[1].UseNamedShardingLeaf());
  EXPECT_EQ(named_sharding.tuple_elements()[1].named_sharding().ToString(),
            "{mesh[axis_0=2,axis_1=3], [{axis_0}, {axis_1}]}");
}

TEST_F(HloShardingTest, ToNamedShardingReplicated) {
  HloSharding hlo_sharding = HloSharding::Replicate();
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_TRUE(named_sharding.IsReplicated());
}

TEST_F(HloShardingTest, ToNamedShardingMaximal) {
  HloSharding hlo_sharding = HloSharding::AssignDevice(5);
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_TRUE(named_sharding.IsMaximal());
  EXPECT_EQ(*named_sharding.mesh().device_assignment().array().begin(), 5);
}

TEST_F(HloShardingTest, ToNamedShardingTiled) {
  HloSharding hlo_sharding = HloSharding::IotaTile({2, 3});
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding.ToString(),
            "{mesh[axis_0=2,axis_1=3], [{axis_0}, {axis_1}]}");
}

TEST_F(HloShardingTest, ToNamedShardingPartialTile) {
  HloSharding hlo_sharding = HloSharding::PartialTile(TileAssignment({2, 3}));
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding.ToString(),
            "{mesh[axis_0=2,axis_1=3], [{axis_0}], replicated={axis_1}}");
}

TEST_F(HloShardingTest, ToNamedShardingIotaWithReshape) {
  HloSharding hlo_sharding = HloSharding::IotaTile({2, 4}, {8}, {0});
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding.ToString(),
            "{mesh[axis_0=2,axis_1=4], [{axis_0}, {axis_1}]}");
}

TEST_F(HloShardingTest, ToNamedShardingIotaWithReshapeTransposeToSingleDim) {
  HloSharding hlo_sharding = HloSharding::IotaTile({4}, {2, 2}, {1, 0});
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding.ToString(),
            "{mesh[axis_0=2,axis_1=2], [{axis_1, axis_0}]}");
}

TEST_F(HloShardingTest, ToNamedShardingIotaWithReshapeAndTranspose) {
  HloSharding hlo_sharding = HloSharding::IotaTile({2, 2}, {2, 2}, {1, 0});
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding.ToString(),
            "{mesh[axis_0=2,axis_1=2], [{axis_1}, {axis_0}]}");
}

TEST_F(HloShardingTest, ToNamedShardingIotaWithReshapeTransposeToTwoDims) {
  HloSharding hlo_sharding =
      HloSharding::IotaTile({6, 35}, {7, 10, 3}, {2, 1, 0});
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding.ToString(),
            "{mesh[axis_0=7,axis_1=2,axis_2=5,axis_3=3], [{axis_3, axis_1}, "
            "{axis_2, axis_0}]}");
}

TEST_F(HloShardingTest, ToNamedShardingSubgroups) {
  HloSharding hlo_sharding = HloSharding::Subgroup(
      TileAssignment({2, 2, 2}),
      {OpSharding::MANUAL, OpSharding::UNREDUCED, OpSharding::REPLICATED});
  NamedSharding named_sharding = HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding.ToString(),
            "{mesh[axis_0=2,axis_1=2,axis_2=2], [], replicated={axis_2}, "
            "unreduced={axis_1}, manual={axis_0}}");
}

class HloShardingV2ToV3ToV2RoundTripTest
    : public HloShardingTest,
      public ::testing::WithParamInterface<HloSharding> {
 public:
  HloSharding V3ToV2Deep(const HloSharding& s) {
    if (s.IsTuple()) {
      std::vector<HloSharding> elements;
      for (const auto& e : s.tuple_elements()) {
        elements.push_back(V3ToV2Deep(e));
      }
      return HloSharding::FlatTuple(elements);
    }
    return HloSharding::V3ToV2Sharding(s.named_sharding());
  }
};

TEST_P(HloShardingV2ToV3ToV2RoundTripTest, RoundTrip) {
  const HloSharding& hlo_sharding = GetParam();
  HloSharding named_sharding = HloSharding::ToV3Sharding(hlo_sharding);
  HloSharding hlo_sharding_restored = V3ToV2Deep(named_sharding);
  EXPECT_EQ(hlo_sharding, hlo_sharding_restored);
}

INSTANTIATE_TEST_SUITE_P(
    V2ToV3ToV2RoundTrip, HloShardingV2ToV3ToV2RoundTripTest,
    ::testing::Values(
        HloSharding::Replicate(), HloSharding::AssignDevice(3),
        HloSharding::IotaTile({2, 4}), HloSharding::IotaTile({2, 4}, {8}, {0}),
        HloSharding::Subgroup(TileAssignment({2, 2}),
                              {OpSharding::MANUAL, OpSharding::REPLICATED}),
        HloSharding::Tuple(
            ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {3, 5}),
                                       ShapeUtil::MakeShape(F32, {2, 4})}),
            {HloSharding::Replicate(), HloSharding::IotaTile({2, 4})})));

class HloShardingV3ToV2ToV3RoundTripTest
    : public HloShardingTest,
      public ::testing::WithParamInterface<NamedSharding> {};

TEST_P(HloShardingV3ToV2ToV3RoundTripTest, RoundTrip) {
  const NamedSharding& named_sharding = GetParam();

  HloSharding hlo_sharding = HloSharding::V3ToV2Sharding(named_sharding);
  NamedSharding named_sharding_restored =
      HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding, named_sharding_restored);
}

INSTANTIATE_TEST_SUITE_P(
    V3ToV2ToV3RoundTrip, HloShardingV3ToV2ToV3RoundTripTest,
    ::testing::Values(
        NamedSharding::Replicate(), NamedSharding::MaximalSharding(3),
        test_utils::FromAxisNames(Mesh({2, 3}, {"axis_0", "axis_1"}),
                                  {{"axis_0"}, {"axis_1"}}),
        test_utils::FromAxisNames(Mesh({2, 2}, {"axis_0", "axis_1"}),
                                  {{"axis_0"}, {"axis_1"}}),
        test_utils::FromAxisNames(Mesh({4}, {"axis_0"}), {{"axis_0"}}),
        test_utils::FromAxisNames(Mesh(Array<int64_t>({2, 2}, {0, 2, 1, 3}),
                                       {"axis_0", "axis_1"}),
                                  {{"axis_0"}, {"axis_1"}})));

TEST_F(HloShardingTest, V3ToV2ToV3RoundTripSubAxes) {
  Mesh mesh({4}, {"axis_0"});
  NamedSharding named_sharding =
      test_utils::FromAxisNames(mesh, {{"axis_0:(1)2"}});

  HloSharding hlo_sharding = HloSharding::V3ToV2Sharding(named_sharding);
  NamedSharding named_sharding_restored =
      HloSharding::ToNamedSharding(hlo_sharding);

  EXPECT_EQ(named_sharding_restored.dim_shardings().size(), 1);
  EXPECT_EQ(named_sharding_restored.dim_sharding(0).getShardedSize(
                named_sharding_restored.mesh()),
            2);

  HloSharding hlo_sharding_restored =
      HloSharding::V3ToV2Sharding(named_sharding_restored);
  EXPECT_EQ(hlo_sharding, hlo_sharding_restored);
}

}  // namespace
}  // namespace xla
