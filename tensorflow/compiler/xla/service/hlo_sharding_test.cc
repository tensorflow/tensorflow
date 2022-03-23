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

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

Array<int64_t> MakeArray(absl::Span<const int64_t> dimensions,
                         absl::Span<const int64_t> contents) {
  Array<int64_t> a(dimensions);
  std::copy(contents.begin(), contents.end(), a.begin());
  return a;
}

OpMetadata GetMetadata(const std::string& op_name) {
  OpMetadata metadata;
  metadata.set_op_name(op_name);
  return metadata;
}

std::vector<OpMetadata> SingleMetadata() { return {GetMetadata("a")}; }

std::vector<OpMetadata> ListMetadata() {
  return {GetMetadata("b"), GetMetadata("c")};
}

class HloShardingTest : public HloTestBase {};

TEST_F(HloShardingTest, Replicate) {
  HloSharding sharding = HloSharding::Replicate();
  EXPECT_TRUE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_TRUE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(65535));

  HloSharding other = HloSharding::Replicate();
  EXPECT_EQ(other, sharding);

  EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4}),
                                 /*num_devices=*/2));
  EXPECT_FALSE(sharding.HasUniqueDevice());
}

TEST_F(HloShardingTest, DevicePlacement) {
  HloSharding sharding = HloSharding::AssignDevice(5);
  EXPECT_FALSE(sharding.IsReplicated());
  EXPECT_TRUE(sharding.IsTileMaximal());
  EXPECT_FALSE(sharding.UsesDevice(0));
  EXPECT_TRUE(sharding.UsesDevice(5));
  EXPECT_EQ(5, sharding.GetUniqueDevice());

  HloSharding other = HloSharding::Replicate();
  EXPECT_NE(other, sharding);

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
  OpSharding proto;
  proto.set_type(OpSharding::TUPLE);
  auto* tiled = proto.add_tuple_shardings();
  tiled->set_type(OpSharding::OTHER);
  tiled->add_tile_assignment_devices(0);
  tiled->add_tile_assignment_devices(1);
  tiled->add_tile_assignment_dimensions(1);
  tiled->add_tile_assignment_dimensions(2);
  *tiled->add_metadata() = GetMetadata("a");
  *tiled->add_metadata() = GetMetadata("b");
  auto* replicated = proto.add_tuple_shardings();
  replicated->set_type(OpSharding::REPLICATED);
  *replicated->add_metadata() = GetMetadata("c");
  auto* manual = proto.add_tuple_shardings();
  manual->set_type(OpSharding::MANUAL);
  HloSharding sharding = HloSharding::FromProto(proto).ConsumeValueOrDie();
  EXPECT_TRUE(protobuf_util::ProtobufEquals(proto, sharding.ToProto()));
}

TEST_F(HloShardingTest, Tile) {
  {
    // Test should fail because of a duplicate tile assignment.
    HloSharding sharding = HloSharding::Tile(MakeArray({2, 2}, {0, 0, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {4, 6}),
                                       /*num_devices=*/4));
  }

  {
    // Test should fail because of more devices used than `num_device`.
    HloSharding sharding = HloSharding::Tile(MakeArray({2, 2}, {0, 1, 2, 3}));
    EXPECT_IS_NOT_OK(sharding.Validate(ShapeUtil::MakeShape(U32, {4, 6}),
                                       /*num_devices=*/2));
  }

  {
    // Test should pass.
    Shape shape = ShapeUtil::MakeShape(U32, {4, 5});
    HloSharding sharding = HloSharding::Tile(MakeArray({2, 2}, {0, 3, 2, 1}));
    EXPECT_IS_OK(sharding.Validate(ShapeUtil::MakeShape(F32, {3, 5}),
                                   /*num_devices=*/5));

    EXPECT_EQ(0, sharding.DeviceForTileIndex({0, 0}));
    EXPECT_EQ(3, sharding.DeviceForTileIndex({0, 1}));
    EXPECT_EQ(2, sharding.DeviceForTileIndex({1, 0}));
    EXPECT_EQ(1, sharding.DeviceForTileIndex({1, 1}));

    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 0),
              (std::vector<int64_t>{0, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 3),
              (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 2),
              (std::vector<int64_t>{2, 0}));
    EXPECT_EQ(sharding.TileOffsetForDevice(shape, 1),
              (std::vector<int64_t>{2, 3}));

    EXPECT_FALSE(sharding.HasUniqueDevice());
  }
}

// Tests that empty tuple is supported.
TEST_F(HloShardingTest, EmptySingleTuple) {
  HloSharding sharding = HloSharding::SingleTuple(ShapeUtil::MakeTupleShape({}),
                                                  HloSharding::AssignDevice(0));
  EXPECT_TRUE(sharding.ExtractSingleSharding());
}

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
  HloSharding tuple_sharding =
      HloSharding::FromProto(proto).ConsumeValueOrDie();

  ShapeTree<HloSharding> shape_tree =
      tuple_sharding.GetAsShapeTree(nested_tuple_shape);
  EXPECT_EQ(shape_tree.element({0}), HloSharding::Replicate());
  EXPECT_EQ(shape_tree.element({1, 0}), HloSharding::AssignDevice(0));
  EXPECT_EQ(shape_tree.element({2}), tiled_sharding);

  EXPECT_IS_OK(tuple_sharding.Validate(nested_tuple_shape, /*num_devices=*/5));
  // Test should fail because tuple element count does not match.
  EXPECT_IS_NOT_OK(tuple_sharding.Validate(ShapeUtil::MakeTupleShape({}),
                                           /*num_devices=*/5));
  // Test should fail because the input type is not a tuple.
  EXPECT_IS_NOT_OK(tuple_sharding.Validate(ShapeUtil::MakeShape(F32, {}),
                                           /*num_devices=*/5));
}

TEST_F(HloShardingTest, NormalizeTrivialSubgroupToManual) {
  HloSharding sharding =
      HloSharding::Subgroup(MakeArray({1, 2, 1}, {0, 1}),
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
    HloSharding sharding1 = HloSharding::Tile(MakeArray({2, 2}, {0, 3, 2, 1}));
    HloSharding sharding2 = HloSharding::Tile(MakeArray({2, 2}, {0, 3, 2, 1}));
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
  HloSharding sharding = HloSharding::Replicate();
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
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.metadata().front(), SingleMetadata().front()));
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/false);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding.metadata().front(), sharding_new_metadata.metadata().front()));
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
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.tuple_elements()[0].metadata().front(),
        SingleMetadata().front()));

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[1].metadata().size(), 2);
    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(protobuf_util::ProtobufEquals(
          sharding_new_metadata.tuple_elements()[1].metadata()[i],
          ListMetadata()[i]));
    }

    ASSERT_EQ(sharding_new_metadata.tuple_elements()[2].metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.tuple_elements()[2].metadata().front(),
        SingleMetadata().front()));
  }
}

TEST_F(HloShardingTest, WithMetadataOverwrite) {
  {
    HloSharding sharding = HloSharding::Replicate();
    auto sharding_new_metadata =
        sharding.WithMetadata(SingleMetadata(), /*overwrite=*/true);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 1);
    EXPECT_TRUE(protobuf_util::ProtobufEquals(
        sharding_new_metadata.metadata().front(), SingleMetadata().front()));
  }

  {
    HloSharding sharding = HloSharding::AssignDevice(7, SingleMetadata());
    auto sharding_new_metadata =
        sharding.WithMetadata(ListMetadata(), /*overwrite=*/true);
    ASSERT_EQ(sharding_new_metadata.metadata().size(), 2);
    for (int i = 0; i < 2; ++i) {
      EXPECT_TRUE(protobuf_util::ProtobufEquals(
          sharding_new_metadata.metadata()[i], ListMetadata()[i]));
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
        EXPECT_TRUE(protobuf_util::ProtobufEquals(sub_sharding.metadata()[i],
                                                  ListMetadata()[i]));
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

}  // namespace
}  // namespace xla
