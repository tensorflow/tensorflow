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
#include "tensorflow/compiler/tf2xla/sharding_util.h"

#include <functional>
#include <string>

#include <gmock/gmock.h>
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(CoreUtilTest, ParseShardingFromDevice) {
  Graph graph(OpRegistry::Global());

  auto core_from_sharding =
      [](std::optional<xla::OpSharding> sharding) -> int64 {
    if (sharding.has_value() &&
        sharding.value().type() == xla::OpSharding::MAXIMAL) {
      return sharding.value().tile_assignment_devices(0);
    } else {
      return -1;
    }
  };

  auto parse_status = ParseShardingFromDevice("", 1);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.value()));
  parse_status = ParseShardingFromDevice("", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.value()));

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:-1", 100);
  EXPECT_FALSE(parse_status.ok());

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:55", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(55, core_from_sharding(parse_status.value()));

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:100", 100);
  EXPECT_FALSE(parse_status.ok());

  parse_status = ParseShardingFromDevice("/cpu:0", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.value()));
}

struct ShardingParams {
  std::string op_name;
  std::string primary_sharding_attr;
};

// Tests GetShardingFromNodeDef for getting the sharding from the expected
// attribute.
class ShardingMetadataAttributesTest
    : public ::testing::TestWithParam<ShardingParams> {};

TEST_P(ShardingMetadataAttributesTest, GetShardingFromNodeDef) {
  const ShardingParams& params = GetParam();

  AttrValue xla_sharding_v1;
  xla_sharding_v1.set_s("{devices=[2,1]0,1}");
  AttrValue xla_sharding_v2;
  xla_sharding_v2.set_s("{devices=[2,1]<=[2]}");
  AttrValue xla_sharding_v1_diff;
  xla_sharding_v1_diff.set_s("{devices=[2,1]1,0}");
  AttrValue index;
  index.set_i(0);
  AttrValue type;
  type.set_type(DataType::DT_FLOAT);

  NodeDef node1;
  node1.set_op(params.op_name);
  node1.set_name("with_sharding");
  node1.mutable_attr()->insert({{params.primary_sharding_attr, xla_sharding_v1},
                                {"index", index},
                                {"T", type}});
  auto sharding1 = GetShardingFromNodeDef(node1, /*add_metadata=*/false);
  TF_ASSERT_OK(sharding1.status());
  EXPECT_TRUE(sharding1->has_value());
  EXPECT_EQ(sharding1->value().tile_assignment_devices_size(), 2);
  EXPECT_EQ(sharding1->value().tile_assignment_devices(0), 0);
  EXPECT_EQ(sharding1->value().tile_assignment_devices(1), 1);

  NodeDef node2;
  node2.set_op(params.op_name);
  node2.set_name("with_sharding_and_XlaShardingV2_consistent");
  node2.mutable_attr()->insert({{params.primary_sharding_attr, xla_sharding_v1},
                                {"_XlaShardingV2", xla_sharding_v2},
                                {"index", index},
                                {"T", type}});
  auto sharding2 = GetShardingFromNodeDef(node2, /*add_metadata=*/false);
  TF_ASSERT_OK(sharding2.status());
  EXPECT_TRUE(sharding2->has_value());
  EXPECT_EQ(sharding2->value().tile_assignment_devices_size(), 0);

  NodeDef node3;
  node3.set_op(params.op_name);
  node3.set_name("with_sharding_and_XlaShardingV2_inconsistent");
  node3.mutable_attr()->insert(
      {{params.primary_sharding_attr, xla_sharding_v1_diff},
       {"_XlaShardingV2", xla_sharding_v2},
       {"index", index},
       {"T", type}});
  auto sharding3 = GetShardingFromNodeDef(node3, /*add_metadata=*/false);
  EXPECT_FALSE(sharding3.status().ok());
  EXPECT_THAT(
      sharding3.status().message(),
      ::testing::HasSubstr(
          "Sharding attribute was not equivalent to XlaShardingV2 attribute"));
}

INSTANTIATE_TEST_SUITE_P(ShardingMetadataAttributesTestCases,
                         ShardingMetadataAttributesTest,
                         ::testing::ValuesIn<ShardingParams>({
                             {"XlaSharding", "sharding"},
                             {"_Arg", "_XlaSharding"},
                         }),
                         [](const ::testing::TestParamInfo<
                             ShardingMetadataAttributesTest::ParamType>& info) {
                           return info.param.op_name;
                         });

class ShardingWithMetadataTest
    : public ::testing::TestWithParam<xla::OpSharding> {};

TEST_P(ShardingWithMetadataTest, GetShardingFromNode) {
  NodeDef node_def;
  {
    node_def.set_op("_Arg");
    node_def.set_name("arg");
    AttrValue xla_sharding;
    xla_sharding.set_s("");
    AttrValue index;
    index.set_i(0);
    AttrValue type;
    type.set_type(DataType::DT_FLOAT);
    node_def.mutable_attr()->insert(
        {{"_XlaSharding", xla_sharding}, {"index", index}, {"T", type}});
  }

  auto check_metadata = [](const xla::OpSharding& sharding) {
    ASSERT_EQ(sharding.metadata_size(), 1);
    const auto& metadata = sharding.metadata(0);
    EXPECT_EQ(metadata.op_type(), "_Arg");
    EXPECT_EQ(metadata.op_name(), "arg");
  };

  auto test_sharding_metadata =
      [&check_metadata](
          const std::function<absl::StatusOr<std::optional<xla::OpSharding>>()>&
              fn) {
        auto status_or_sharding = fn();
        TF_ASSERT_OK(status_or_sharding.status());
        ASSERT_TRUE(status_or_sharding.value().has_value());
        auto& sharding = status_or_sharding.value();
        ASSERT_TRUE(sharding.has_value());
        if (sharding->type() == xla::OpSharding::TUPLE) {
          EXPECT_TRUE(sharding->metadata().empty());
          for (const auto& sharding_element : sharding->tuple_shardings()) {
            check_metadata(sharding_element);
          }
        } else {
          check_metadata(sharding.value());
        }
      };

  {
    test_sharding_metadata([&node_def]() {
      return GetShardingFromNodeDef(node_def, /*add_metadata=*/true);
    });
  }

  {
    test_sharding_metadata([&node_def]() {
      return ParseShardingFromDevice(node_def, /*num_cores_per_replica=*/1,
                                     /*add_metadata=*/true);
    });
  }

  {
    Graph graph(OpRegistry::Global());
    absl::Status status;
    Node* node = graph.AddNode(node_def, &status);
    TF_ASSERT_OK(status);

    test_sharding_metadata([node]() {
      return ParseShardingFromDevice(*node, /*num_cores_per_replica=*/1,
                                     /*add_metadata=*/true);
    });
  }
}

xla::OpSharding CreateTupleSharding() {
  xla::OpSharding sharding;
  sharding.set_type(xla::OpSharding::TUPLE);
  sharding.add_tuple_shardings()->set_type(xla::OpSharding::REPLICATED);
  sharding.add_tuple_shardings()->set_type(xla::OpSharding::REPLICATED);
  return sharding;
}

INSTANTIATE_TEST_SUITE_P(GetShardingFromNode, ShardingWithMetadataTest,
                         ::testing::Values(xla::sharding_builder::Replicate(),
                                           CreateTupleSharding()));

TEST(AddSdyShardingFrontendAttributeTest, NoSharding) {
  xla::XlaBuilder builder("test_builder");
  xla::XlaOp param = xla::Parameter(
      &builder, 0, xla::ShapeUtil::MakeShape(xla::F32, {}), "p0");
  EXPECT_FALSE(builder.sharding().has_value());
  TF_EXPECT_OK(addSdyShardingFrontendAttribute(
      &builder, param, xla::ShapeUtil::MakeShape(xla::F32, {})));
}

TEST(AddSdyShardingFrontendAttributeTest, WithSharding) {
  xla::XlaBuilder builder("test_builder");
  xla::OpSharding opSharding;
  opSharding.set_type(xla::OpSharding::OTHER);
  opSharding.add_tile_assignment_dimensions(2);
  opSharding.add_tile_assignment_dimensions(4);
  opSharding.add_iota_reshape_dims(8);
  opSharding.add_iota_transpose_perm(0);
  xla::XlaScopedShardingAssignment assign_sharding(&builder, opSharding);

  xla::XlaOp param = xla::Parameter(
      &builder, 0, xla::ShapeUtil::MakeShape(xla::F32, {2, 3}), "p0");
  TF_EXPECT_OK(addSdyShardingFrontendAttribute(
      &builder, param, xla::ShapeUtil::MakeShape(xla::F32, {2, 3}),
      /*is_single_arg=*/true));

  TF_ASSERT_OK_AND_ASSIGN(xla::XlaComputation computation, builder.Build());
  const xla::HloModuleProto& hloModuleProto = computation.proto();

  const xla::HloInstructionProto& instruction =
      hloModuleProto.computations(0).instructions(0);

  EXPECT_TRUE(instruction.frontend_attributes().map().contains(
      xla::HloSharding::kShardingFrontendAttrName));
  EXPECT_EQ(instruction.frontend_attributes().map().at(
                xla::HloSharding::kShardingFrontendAttrName),
            "#sdy.sharding<mesh<[\"_axis_0\"=2, \"_axis_1\"=4]>, "
            "[{\"_axis_0\"}, {\"_axis_1\"}]>");
}

}  // namespace tensorflow
