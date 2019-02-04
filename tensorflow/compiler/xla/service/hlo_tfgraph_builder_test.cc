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

#include "tensorflow/compiler/xla/service/hlo_tfgraph_builder.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace xla {
namespace hlo_graph_dumper {
namespace {

using ::tensorflow::GraphDef;

class HloTfGraphBuilderTest : public HloTestBase {
 protected:
  HloTfGraphBuilderTest() {}
  HloTfGraphBuilder generator_;

  // Create a computation which takes a scalar and returns its negation.
  std::unique_ptr<HloComputation> CreateNegateComputation() {
    auto builder = HloComputation::Builder("Negate");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, r0f32_, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateUnary(r0f32_, HloOpcode::kNegate, param));
    return builder.Build();
  }

  // Creates a computation which calls map with the given computation.
  std::unique_ptr<HloComputation> CreateMapComputation(
      HloComputation *map_computation) {
    auto builder = HloComputation::Builder("Map");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, r0f32_, "param0"));
    builder.AddInstruction(
        HloInstruction::CreateMap(r0f32_, {param}, map_computation));
    return builder.Build();
  }
  Shape r0f32_ = ShapeUtil::MakeShape(PrimitiveType::F32, {});
};

static const tensorflow::AttrValue &GetNodeAttr(const tensorflow::NodeDef &node,
                                                const string &attr_name) {
  auto attr = node.attr().find(attr_name);
  CHECK(attr != node.attr().end());
  return attr->second;
}

TEST_F(HloTfGraphBuilderTest, CheckConcatenateDimsAndShapes) {
  auto builder = HloComputation::Builder("Concatenate");
  Shape shape = ShapeUtil::MakeShape(PrimitiveType::F32, {2, 2});
  auto param_1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  auto param_2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "param1"));
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {param_1, param_2}, 1));
  TF_CHECK_OK(generator_.AddComputation(*builder.Build()));
  GraphDef graph_def = generator_.GetGraphDef();
  EXPECT_EQ(graph_def.node_size(), 3);
  const auto &node = graph_def.node(2);
  EXPECT_EQ(node.name(), "Concatenate/concatenate");

  // Check dimensions.
  auto dims_value = GetNodeAttr(node, "dims");
  EXPECT_EQ(dims_value.list().i_size(), 1);
  EXPECT_EQ(dims_value.list().i(0), 1);

  // Check shapes.
  auto shape_value = GetNodeAttr(node, "_output_shapes");
  EXPECT_EQ(shape_value.list().shape_size(), 1);
  EXPECT_EQ(shape_value.list().shape(0).dim_size(), 2);
  EXPECT_EQ(shape_value.list().shape(0).dim(0).size(), 2);
  EXPECT_EQ(shape_value.list().shape(0).dim(1).size(), 4);
}

TEST_F(HloTfGraphBuilderTest, CheckScalarValue) {
  auto builder = HloComputation::Builder("Const");
  HloInstruction *instruction = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(123)));
  OpMetadata metadata;
  metadata.set_op_name("x");
  metadata.set_op_type("y");
  instruction->set_metadata(metadata);
  TF_CHECK_OK(generator_.AddComputation(*builder.Build()));
  GraphDef graph_def = generator_.GetGraphDef();
  EXPECT_EQ(graph_def.node_size(), 1);
  const auto &node = graph_def.node(0);
  EXPECT_EQ(GetNodeAttr(node, "value").s(), "123");
  EXPECT_EQ(GetNodeAttr(node, "type").s(), "S32");
  EXPECT_EQ(GetNodeAttr(node, "tf_op_name").s(), "x");
  EXPECT_EQ(GetNodeAttr(node, "tf_op_type").s(), "y");
}

TEST_F(HloTfGraphBuilderTest, SimpleNegateComputation) {
  auto negate_computation = CreateNegateComputation();
  TF_CHECK_OK(generator_.AddComputation(*negate_computation));
  GraphDef graph_def = generator_.GetGraphDef();
  EXPECT_EQ(graph_def.node_size(), 2);
  EXPECT_EQ(graph_def.node(0).name(), "Negate/param0.0");
  EXPECT_EQ(graph_def.node(0).op(), "HloParameter");
  EXPECT_EQ(graph_def.node(1).name(), "Negate/negate");
  EXPECT_EQ(graph_def.node(1).op(), "HloNegate");
  EXPECT_EQ(graph_def.node(1).input_size(), 1);
  EXPECT_EQ(graph_def.node(1).input(0), "Negate/param0.0");
}

TEST_F(HloTfGraphBuilderTest, GreaterThanOrEqualTo) {
  auto builder = HloComputation::Builder("GE");
  auto param_1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto param_2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32_, "param1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kGe, param_1, param_2));
  TF_CHECK_OK(generator_.AddComputation(*builder.Build()));
  GraphDef graph_def = generator_.GetGraphDef();
  EXPECT_EQ(graph_def.node_size(), 3);
  EXPECT_EQ(graph_def.node(0).name(), "GE/param0.0");
  EXPECT_EQ(graph_def.node(1).name(), "GE/param1.1");
  EXPECT_EQ(graph_def.node(2).input_size(), 2);
  EXPECT_EQ(graph_def.node(2).name(), "GE/greater-than-or-equal-to");
  EXPECT_EQ(graph_def.node(2).op(), "HloGreaterThanOrEqualTo");
}

TEST_F(HloTfGraphBuilderTest, IncorparateTfOpsStructure) {
  auto builder = HloComputation::Builder("GE");
  auto param_1 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto param_2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, r0f32_, "param1"));
  auto ge = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kGe, param_1, param_2));
  OpMetadata metadata;
  metadata.set_op_name("x/y");
  metadata.set_op_type("Y");
  ge->set_metadata(metadata);
  TF_CHECK_OK(generator_.AddComputation(*builder.Build()));
  GraphDef graph_def = generator_.GetGraphDef();
  EXPECT_EQ(graph_def.node_size(), 3);
  EXPECT_EQ(graph_def.node(0).name(), "GE/param0.0");
  EXPECT_EQ(graph_def.node(1).name(), "GE/param1.1");
  EXPECT_EQ(graph_def.node(2).input_size(), 2);
  EXPECT_EQ(graph_def.node(2).name(), "GE/x/y/greater-than-or-equal-to");
  EXPECT_EQ(graph_def.node(2).op(), "HloGreaterThanOrEqualTo");
}

TEST_F(HloTfGraphBuilderTest, EmbeddedComputationsDiamond) {
  // Create computations with a diamond-shaped callgraph.
  auto negate_computation = CreateNegateComputation();
  auto map1_computation = CreateMapComputation(negate_computation.get());
  auto map2_computation = CreateMapComputation(negate_computation.get());

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32_, "param0"));
  auto map1 = builder.AddInstruction(
      HloInstruction::CreateMap(r0f32_, {param}, map1_computation.get()));
  auto map2 = builder.AddInstruction(
      HloInstruction::CreateMap(r0f32_, {param}, map2_computation.get()));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, map1, map2));
  auto computation = builder.Build();
  TF_CHECK_OK(generator_.AddComputation(*computation));
  EXPECT_GT(generator_.GetGraphDef().node_size(), 0);
}

TEST_F(HloTfGraphBuilderTest, TokenHasNoLayout) {
  auto builder = HloComputation::Builder("Token");
  auto token = builder.AddInstruction(HloInstruction::CreateToken());
  OpMetadata metadata;
  metadata.set_op_name("x");
  metadata.set_op_type("y");
  token->set_metadata(metadata);
  TF_CHECK_OK(generator_.AddComputation(*builder.Build()));
  GraphDef graph_def = generator_.GetGraphDef();
  EXPECT_EQ(graph_def.node_size(), 1);
  const auto &node = graph_def.node(0);
  EXPECT_EQ(GetNodeAttr(node, "type").s(), "TOKEN");
  EXPECT_EQ(GetNodeAttr(node, "layout").s(), "");
  EXPECT_EQ(GetNodeAttr(node, "tf_op_name").s(), "x");
  EXPECT_EQ(GetNodeAttr(node, "tf_op_type").s(), "y");
}

}  // namespace
}  // namespace hlo_graph_dumper
}  // namespace xla
