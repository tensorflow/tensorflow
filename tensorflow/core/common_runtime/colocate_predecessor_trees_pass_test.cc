/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/colocate_predecessor_trees_pass.h"

#include <memory>
#include <string>

#include "tensorflow/cc/framework/scope.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/test.h"

namespace tensorflow {

const char kCpu0[] = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0";
const char kCpu1[] = "/job:tpu_host_worker/replica:0/task:0/device:CPU:1";
const char kClassAttr[] = "_class";

// Return the node with name `name`.
Node* GetNode(const Graph& graph, const std::string& name) {
  for (Node* node : graph.nodes()) {
    if (node->name() == name) return node;
  }
  return nullptr;
}

// Test the pass is skipped by default because flag enable_tf2min_ici_weight is
// false by default.
TEST(ColocatePredecessorTreesPassTest, ICIFlagFalse) {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* const_0 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_0")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(1.0)));
  Node* const_1 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_1")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", const_0, const_1, builder.opts().WithName("fill"));
  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));
  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity_1"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_requested_device(kCpu0);
  GetNode(*graph, "identity_1")->set_requested_device(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "const_0")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "const_1")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "fill")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "identity")->def(), kClassAttr));
}

// Test a simple colocate predecessor tree example.
TEST(ColocatePredecessorTreesPassTest, SimpleExample) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* const_0 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_0")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(1.0)));
  Node* const_1 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_1")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", const_0, const_1, builder.opts().WithName("fill"));
  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));
  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity_1"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_requested_device(kCpu0);
  GetNode(*graph, "identity_1")->set_requested_device(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_0")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_1")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "fill")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "identity")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "identity_1")->def(), kClassAttr));

  std::string expected_colocation_info = "loc:@identity";
  const AttrValue* input_value;
  TF_EXPECT_OK(
      GetNode(*graph, "const_0")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "const_1")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(GetNode(*graph, "fill")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "identity")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "identity_1")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
}

// Test colocate two predecessor trees case.
TEST(ColocatePredecessorTreesPassTest, PropagateTwoTrees) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* const_0 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_0")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(1.0)));
  Node* const_1 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_1")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", const_0, const_1, builder.opts().WithName("fill"));
  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));

  Node* const_2 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_2")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(1.0)));
  Node* const_3 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_3")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill_1 = ops::BinaryOp("Fill", const_2, const_3,
                               builder.opts().WithName("fill_1"));
  ops::UnaryOp("Identity", fill_1, builder.opts().WithName("identity_1"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_requested_device(kCpu0);
  GetNode(*graph, "identity_1")->set_requested_device(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_0")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_1")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "fill")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "identity")->def(), kClassAttr));

  std::string expected_colocation_info = "loc:@identity";
  const AttrValue* input_value;
  TF_EXPECT_OK(
      GetNode(*graph, "const_0")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "const_1")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(GetNode(*graph, "fill")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "identity")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);

  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_2")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_3")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "fill_1")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "identity_1")->def(), kClassAttr));

  std::string expected_colocation_info_1 = "loc:@identity_1";
  TF_EXPECT_OK(
      GetNode(*graph, "const_2")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info_1);
  TF_EXPECT_OK(
      GetNode(*graph, "const_3")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info_1);
  TF_EXPECT_OK(
      GetNode(*graph, "fill_1")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info_1);
  TF_EXPECT_OK(
      GetNode(*graph, "identity_1")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info_1);
}

// Test a simple colocate predecessor tree example.
TEST(ColocatePredecessorTreesPassTest, RootHasMultipleOutputs) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* const_0 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_0")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(1.0)));
  Node* const_1 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_1")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", const_0, const_1, builder.opts().WithName("fill"));
  Node* identity =
      ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));
  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity_0"));
  ops::UnaryOp("Identity", identity, builder.opts().WithName("identity_1"));
  ops::UnaryOp("Identity", identity, builder.opts().WithName("identity_2"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_requested_device(kCpu0);
  GetNode(*graph, "identity_0")->set_requested_device(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_0")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "const_1")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "fill")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "identity")->def(), kClassAttr));
  EXPECT_TRUE(HasNodeAttr(GetNode(*graph, "identity_0")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "identity_1")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "identity_2")->def(), kClassAttr));

  std::string expected_colocation_info = "loc:@identity";
  const AttrValue* input_value;
  TF_EXPECT_OK(
      GetNode(*graph, "const_0")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "const_1")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(GetNode(*graph, "fill")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "identity")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
  TF_EXPECT_OK(
      GetNode(*graph, "identity_0")->attrs().Find(kClassAttr, &input_value));
  EXPECT_EQ(input_value->list().s().at(0), expected_colocation_info);
}

// Test that a const op has device attr, no colocation info is propagated.
TEST(ColocatePredecessorTreesPassTest, ConstHasDeviceAttr) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* const_0 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_0")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(1.0)));
  Node* const_1 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_1")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", const_0, const_1, builder.opts().WithName("fill"));

  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_requested_device(kCpu0);
  GetNode(*graph, "const_0")->set_requested_device(kCpu1);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "const_0")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "const_1")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "fill")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "identity")->def(), kClassAttr));
}

// Test that a const op has colocation info, no colocation info is propagated.
TEST(ColocatePredecessorTreesPassTest, ConstHasColocationInfo) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* const_0 =
      ops::SourceOp("Const", builder.opts()
                                 .WithName("const_0")
                                 .WithAttr("dtype", DT_INT32)
                                 .WithAttr("value", Tensor(1.0))
                                 .WithAttr("_class", {"loc:@fill"}));
  Node* const_1 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_1")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", const_0, const_1, builder.opts().WithName("fill"));
  Node* identity =
      ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_requested_device(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);

  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_TRUE(HasNodeAttr(const_0->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(const_1->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(fill->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(identity->def(), kClassAttr));
}

// Test that one input is Arg, no colocation info is propagated.
TEST(ColocatePredecessorTreesPassTest, InputArg) {
  flags::Global().enable_tf2min_ici_weight.reset(true);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* arg_0 = ops::SourceOp("_Arg", builder.opts()
                                          .WithName("arg_0")
                                          .WithAttr("T", DT_INT32)
                                          .WithAttr("index", 0));
  Node* const_0 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_0")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", arg_0, const_0, builder.opts().WithName("fill"));

  ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_requested_device(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "arg_0")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "const_0")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "fill")->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(GetNode(*graph, "identity")->def(), kClassAttr));
}

}  // namespace tensorflow
