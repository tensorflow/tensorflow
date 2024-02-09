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
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/lib/core/status_test_util.h"
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

// Test a simple colocate predecessor tree example.
TEST(ColocatePredecessorTreesPassTest, SimpleExample) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  Node* const_0 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_1")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(1.0)));
  Node* const_1 = ops::SourceOp("Const", builder.opts()
                                             .WithName("const_2")
                                             .WithAttr("dtype", DT_INT32)
                                             .WithAttr("value", Tensor(2.0)));
  Node* fill =
      ops::BinaryOp("Fill", const_0, const_1, builder.opts().WithName("fill"));
  Node* identity =
      ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_assigned_device_name(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_FALSE(HasNodeAttr(const_0->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(const_1->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(fill->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(identity->def(), kClassAttr));
}

// Test that a const op has device attr, no colocation info is propagated.
TEST(ColocatePredecessorTreesPassTest, ConstHasDeviceAttr) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
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

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "const_0")->set_assigned_device_name(kCpu1);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_FALSE(HasNodeAttr(const_0->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(const_1->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(fill->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(identity->def(), kClassAttr));
}

// Test that a const op has colocation info, no colocation info is propagated.
TEST(ColocatePredecessorTreesPassTest, ConstHasColocationInfo) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
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
  GetNode(*graph, "identity")->set_assigned_device_name(kCpu0);

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
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
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
  Node* identity =
      ops::UnaryOp("Identity", fill, builder.opts().WithName("identity"));

  TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  GetNode(*graph, "identity")->set_assigned_device_name(kCpu0);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ColocatePredecessorTreesPass pass;
  TF_ASSERT_OK(pass.Run(options));

  EXPECT_FALSE(HasNodeAttr(arg_0->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(const_0->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(fill->def(), kClassAttr));
  EXPECT_FALSE(HasNodeAttr(identity->def(), kClassAttr));
}

}  // namespace tensorflow
