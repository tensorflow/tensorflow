/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/replicate_constants_pass.h"

#include <memory>
#include <string>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/config/flag_defs.h"
#include "tensorflow/core/config/flags.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/test.h"

namespace tensorflow {

const char kCpu0[] = "/job:tpu_host_worker/replica:0/task:0/device:CPU:0";
const char kCpu1[] = "/job:tpu_host_worker/replica:0/task:1/device:CPU:0";
const char kTpu00[] = "/job:tpu_host_worker/replica:0/task:0/device:TPU:0";
const char kTpu01[] = "/job:tpu_host_worker/replica:0/task:0/device:TPU:1";
const char kTpu10[] = "/job:tpu_host_worker/replica:0/task:1/device:TPU:0";
const char kTpu11[] = "/job:tpu_host_worker/replica:0/task:1/device:TPU:1";

// Return the node with name `name`.
Node* GetNode(const Graph& graph, const std::string& name) {
  for (Node* node : graph.nodes()) {
    if (node->name() == name) return node;
  }
  CHECK(false) << "Unknown node name: " << name;
  return nullptr;
}

// Return the first predecessor of `node`.
Node* GetPredecessor(Node* node) {
  auto it = node->in_nodes().begin();
  CHECK(it != node->in_nodes().end())
      << "No predecessor for " << node->name() << "\n";
  return *it;
}

// There exists an edge from `src` to `dst`.
bool IsEdge(Node* src, Node* dst) {
  for (Node* node : src->out_nodes()) {
    if (node == dst) return true;
  }
  return false;
}

// Test that a small constant is replicated to each successor's device.
TEST(ReplicateConstantsPassTest, TestSmallConstant) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    Output const0 =
        ops::Const(scope.WithOpName("const"), 1.0f, TensorShape({}));
    ops::Negate dst0(scope.WithOpName("dst0"), const0);
    ops::Negate dst1(scope.WithOpName("dst1"), const0);
    ops::Negate dst2(scope.WithOpName("dst2"), const0);
    TF_CHECK_OK(scope.ToGraph(graph.get()));
  }
  GetNode(*graph, "const")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst0")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst1")->set_assigned_device_name(kCpu1);
  GetNode(*graph, "dst2")->set_assigned_device_name(kCpu1);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ReplicateConstantsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  Node* dst0 = GetNode(*graph, "dst0");
  Node* dst1 = GetNode(*graph, "dst1");
  Node* dst2 = GetNode(*graph, "dst2");
  EXPECT_EQ(dst0->assigned_device_name(),
            GetPredecessor(dst0)->assigned_device_name());
  EXPECT_EQ(dst1->assigned_device_name(),
            GetPredecessor(dst1)->assigned_device_name());
  EXPECT_EQ(dst2->assigned_device_name(),
            GetPredecessor(dst2)->assigned_device_name());
}

// Test that a large constant is ignored.
TEST(ReplicateConstantsPassTest, TestLargeConstant) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    Output const0 =
        ops::Const(scope.WithOpName("const"),
                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    ops::Negate dst0(scope.WithOpName("dst0"), const0);
    ops::Negate dst1(scope.WithOpName("dst1"), const0);
    ops::Negate dst2(scope.WithOpName("dst2"), const0);
    TF_CHECK_OK(scope.ToGraph(graph.get()));
  }
  GetNode(*graph, "const")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst0")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst1")->set_assigned_device_name(kCpu1);
  GetNode(*graph, "dst2")->set_assigned_device_name(kCpu1);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ReplicateConstantsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  Node* dst0 = GetNode(*graph, "dst0");
  Node* dst1 = GetNode(*graph, "dst1");
  Node* dst2 = GetNode(*graph, "dst2");
  EXPECT_EQ(dst0->assigned_device_name(),
            GetPredecessor(dst0)->assigned_device_name());
  EXPECT_NE(dst1->assigned_device_name(),
            GetPredecessor(dst1)->assigned_device_name());
  EXPECT_NE(dst2->assigned_device_name(),
            GetPredecessor(dst2)->assigned_device_name());
}

// Test that a constant with a control successor is ignored.
TEST(ReplicateConstantsPassTest, TestControlOut) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    Output const0 =
        ops::Const(scope.WithOpName("const0"), 1.0f, TensorShape({}));
    Output ctrl_succ =
        ops::Const(scope.WithOpName("ctrl_succ"), 1.0f, TensorShape({}));
    ops::Negate dst0(scope.WithOpName("dst0"), const0);
    ops::Negate dst1(scope.WithOpName("dst1"), const0);
    ops::Negate dst2(scope.WithOpName("dst2"), const0);
    TF_CHECK_OK(scope.ToGraph(graph.get()));
  }
  GetNode(*graph, "const0")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "ctrl_succ")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst0")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst1")->set_assigned_device_name(kCpu1);
  GetNode(*graph, "dst2")->set_assigned_device_name(kCpu1);
  graph->AddControlEdge(GetNode(*graph, "const0"),
                        GetNode(*graph, "ctrl_succ"));

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ReplicateConstantsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  Node* dst0 = GetNode(*graph, "dst0");
  Node* dst1 = GetNode(*graph, "dst1");
  Node* dst2 = GetNode(*graph, "dst2");
  EXPECT_EQ(dst0->assigned_device_name(),
            GetPredecessor(dst0)->assigned_device_name());
  EXPECT_NE(dst1->assigned_device_name(),
            GetPredecessor(dst1)->assigned_device_name());
  EXPECT_NE(dst2->assigned_device_name(),
            GetPredecessor(dst2)->assigned_device_name());
}

// Test that a constant on a TPU is ignored.
TEST(ReplicateConstantsPassTest, TestTpuConst) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    Output const0 =
        ops::Const(scope.WithOpName("const0"), 1.0f, TensorShape({}));
    ops::Negate dst0(scope.WithOpName("dst0"), const0);
    ops::Negate dst1(scope.WithOpName("dst1"), const0);
    ops::Negate dst2(scope.WithOpName("dst2"), const0);
    TF_CHECK_OK(scope.ToGraph(graph.get()));
  }
  GetNode(*graph, "const0")->set_assigned_device_name(kTpu00);
  GetNode(*graph, "dst0")->set_assigned_device_name(kTpu00);
  GetNode(*graph, "dst1")->set_assigned_device_name(kTpu10);
  GetNode(*graph, "dst2")->set_assigned_device_name(kTpu10);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ReplicateConstantsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  Node* dst0 = GetNode(*graph, "dst0");
  Node* dst1 = GetNode(*graph, "dst1");
  Node* dst2 = GetNode(*graph, "dst2");
  EXPECT_EQ(dst0->assigned_device_name(),
            GetPredecessor(dst0)->assigned_device_name());
  EXPECT_NE(dst1->assigned_device_name(),
            GetPredecessor(dst1)->assigned_device_name());
  EXPECT_NE(dst2->assigned_device_name(),
            GetPredecessor(dst2)->assigned_device_name());
}

TEST(ReplicateConstantsPassTest, TestSmallAndLargeConstants) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    Output small = ops::Const(scope.WithOpName("small"), 1.0f, TensorShape({}));
    Output large =
        ops::Const(scope.WithOpName("large"),
                   {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                    10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});
    ops::Add dst0(scope.WithOpName("dst0"), small, large);
    ops::Add dst1(scope.WithOpName("dst1"), small, large);
    ops::Add dst2(scope.WithOpName("dst2"), small, large);
    TF_CHECK_OK(scope.ToGraph(graph.get()));
  }
  GetNode(*graph, "small")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "large")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst0")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst1")->set_assigned_device_name(kCpu1);
  GetNode(*graph, "dst2")->set_assigned_device_name(kCpu1);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ReplicateConstantsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  Node* small0 = GetNode(*graph, "small/replicate/_0");
  Node* small1 = GetNode(*graph, "small/replicate/_1");
  Node* large = GetNode(*graph, "large");
  Node* dst0 = GetNode(*graph, "dst0");
  Node* dst1 = GetNode(*graph, "dst1");
  Node* dst2 = GetNode(*graph, "dst2");
  EXPECT_EQ(small0->assigned_device_name(), kCpu0);
  EXPECT_EQ(small1->assigned_device_name(), kCpu1);
  EXPECT_EQ(large->assigned_device_name(), kCpu0);
  EXPECT_EQ(dst0->assigned_device_name(), kCpu0);
  EXPECT_EQ(dst1->assigned_device_name(), kCpu1);
  EXPECT_EQ(dst1->assigned_device_name(), kCpu1);
  EXPECT_TRUE(IsEdge(small0, dst0));
  EXPECT_TRUE(IsEdge(large, dst0));
  EXPECT_TRUE(IsEdge(small1, dst1));
  EXPECT_TRUE(IsEdge(large, dst1));
  EXPECT_TRUE(IsEdge(small1, dst2));
  EXPECT_TRUE(IsEdge(large, dst2));
}

// Test that a constant at a CPU with TPU successors is replicated to the
// TPUs' host CPUs.
TEST(ReplicateConstantsPassTest, TestTpuDestinations) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    Output const0 =
        ops::Const(scope.WithOpName("const"), 1.0f, TensorShape({}));
    ops::Negate dst00(scope.WithOpName("dst00"), const0);
    ops::Negate dst01(scope.WithOpName("dst01"), const0);
    ops::Negate dst10(scope.WithOpName("dst10"), const0);
    ops::Negate dst11(scope.WithOpName("dst11"), const0);
    TF_CHECK_OK(scope.ToGraph(graph.get()));
  }
  GetNode(*graph, "const")->set_assigned_device_name(kCpu0);
  GetNode(*graph, "dst00")->set_assigned_device_name(kTpu00);
  GetNode(*graph, "dst01")->set_assigned_device_name(kTpu01);
  GetNode(*graph, "dst10")->set_assigned_device_name(kTpu10);
  GetNode(*graph, "dst11")->set_assigned_device_name(kTpu11);

  GraphDef before;
  graph->ToGraphDef(&before);
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  ReplicateConstantsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  GraphDef actual;
  graph->ToGraphDef(&actual);

  Node* const0 = GetNode(*graph, "const/replicate/_0");
  Node* const1 = GetNode(*graph, "const/replicate/_1");
  Node* dst00 = GetNode(*graph, "dst00");
  Node* dst01 = GetNode(*graph, "dst01");
  Node* dst10 = GetNode(*graph, "dst10");
  Node* dst11 = GetNode(*graph, "dst11");
  EXPECT_EQ(const0->assigned_device_name(), kCpu0);
  EXPECT_EQ(const1->assigned_device_name(), kCpu1);
  EXPECT_TRUE(IsEdge(const0, dst00));
  EXPECT_TRUE(IsEdge(const0, dst01));
  EXPECT_TRUE(IsEdge(const1, dst10));
  EXPECT_TRUE(IsEdge(const1, dst11));
}

}  // namespace tensorflow
