/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/partially_decluster_pass.h"

#include "absl/memory/memory.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/test_util.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/tf2xla/cc/ops/xla_ops.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {
REGISTER_OP("FakeNullary").Output("out: int32");

REGISTER_OP("FakeBinary")
    .Input("host_in: int32")
    .Input("device_in: int32")
    .Output("host_out: int32")
    .Output("device_out: int32");

REGISTER_OP("FakeResourceVar").Output("out: resource");

REGISTER_OP("FakeResourceUpdate")
    .Input("in: resource")
    .Output("out: resource")
    .Output("something_else: int32");

class FakeBinaryOp : public OpKernel {
 public:
  explicit FakeBinaryOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override { CHECK(false); }
};

class FakeResourceUpdateOp : public OpKernel {
 public:
  explicit FakeResourceUpdateOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override { CHECK(false); }
};

REGISTER_KERNEL_BUILDER(Name("FakeBinary")
                            .Device(DEVICE_CPU)
                            .HostMemory("host_in")
                            .HostMemory("host_out"),
                        FakeBinaryOp);

REGISTER_KERNEL_BUILDER(
    Name("FakeResourceUpdate").Device(DEVICE_CPU).HostMemory("something_else"),
    FakeResourceUpdateOp);

absl::Status PartiallyDecluster(std::unique_ptr<Graph>* graph) {
  FixupSourceAndSinkEdges(graph->get());
  // Assign all nodes to the CPU device.
  static const char* kCpuDevice = "/job:localhost/replica:0/task:0/cpu:0";
  for (Node* n : (*graph)->nodes()) {
    if (n->assigned_device_name().empty()) {
      n->set_assigned_device_name(kCpuDevice);
    }
  }

  GraphOptimizationPassWrapper wrapper;
  GraphOptimizationPassOptions opt_options =
      wrapper.CreateGraphOptimizationPassOptions(graph);

  PartiallyDeclusterPass pass;
  return pass.Run(opt_options);
}

Node* FindNodeByName(const Graph& graph, const string& name) {
  for (Node* node : graph.nodes()) {
    if (node->name() == name) {
      return node;
    }
  }
  return nullptr;
}

bool GetInputsForNode(const Graph& graph, const string& node_name,
                      std::vector<Node*>* inputs) {
  const Node* node = FindNodeByName(graph, node_name);
  if (node == nullptr) {
    return false;
  }
  for (const Edge* e : node->in_edges()) {
    inputs->push_back(e->src());
  }
  std::sort(inputs->begin(), inputs->end(), NodeComparatorName());
  return true;
}

TEST(PartiallyDeclusterPassTest, ClusteredAndUnclustered) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* input =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input"));
    Node* clustered_producer =
        ops::BinaryOp("FakeBinary", input, input,
                      builder.opts().WithName("ClusteredProducer"));
    ops::BinaryOp("FakeBinary", clustered_producer, input,
                  builder.opts().WithName("UnclusteredConsumer"));
    Node* clustered_consumer =
        ops::BinaryOp("FakeBinary", {clustered_producer, 1}, input,
                      builder.opts().WithName("ClusteredConsumer"));
    clustered_producer->AddAttr(kXlaClusterAttr, "cluster_0");
    clustered_consumer->AddAttr(kXlaClusterAttr, "cluster_0");
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(PartiallyDecluster(&graph));
  std::vector<Node*> unclustered_consumer_inputs;
  ASSERT_TRUE(GetInputsForNode(*graph, "UnclusteredConsumer",
                               &unclustered_consumer_inputs));
  ASSERT_EQ(unclustered_consumer_inputs.size(), 2);
  EXPECT_EQ(unclustered_consumer_inputs[0]->name(),
            "ClusteredProducer/declustered");
  EXPECT_EQ(unclustered_consumer_inputs[1]->name(), "Input");

  std::vector<Node*> clustered_consumer_inputs;
  ASSERT_TRUE(GetInputsForNode(*graph, "ClusteredConsumer",
                               &clustered_consumer_inputs));
  ASSERT_EQ(clustered_consumer_inputs.size(), 2);
  EXPECT_EQ(clustered_consumer_inputs[0]->name(), "ClusteredProducer");
  EXPECT_EQ(clustered_consumer_inputs[1]->name(), "Input");
}

TEST(PartiallyDeclusterPassTest, DifferentClusters) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* input =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input"));
    Node* clustered_producer =
        ops::BinaryOp("FakeBinary", input, input,
                      builder.opts().WithName("ClusteredProducer"));
    Node* consumer_in_different_cluster =
        ops::BinaryOp("FakeBinary", clustered_producer, input,
                      builder.opts().WithName("ConsumerInDifferentCluster"));
    Node* clustered_consumer =
        ops::BinaryOp("FakeBinary", input, {clustered_producer, 1},
                      builder.opts().WithName("ClusteredConsumer"));
    clustered_producer->AddAttr(kXlaClusterAttr, "cluster_0");
    clustered_consumer->AddAttr(kXlaClusterAttr, "cluster_0");
    consumer_in_different_cluster->AddAttr(kXlaClusterAttr, "cluster_1");
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(PartiallyDecluster(&graph));
  std::vector<Node*> inputs;
  ASSERT_TRUE(GetInputsForNode(*graph, "ConsumerInDifferentCluster", &inputs));
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0]->name(), "ClusteredProducer/declustered");
  EXPECT_EQ(inputs[1]->name(), "Input");
}

TEST(PartiallyDeclusterPassTest, DontDeclusterIfUserIsDeviceMem) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* input =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input"));
    Node* clustered_producer =
        ops::BinaryOp("FakeBinary", input, input,
                      builder.opts().WithName("ClusteredProducer"));
    // The first input is hostmem and the second input is devicemem.
    Node* consumer_in_different_cluster =
        ops::BinaryOp("FakeBinary", input, clustered_producer,
                      builder.opts().WithName("ConsumerInDifferentCluster"));
    Node* clustered_consumer =
        ops::BinaryOp("FakeBinary", input, {clustered_producer, 1},
                      builder.opts().WithName("ClusteredConsumer"));
    clustered_producer->AddAttr(kXlaClusterAttr, "cluster_0");
    clustered_consumer->AddAttr(kXlaClusterAttr, "cluster_0");
    consumer_in_different_cluster->AddAttr(kXlaClusterAttr, "cluster_1");
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(PartiallyDecluster(&graph));
  std::vector<Node*> inputs;
  ASSERT_TRUE(GetInputsForNode(*graph, "ConsumerInDifferentCluster", &inputs));
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0]->name(), "ClusteredProducer");
  EXPECT_EQ(inputs[1]->name(), "Input");
}

TEST(PartiallyDeclusterPassTest, DontDuplicateResourceVarOps) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* input =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input"));
    Node* resource_var = ops::SourceOp("FakeResourceVar",
                                       builder.opts().WithName("ResourceVar"));
    Node* clustered_producer =
        ops::UnaryOp("FakeResourceUpdate", resource_var,
                     builder.opts().WithName("ClusteredProducer"));
    Node* consumer_in_different_cluster =
        ops::BinaryOp("FakeBinary", {clustered_producer, 1}, input,
                      builder.opts().WithName("ConsumerInDifferentCluster"));
    Node* clustered_consumer =
        ops::BinaryOp("FakeBinary", input, {clustered_producer, 1},
                      builder.opts().WithName("ClusteredConsumer"));
    clustered_producer->AddAttr(kXlaClusterAttr, "cluster_0");
    clustered_consumer->AddAttr(kXlaClusterAttr, "cluster_0");
    consumer_in_different_cluster->AddAttr(kXlaClusterAttr, "cluster_1");
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(PartiallyDecluster(&graph));
  std::vector<Node*> inputs;
  ASSERT_TRUE(GetInputsForNode(*graph, "ConsumerInDifferentCluster", &inputs));
  ASSERT_EQ(inputs.size(), 2);
  EXPECT_EQ(inputs[0]->name(), "ClusteredProducer");
  EXPECT_EQ(inputs[1]->name(), "Input");
}

TEST(PartiallyDeclusterPassTest, DeclusterDependentNodes) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* input =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input"));
    Node* clustered_producer_0 =
        ops::BinaryOp("FakeBinary", input, input,
                      builder.opts().WithName("ClusteredProducer0"));
    Node* clustered_producer_1 =
        ops::BinaryOp("FakeBinary", clustered_producer_0, input,
                      builder.opts().WithName("ClusteredProducer1"));
    ops::BinaryOp("FakeBinary", clustered_producer_1, input,
                  builder.opts().WithName("UnclusteredConsumer"));
    Node* clustered_consumer =
        ops::BinaryOp("FakeBinary", {clustered_producer_1, 1}, input,
                      builder.opts().WithName("ClusteredConsumer"));
    clustered_producer_0->AddAttr(kXlaClusterAttr, "cluster_0");
    clustered_producer_1->AddAttr(kXlaClusterAttr, "cluster_0");
    clustered_consumer->AddAttr(kXlaClusterAttr, "cluster_0");
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(PartiallyDecluster(&graph));
  std::vector<Node*> unclustered_consumer_inputs, declustered_producer_1_inputs;

  ASSERT_TRUE(GetInputsForNode(*graph, "UnclusteredConsumer",
                               &unclustered_consumer_inputs));
  ASSERT_EQ(unclustered_consumer_inputs.size(), 2);
  EXPECT_EQ(unclustered_consumer_inputs[0]->name(),
            "ClusteredProducer1/declustered");
  EXPECT_EQ(unclustered_consumer_inputs[1]->name(), "Input");

  ASSERT_TRUE(GetInputsForNode(*graph, "ClusteredProducer1/declustered",
                               &declustered_producer_1_inputs));
  ASSERT_EQ(declustered_producer_1_inputs.size(), 2);
  EXPECT_EQ(declustered_producer_1_inputs[0]->name(),
            "ClusteredProducer0/declustered");
  EXPECT_EQ(declustered_producer_1_inputs[1]->name(), "Input");
}

void AddToCluster(absl::Span<Node* const> nodes,
                  absl::string_view cluster_name) {
  for (Node* n : nodes) {
    n->AddAttr(kXlaClusterAttr, string(cluster_name));
  }
}

TEST(PartiallyDeclusterPassTest, DeclusterMustBeConstantNodes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output shape_a = ops::Placeholder(s.WithOpName("shape_a"), DT_INT32,
                                    ops::Placeholder::Attrs{});
  Output shape_b = ops::Placeholder(s.WithOpName("shape_b"), DT_INT32,
                                    ops::Placeholder::Attrs{});
  Output shape = ops::Add(s.WithOpName("shape"), shape_a, shape_b);

  Output reshape_input = ops::Placeholder(s.WithOpName("reshape_input"),
                                          DT_FLOAT, ops::Placeholder::Attrs{});
  Output reshape = ops::Reshape(s.WithOpName("reshape"), reshape_input, shape);

  AddToCluster({shape.node(), reshape.node()}, "cluster_0");

  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(s.ToGraph(graph.get()));
  TF_ASSERT_OK(PartiallyDecluster(&graph));

  const Node* n = FindNodeByName(*graph, "shape");
  ASSERT_NE(n, nullptr);

  EXPECT_EQ(GetXlaClusterForNode(*n), std::nullopt);
}

TEST(PartiallyDeclusterPassTest, DeclusteringStopsAtMetadataOps) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input_a = ops::Placeholder(s.WithOpName("input_a"), DT_INT32,
                                    ops::Placeholder::Attrs{});
  Output input_b = ops::Placeholder(s.WithOpName("shape_b"), DT_FLOAT,
                                    ops::Placeholder::Attrs{});
  Output mul = ops::Mul(s.WithOpName("mul"), input_b, input_b);
  Output shape_of_mul = ops::Shape(s.WithOpName("shape_of_mul"), mul);

  Output shape = ops::Add(s.WithOpName("shape"), shape_of_mul, input_a);

  Output reshape_input = ops::Placeholder(s.WithOpName("reshape_input"),
                                          DT_FLOAT, ops::Placeholder::Attrs{});
  Output reshape = ops::Reshape(s.WithOpName("reshape"), reshape_input, shape);

  AddToCluster({mul.node(), shape_of_mul.node(), shape.node(), reshape.node()},
               "cluster_0");

  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(s.ToGraph(graph.get()));
  TF_ASSERT_OK(PartiallyDecluster(&graph));

  const Node* n = FindNodeByName(*graph, "shape");
  ASSERT_NE(n, nullptr);

  EXPECT_EQ(GetXlaClusterForNode(*n), "cluster_0");
}

TEST(PartiallyDeclusterPassTest, EdgeAcrossDifferentClusters) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output shape_a = ops::Placeholder(s.WithOpName("shape_a"), DT_INT32,
                                    ops::Placeholder::Attrs{});
  Output shape_b = ops::Placeholder(s.WithOpName("shape_b"), DT_INT32,
                                    ops::Placeholder::Attrs{});
  Output shape = ops::Add(s.WithOpName("shape"), shape_a, shape_b);

  Output reshape_input = ops::Placeholder(s.WithOpName("reshape_input"),
                                          DT_FLOAT, ops::Placeholder::Attrs{});
  Output reshape = ops::Reshape(s.WithOpName("reshape"), reshape_input, shape);

  AddToCluster({reshape.node()}, "cluster_0");
  AddToCluster({shape.node()}, "cluster_1");

  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(s.ToGraph(graph.get()));
  TF_ASSERT_OK(PartiallyDecluster(&graph));

  const Node* n = FindNodeByName(*graph, "shape");
  ASSERT_NE(n, nullptr);

  EXPECT_EQ(GetXlaClusterForNode(*n), "cluster_1");
}

TEST(PartiallyDeclusterPassTest, DontDeclusterXlaDeviceOps) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output shape_a = ops::Placeholder(s.WithOpName("shape_a"), DT_INT32,
                                    ops::Placeholder::Attrs{});
  Output shape_b = ops::Placeholder(s.WithOpName("shape_b"), DT_INT32,
                                    ops::Placeholder::Attrs{});
  Output shape = ops::Add(s.WithOpName("shape"), shape_a, shape_b);

  Output reshape_input = ops::Placeholder(s.WithOpName("reshape_input"),
                                          DT_FLOAT, ops::Placeholder::Attrs{});
  Output reshape = ops::Reshape(s.WithOpName("reshape"), reshape_input, shape);

  AddToCluster({shape.node(), reshape.node()}, "cluster_0");

  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(s.ToGraph(graph.get()));

  // This is needed to register the XLA_GPU device.
  std::vector<std::unique_ptr<Device>> devices;
  TF_ASSERT_OK(DeviceFactory::AddDevices(
      SessionOptions(), "/job:localhost/replica:0/task:0", &devices));

  // Scope::ToGraph loses the assigned device name since it goes through
  // GraphDef/NodeDef which does not have a field for the assigned device name.
  Node* n = FindNodeByName(*graph, "shape");
  ASSERT_NE(n, nullptr);
  n->set_assigned_device_name(
      "/job:localhost/replica:0/task:0/device:XLA_GPU:0");

  TF_ASSERT_OK(PartiallyDecluster(&graph));

  EXPECT_EQ(GetXlaClusterForNode(*n), "cluster_0");
}

TEST(PartiallyDeclusterPassTest, EliminatedUnusedNodes) {
  const char* const kClusteredProducer0Name = "ClusteredProducer0";
  const char* const kClusteredProducer1Name = "ClusteredProducer1";

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
    Node* input =
        ops::SourceOp("FakeNullary", builder.opts().WithName("Input"));
    Node* clustered_producer_0 =
        ops::BinaryOp("FakeBinary", input, input,
                      builder.opts().WithName(kClusteredProducer0Name));
    Node* clustered_producer_1 =
        ops::BinaryOp("FakeBinary", clustered_producer_0, input,
                      builder.opts().WithName(kClusteredProducer1Name));
    ops::BinaryOp("FakeBinary", clustered_producer_1, input,
                  builder.opts().WithName("UnclusteredConsumer"));
    clustered_producer_0->AddAttr(kXlaClusterAttr, "cluster_0");
    clustered_producer_1->AddAttr(kXlaClusterAttr, "cluster_0");
    TF_EXPECT_OK(GraphDefBuilderToGraph(builder, graph.get()));
  }

  TF_ASSERT_OK(PartiallyDecluster(&graph));
  EXPECT_EQ(FindNodeByName(*graph, kClusteredProducer0Name), nullptr);
  EXPECT_EQ(FindNodeByName(*graph, kClusteredProducer1Name), nullptr);
}

TEST(PartiallyDeclusterPassTest, MetadataOpsDontStartClusters) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::Scope in_cluster_and = root.WithXlaCluster("cluster_0");

  Output a = ops::Placeholder(root.WithOpName("a"), DT_FLOAT);
  Output b = ops::Shape(in_cluster_and.WithOpName("b"), a);
  Output c = ops::Rank(in_cluster_and.WithOpName("c"), b);
  Output d = ops::Size(in_cluster_and.WithOpName("d"), c);
  (void)ops::Shape(in_cluster_and.WithOpName("e"), d);

  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(graph.get()));

  TF_ASSERT_OK(PartiallyDecluster(&graph));

  Node* n_b = FindNodeByName(*graph, "b");
  ASSERT_NE(n_b, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_b), std::nullopt);

  Node* n_c = FindNodeByName(*graph, "c");
  ASSERT_NE(n_c, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_c), std::nullopt);

  Node* n_d = FindNodeByName(*graph, "d");
  ASSERT_NE(n_d, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_d), std::nullopt);

  Node* n_e = FindNodeByName(*graph, "e");
  ASSERT_NE(n_e, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_e), std::nullopt);
}

TEST(PartiallyDeclusterPassTest, MetaConsumersArentDeclustered) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::Scope in_cluster_and = root.WithXlaCluster("cluster_0");
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Output a = ops::Placeholder(root.WithOpName("a"), DT_FLOAT);
  Output b = ops::Add(in_cluster_and.WithOpName("b"), a, a);
  Output c = ops::Rank(in_cluster_and.WithOpName("c"), b);

  Output e;
  TF_ASSERT_OK(
      CreateOutputWithScope("FakeBinary", {c, c}, root.WithOpName("e"), &e));

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(PartiallyDecluster(&graph));

  Node* n_b = FindNodeByName(*graph, "b");
  ASSERT_NE(n_b, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_b), "cluster_0");

  Node* n_c = FindNodeByName(*graph, "c");
  ASSERT_NE(n_c, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_c), "cluster_0");
}

TEST(PartiallyDeclusterPassTest, ConstInputsToSliceArentDeclustered) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::Scope in_cluster_and = root.WithXlaCluster("cluster_0");
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Output a = ops::Placeholder(root.WithOpName("a"), DT_FLOAT,
                              ops::Placeholder::Attrs{{4}});
  Output b = ops::Const(in_cluster_and.WithOpName("b"), {1});
  Output c = ops::Const(in_cluster_and.WithOpName("c"), {2});
  Output d = ops::Slice(in_cluster_and.WithOpName("d"), a, b, c);

  TF_ASSERT_OK(root.ToGraph(graph.get()));
  TF_ASSERT_OK(PartiallyDecluster(&graph));

  Node* n_b = FindNodeByName(*graph, "b");
  ASSERT_NE(n_b, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_b), "cluster_0");

  Node* n_c = FindNodeByName(*graph, "c");
  ASSERT_NE(n_c, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_c), "cluster_0");
}

TEST(PartiallyDeclusterPassTest,
     ConstInLoopWithCrossDeviceControlInputsAreDeclustered) {
  // Based on DontClusterTheSpecialIdentityDrivingConstsInLoop in
  // mark_for_compilation_pass_test.cc
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::Scope in_cluster_and = root.WithXlaCluster("cluster_0");
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Output a = ops::Placeholder(root.WithOpName("a"), DT_FLOAT,
                              ops::Placeholder::Attrs{{4}});
  Output b = ops::Const(in_cluster_and.WithOpName("b"), {1});
  Output c = ops::Const(in_cluster_and.WithOpName("c"), {2});
  Output slice = ops::Slice(in_cluster_and.WithOpName("slice"), a, b, c);
  Output cond = ops::Placeholder(root.WithOpName("cond"), DT_BOOL);
  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  Output loop_cond = ops::LoopCond(root.WithOpName("loop_cond"), cond);
  ops::Switch switch_node(root.WithOpName("switch"), value, loop_cond);
  Output identity =
      ops::Identity(root.WithOpName("identity"), switch_node.output_true);
  root.graph()->AddControlEdge(identity.node(), b.node());

  TF_ASSERT_OK(root.ToGraph(graph.get()));

  // This is needed to register the XLA_GPU device.
  std::vector<std::unique_ptr<Device>> devices;
  TF_ASSERT_OK(DeviceFactory::AddDevices(
      SessionOptions(), "/job:localhost/replica:0/task:0", &devices));

  // Scope::ToGraph loses the assigned device name since it goes through
  // GraphDef/NodeDef which does not have a field for the assigned device name.
  Node* identity_node = FindNodeByName(*graph, "identity");
  ASSERT_NE(identity_node, nullptr);
  identity_node->set_assigned_device_name(
      "/job:localhost/replica:0/task:0/device:XLA_GPU:0");

  TF_ASSERT_OK(PartiallyDecluster(&graph));

  Node* n_b = FindNodeByName(*graph, "b");
  ASSERT_NE(n_b, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_b), std::nullopt);

  Node* n_c = FindNodeByName(*graph, "c");
  ASSERT_NE(n_c, nullptr);
  EXPECT_EQ(GetXlaClusterForNode(*n_c), "cluster_0");
}

}  // namespace
}  // namespace tensorflow
