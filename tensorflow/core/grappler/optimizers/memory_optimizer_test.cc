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

#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/device_properties.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class RecomputeSubgraphTest : public GrapplerTest {};

TEST_F(RecomputeSubgraphTest, SimpleSubgraph) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("a"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Identity(s.WithOpName("b"), a);  // Recomputed
  Output c = ops::Identity(s.WithOpName("c"), b);
  Output d = ops::AddN(s.WithOpName("gradients/d"), {c});
  Output e = ops::AddN(s.WithOpName("gradients/e"), {d, b});
  Output f = ops::AddN(s.WithOpName("gradients/f"), {e, a});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(6, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  (*pre_transform_node_map.GetNode("b")->mutable_attr())["_recompute_hint"]
      .set_i(0);

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);

  TF_EXPECT_OK(status);
  NodeMap post_transform_node_map(&output);
  EXPECT_EQ(8, output.node_size());
  NodeDef* transformed_e = post_transform_node_map.GetNode(e.name());
  EXPECT_EQ(2, transformed_e->input_size());
  EXPECT_EQ("gradients/d", transformed_e->input(0));
  EXPECT_EQ("Recomputed/b", transformed_e->input(1));
  NodeDef* recomputed_b = post_transform_node_map.GetNode("Recomputed/b");
  EXPECT_EQ(2, recomputed_b->input_size());
  EXPECT_EQ("a", recomputed_b->input(0));
  EXPECT_EQ("^RecomputeTrigger/b", recomputed_b->input(1));
  NodeDef* recompute_trigger =
      post_transform_node_map.GetNode("RecomputeTrigger/b");
  EXPECT_EQ(1, recompute_trigger->input_size());
  EXPECT_EQ("^gradients/d", recompute_trigger->input(0));
}

TEST_F(RecomputeSubgraphTest, NoFeedsRecomputed) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("a"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Identity(s.WithOpName("b"), a);  // Would be recomputed, but
                                                   // for being fed
  Output c = ops::Identity(s.WithOpName("c"), b);
  Output d = ops::AddN(s.WithOpName("gradients/d"), {c});
  Output e = ops::AddN(s.WithOpName("gradients/e"), {d, b});
  Output f = ops::AddN(s.WithOpName("gradients/f"), {e, a});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.feed.emplace_back("b", Tensor());
  EXPECT_EQ(6, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  (*pre_transform_node_map.GetNode("b")->mutable_attr())["_recompute_hint"]
      .set_i(0);

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);

  TF_EXPECT_OK(status);
  EXPECT_EQ(6, output.node_size());
}

TEST_F(RecomputeSubgraphTest, TwoInputSubgraphs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("a"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Variable(s.WithOpName("b"), {2, 3, 4}, DT_FLOAT);
  Output d = ops::AddN(
      s.WithOpName("some_name_scope/gradients/two_subgraph_inputs"), {a, b});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(3, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  (*pre_transform_node_map.GetNode("a")->mutable_attr())["_recompute_hint"]
      .set_i(0);
  (*pre_transform_node_map.GetNode("b")->mutable_attr())["_recompute_hint"]
      .set_i(0);

  MemoryOptimizer optimizer(RewriterConfig::MANUAL,
                            "some_name_scope/gradients");
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);

  TF_EXPECT_OK(status);
  NodeMap post_transform_node_map(&output);
  // Mostly checking that this case does not crash.
  EXPECT_EQ(7, output.node_size());
  EXPECT_NE(post_transform_node_map.GetNode("Recomputed/a"), nullptr);
  EXPECT_NE(post_transform_node_map.GetNode("Recomputed/b"), nullptr);
  EXPECT_NE(post_transform_node_map.GetNode("RecomputeTrigger/a"), nullptr);
  EXPECT_NE(post_transform_node_map.GetNode("RecomputeTrigger/b"), nullptr);
}

TEST_F(RecomputeSubgraphTest, MultiNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("Conv"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Identity(s.WithOpName("BN"), a);    // Recomputed
  Output c = ops::Identity(s.WithOpName("ReLU"), b);  // Recomputed
  Output d = ops::Identity(s.WithOpName("Conv1"), c);

  // The "gradients/" prefix means the heuristic will pick these up as
  // candidates to have their inputs recomputed.
  Output trigger = ops::AddN(s.WithOpName("gradients/BN1Grad"), {d});
  Output e = ops::AddN(s.WithOpName("gradients/Conv1Grad"), {trigger, c});
  Output f = ops::AddN(s.WithOpName("gradients/ReLUGrad"), {e, c});
  Output g = ops::AddN(s.WithOpName("gradients/BNGrad"), {f, a});
  Output h = ops::AddN(s.WithOpName("gradients/ConvGrad"), {g});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(9, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  // Set op types so that the heuristic will pick these nodes up to be
  // recomputed
  pre_transform_node_map.GetNode("BN")->set_op("FusedBatchNorm");
  pre_transform_node_map.GetNode("ReLU")->set_op("Relu");

  MemoryOptimizer optimizer(RewriterConfig::RECOMPUTATION_HEURISTICS);
  GraphDef first_pass_output;
  Status first_pass_status =
      optimizer.Optimize(nullptr, item, &first_pass_output);
  TF_EXPECT_OK(first_pass_status);

  NodeMap post_transform_node_map(&first_pass_output);
  EXPECT_EQ(13, first_pass_output.node_size());
  NodeDef* transformed_e = post_transform_node_map.GetNode(e.name());
  EXPECT_EQ(2, transformed_e->input_size());
  EXPECT_EQ("gradients/BN1Grad", transformed_e->input(0));
  EXPECT_EQ("Recomputed/ReLU", transformed_e->input(1));
  NodeDef* transformed_f = post_transform_node_map.GetNode(f.name());
  EXPECT_EQ(2, transformed_f->input_size());
  EXPECT_EQ("gradients/Conv1Grad", transformed_f->input(0));
  EXPECT_EQ("Recomputed/ReLU", transformed_f->input(1));
  NodeDef* transformed_g = post_transform_node_map.GetNode(g.name());
  EXPECT_EQ(2, transformed_g->input_size());
  EXPECT_EQ("gradients/ReLUGrad", transformed_g->input(0));
  EXPECT_EQ("Conv", transformed_g->input(1));

  NodeDef* recomputed_b = post_transform_node_map.GetNode("Recomputed/BN");
  EXPECT_EQ(2, recomputed_b->input_size());
  EXPECT_EQ("Conv", recomputed_b->input(0));
  EXPECT_EQ("^RecomputeTrigger/BN", recomputed_b->input(1));
  NodeDef* recompute_trigger_b =
      post_transform_node_map.GetNode("RecomputeTrigger/BN");
  EXPECT_EQ(1, recompute_trigger_b->input_size());
  EXPECT_EQ("^RecomputeTrigger/ReLU", recompute_trigger_b->input(0));

  NodeDef* recomputed_c = post_transform_node_map.GetNode("Recomputed/ReLU");
  EXPECT_EQ(2, recomputed_c->input_size());
  EXPECT_EQ("Recomputed/BN", recomputed_c->input(0));
  EXPECT_EQ("^RecomputeTrigger/ReLU", recomputed_c->input(1));
  NodeDef* recompute_trigger_c =
      post_transform_node_map.GetNode("RecomputeTrigger/ReLU");
  EXPECT_EQ(1, recompute_trigger_c->input_size());
  EXPECT_EQ("^gradients/BN1Grad", recompute_trigger_c->input(0));
}

class MemoryOptimizerTest : public GrapplerTest {
 public:
  static std::unique_ptr<VirtualCluster> CreateVirtualCluster() {
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_frequency(1000);
    cpu_device.set_num_cores(4);
    cpu_device.set_bandwidth(32);
    cpu_device.set_memory_size(1024 * 1024);
    DeviceProperties gpu_device;
    gpu_device.set_type("GPU");
    gpu_device.set_frequency(1000);
    gpu_device.set_num_cores(24);
    gpu_device.set_bandwidth(128);
    gpu_device.set_memory_size(1024 * 1024);
    gpu_device.mutable_environment()->insert({"architecture", "6"});
    std::unordered_map<string, DeviceProperties> devices;
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    devices["/job:localhost/replica:0/task:0/gpu:0"] = gpu_device;
    return std::unique_ptr<VirtualCluster>(new VirtualCluster(devices));
  }
};

TEST_F(MemoryOptimizerTest, SimpleSwapping) {
  // Build a simple graph with an op that's marked for swapping.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a =
      ops::Variable(s.WithOpName("a").WithDevice("/gpu:0"), {10, 10}, DT_FLOAT);
  Output b = ops::AddN(s.WithOpName("b").WithDevice("/gpu:0"), {a});
  Output c = ops::AddN(s.WithOpName("c").WithDevice("/gpu:0"), {b});
  Output d = ops::AddN(s.WithOpName("d").WithDevice("/gpu:0"), {c});
  Output e = ops::AddN(s.WithOpName("e").WithDevice("/gpu:0"), {b, d});

  Output constant = ops::Const(s.WithOpName("constant"), 0.0f, {10, 10});
  Output init = ops::Assign(s.WithOpName("init"), a, constant);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  EXPECT_EQ(7, item.graph.node_size());
  EXPECT_EQ(NodeName(e.name()), item.graph.node(4).name());
  AttrValue& val =
      (*item.graph.mutable_node(4)->mutable_attr())["_swap_to_host"];
  val.mutable_list()->add_i(0);

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  Status status = optimizer.Optimize(cluster.get(), item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(9, output.node_size());
  const NodeDef& new_e = output.node(6);
  EXPECT_EQ(NodeName(e.name()), new_e.name());

  EXPECT_EQ(2, new_e.input_size());
  EXPECT_EQ(NodeName(d.name()), new_e.input(1));
  EXPECT_EQ("swap_in_e_0", new_e.input(0));

  const NodeDef& swap_out = output.node(7);
  EXPECT_EQ("swap_out_e_0", swap_out.name());
  EXPECT_EQ("_CopyFromGpuToHost", swap_out.op());

  const NodeDef& swap_in = output.node(8);
  EXPECT_EQ("swap_in_e_0", swap_in.name());
  EXPECT_EQ("_CopyFromHostToGpu", swap_in.op());

  EXPECT_EQ(NodeName(b.name()), swap_out.input(0));
  EXPECT_EQ(NodeName(swap_out.name()), swap_in.input(0));
  EXPECT_EQ("^c", swap_in.input(1));

  const NodeDef& new_c = output.node(4);
  EXPECT_EQ(NodeName(c.name()), new_c.name());
  EXPECT_EQ("^swap_out_e_0", new_c.input(1));

  // Run the optimizer a second time to ensure it's idempotent.
  GrapplerItem item_copy = item.WithGraph(std::move(output));
  status = optimizer.Optimize(cluster.get(), item_copy, &output);
  TF_EXPECT_OK(status);

#if GOOGLE_CUDA
  item.fetch = {"e"};
  item.init_ops = {init.name()};
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
#endif
}

TEST_F(MemoryOptimizerTest, SwappingHeuristics) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output v = ops::Variable(s.WithOpName("v").WithDevice("/gpu:0"),
                           {128, 128, 8}, DT_FLOAT);
  Output a = ops::Identity(s.WithOpName("a").WithDevice("/gpu:0"), v);
  Output b = ops::Square(s.WithOpName("b").WithDevice("/gpu:0"), v);
  Output c = ops::Sqrt(s.WithOpName("c").WithDevice("/gpu:0"), a);
  Output d = ops::Identity(s.WithOpName("d").WithDevice("/gpu:0"), b);
  Output axis = ops::Const(s.WithOpName("axis"), 0);
  Output e =
      ops::Concat(s.WithOpName("e").WithDevice("/gpu:0"), {a, b, c, d}, axis);
  Output f = ops::Square(s.WithOpName("f").WithDevice("/gpu:0"), a);
  Output g = ops::Sqrt(s.WithOpName("g").WithDevice("/gpu:0"), b);
  Output h = ops::Exp(s.WithOpName("h").WithDevice("/gpu:0"), c);
  Output i = ops::Log(s.WithOpName("i").WithDevice("/gpu:0"), d);

  Output constant = ops::Const(s.WithOpName("constant"), 0.0f, {128, 128, 8});
  Output init = ops::Assign(s.WithOpName("init"), v, constant);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"e", "f", "g", "h", "i"};
  item.init_ops = {init.name()};

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());

  MemoryOptimizer optimizer(RewriterConfig::SWAPPING_HEURISTICS);
  GraphDef output;
  Status status = optimizer.Optimize(cluster.get(), item, &output);
  TF_EXPECT_OK(status);

  for (const auto& node : output.node()) {
    if (node.name() == "e") {
      EXPECT_EQ(5, node.input_size());
      EXPECT_EQ("a", node.input(0));
      EXPECT_EQ("swap_in_e_1", node.input(1));
      EXPECT_EQ("swap_in_e_2", node.input(2));
      EXPECT_EQ("swap_in_e_3", node.input(3));
      EXPECT_EQ("axis", node.input(4));
    }
  }

#if GOOGLE_CUDA
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
  }
#endif
}

TEST_F(MemoryOptimizerTest, UnswappableInputs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output v = ops::Variable(s.WithOpName("v").WithDevice("/gpu:0"),
                           {128, 128, 8}, DT_FLOAT);
  Output a = ops::Square(s.WithOpName("a").WithDevice("/gpu:0"), v);
  Output b = ops::Identity(s.WithOpName("b").WithDevice("/gpu:0"), {a});
  Output c = ops::Identity(s.WithOpName("c").WithDevice("/gpu:0"), {a});
  Output index = ops::Const(s.WithOpName("index"), {0});
  Output indices = ops::Tile(s.WithOpName("indices"), index, {128});
  Output d =
      ops::ScatterAdd(s.WithOpName("d").WithDevice("/gpu:0"), v, indices, c);
  Output axis = ops::Const(s.WithOpName("axis"), 0);
  Output e =
      ops::Concat(s.WithOpName("e").WithDevice("/gpu:0"), {b, c, d}, axis);

  Output constant = ops::Const(s.WithOpName("constant"), 0.0f, {128, 128, 8});
  Output init = ops::Assign(s.WithOpName("init"), v, constant);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"e"};
  item.init_ops = {init.name()};

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());

  MemoryOptimizer optimizer(RewriterConfig::SWAPPING_HEURISTICS);
  GraphDef output;
  Status status = optimizer.Optimize(cluster.get(), item, &output);
  TF_EXPECT_OK(status);

  for (const auto& node : output.node()) {
    if (node.name() == "e") {
      // The d node isn't swappable.
      EXPECT_EQ(5, node.input_size());
      EXPECT_EQ("d", node.input(2));
      EXPECT_EQ("^swap_out_d_2", node.input(4));
    }
  }

#if GOOGLE_CUDA
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
#endif
}

TEST_F(MemoryOptimizerTest, AccumulationRewrites) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::RandomNormal(s.WithOpName("a").WithDevice("/cpu:0"),
                               {128, 128, 8}, DT_FLOAT);
  Output b = ops::RandomNormal(s.WithOpName("b").WithDevice("/cpu:0"),
                               {128, 128, 8}, DT_FLOAT);
  Output c = ops::RandomNormal(s.WithOpName("c").WithDevice("/cpu:0"),
                               {128, 128, 8}, DT_FLOAT);
  Output d = ops::AddN(s.WithOpName("d").WithDevice("/cpu:0"), {a, b, c});
  Output e = ops::Square(s.WithOpName("e").WithDevice("/cpu:0"), d);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"e"};

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());
  MemoryOptimizer optimizer(RewriterConfig::SCHEDULING_HEURISTICS);
  GraphDef output;
  Status status = optimizer.Optimize(cluster.get(), item, &output);
  TF_EXPECT_OK(status);

  int count = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "d") {
      EXPECT_EQ("DestroyTemporaryVariable", node.op());
      count++;
    } else if (node.name() == "d/tmp_var_initializer") {
      EXPECT_EQ("Assign", node.op());
      count++;
    } else if (node.name() == "d/tmp_var") {
      EXPECT_EQ("TemporaryVariable", node.op());
      count++;
    } else if (node.name() == "e") {
      EXPECT_EQ("Square", node.op());
      EXPECT_EQ("d", node.input(0));
      count++;
    }
  }
  EXPECT_EQ(4, count);

  std::vector<string> fetch = {"a", "b", "c", "e"};
  auto tensors = EvaluateNodes(output, fetch, {});
  EXPECT_EQ(4, tensors.size());

  for (int i = 0; i < tensors[0].NumElements(); ++i) {
    float actual = tensors[3].flat<float>()(i);
    float expected = 0.0f;
    for (int j = 0; j < 3; ++j) {
      expected += tensors[j].flat<float>()(i);
    }
    expected *= expected;
    EXPECT_NEAR(actual, expected, 1e-4);
  }
}

class RelaxAllocatorConstraintsTest : public GrapplerTest {};

TEST_F(RelaxAllocatorConstraintsTest, SameDevice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output constant = ops::Const(s.WithOpName("constant").WithDevice("/cpu:0"),
                               -3.14f, {128, 128});
  Output variable = ops::Variable(s.WithOpName("variable").WithDevice("/cpu:0"),
                                  {128, 128}, DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign").WithDevice("/cpu:0"),
                              variable, constant);
  Output exp = ops::Exp(s.WithOpName("exp").WithDevice("/cpu:0"), assign);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  auto node = output.node(2);
  EXPECT_EQ("assign", node.name());
  EXPECT_EQ(1, node.attr().count("_grappler_relax_allocator_constraints"));
  EXPECT_EQ(true, node.attr().at("_grappler_relax_allocator_constraints").b());

  item.fetch = {"exp"};
  item.init_ops = {"variable"};
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(RelaxAllocatorConstraintsTest, DifferentDevice) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output constant = ops::Const(s.WithOpName("constant").WithDevice("/cpu:0"),
                               -3.14f, {128, 128});
  Output variable = ops::Variable(s.WithOpName("variable").WithDevice("/cpu:0"),
                                  {128, 128}, DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign").WithDevice("/cpu:0"),
                              variable, constant);
  // exp runs on a different device, so we cannot relax the allocation
  // constraints on assign.
  Output exp = ops::Exp(s.WithOpName("exp").WithDevice("/gpu:0"), assign);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  auto node = output.node(2);
  EXPECT_EQ("assign", node.name());
  EXPECT_EQ(0, node.attr().count("_grappler_relax_allocator_constraints"));
#if GOOGLE_CUDA
  item.fetch = {"exp"};
  item.init_ops = {"variable"};
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
#endif
}

TEST_F(RelaxAllocatorConstraintsTest, SameDeviceType) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output constant = ops::Const(s.WithOpName("constant").WithDevice("/cpu:0"),
                               -3.14f, {128, 128});
  Output variable = ops::Variable(s.WithOpName("variable").WithDevice("/cpu:0"),
                                  {128, 128}, DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign").WithDevice("/cpu:0"),
                              variable, constant);
  // Assign and Exp run on different devies, but do not straddle a CPU:GPU
  // boundary, so we can we do not need to enforce allocation in pinned memory.
  Output exp = ops::Exp(s.WithOpName("exp").WithDevice("/cpu:1"), assign);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  auto node = output.node(2);
  EXPECT_EQ("assign", node.name());
  EXPECT_EQ(1, node.attr().count("_grappler_relax_allocator_constraints"));
  EXPECT_TRUE(node.attr().at("_grappler_relax_allocator_constraints").b());
}

TEST_F(RelaxAllocatorConstraintsTest, SendNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output constant = ops::Const(s.WithOpName("constant").WithDevice("/cpu:0"),
                               -3.14f, {128, 128});
  Output variable = ops::Variable(s.WithOpName("variable").WithDevice("/cpu:0"),
                                  {128, 128}, DT_FLOAT);
  Output assign = ops::Assign(s.WithOpName("assign").WithDevice("/cpu:0"),
                              variable, constant);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  NodeDef* send = item.graph.add_node();
  // Add a send node to the graph in the fanout of "assign".
  send->set_name("send");
  send->set_op("_Send");
  send->add_input("assign");

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  auto node = output.node(2);
  EXPECT_EQ("assign", node.name());
  EXPECT_EQ(0, node.attr().count("_grappler_relax_allocator_constraints"));
}

TEST_F(RelaxAllocatorConstraintsTest, AssignNodeInFanout) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output constant0 = ops::Const(s.WithOpName("constant0").WithDevice("/cpu:0"),
                                -42.0f, {128, 128});
  Output variable0 = ops::Variable(
      s.WithOpName("variable0").WithDevice("/cpu:0"), {128, 128}, DT_FLOAT);
  Output assign0 = ops::Assign(s.WithOpName("assign0").WithDevice("/cpu:0"),
                               variable0, constant0);
  // The rest of the graph is on a second device, so we can relax the
  // constraint for assign1, but not for assign0.
  Output exp1 = ops::Exp(s.WithOpName("exp1").WithDevice("/gpu:0"), assign0);
  Output variable1 = ops::Variable(
      s.WithOpName("variable1").WithDevice("/gpu:0"), {128, 128}, DT_FLOAT);
  Output assign1 = ops::Assign(s.WithOpName("assign1").WithDevice("/gpu:0"),
                               variable1, exp1);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  TF_EXPECT_OK(optimizer.Optimize(nullptr, item, &output));

  auto node = output.node(3);
  EXPECT_EQ("assign0", node.name());
  EXPECT_EQ(0, node.attr().count("_grappler_relax_allocator_constraints"));

  node = output.node(5);
  EXPECT_EQ("assign1", node.name());
  EXPECT_EQ(1, node.attr().count("_grappler_relax_allocator_constraints"));
  EXPECT_EQ(true, node.attr().at("_grappler_relax_allocator_constraints").b());

#if GOOGLE_CUDA
  item.fetch = {"assign0", "assign1"};
  item.init_ops = {"exp1", "variable1"};
  auto tensors_expected = EvaluateFetchNodes(item);
  GrapplerItem optimized = item.WithGraph(std::move(output));
  auto tensors = EvaluateFetchNodes(optimized);
  for (int i = 0; i < tensors_expected.size(); ++i) {
    test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
  }
#endif
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
