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
#include "tensorflow/core/grappler/optimizers/scoped_allocator_optimizer.h"

#include <unordered_set>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace grappler {
namespace {

class ScopedAllocatorOptimizerTest : public ::testing::Test {
 public:
  std::unique_ptr<Session> CreateSession(const GraphDef& graph,
                                         const ConfigProto& config) {
    SessionOptions options;
    options.config = config;
    (*options.config.mutable_device_count())["CPU"] = 2;
    Session* session = NewSession(options);
    TF_CHECK_OK(session->Create(graph));
    return std::unique_ptr<Session>(session);
  }

  std::vector<Tensor> EvaluateNodes(const GraphDef& graph,
                                    const std::vector<string>& fetch) {
    SessionOptions options;
    std::unique_ptr<Session> session(NewSession(options));
    TF_CHECK_OK(session->Create(graph));
    RunOptions run_options;
    std::vector<Tensor> output_tensors;
    TF_CHECK_OK(
        session->Run(run_options, {}, fetch, fetch, &output_tensors, nullptr));
    TF_CHECK_OK(session->Close());
    return output_tensors;
  }

  // Constructs the following graph.
  // (Flow is top to bottom, like nature intends.)
  //
  // The intended optimization is to have s1 and s2 allocate from
  // a new ScopedAllocator, then replace a1 and a2 with a3 that
  // reads from the backing buffer.
  /*
        a    b    c
         \  / \  /
          s1   s2
          |    |
         (i1) (i2)  if forward is true
          |    |
          a1   a2
          |    |
          r1   r2
  */
  void BuildAbsGraph(GraphDef* graph_def, bool forward) {
    Scope s = Scope::NewRootScope();
    s = s.WithDevice("/job:localhost/replica:0/task:0/device:CPU:0");

    Output a =
        ops::Const<float>(s.WithOpName("a"), {1.0, 0.0, 0.0, -1.0}, {2, 2});
    Output b =
        ops::Const<float>(s.WithOpName("b"), {1.0, -2.0, 3.0, 4.0}, {2, 2});
    Output c =
        ops::Const<float>(s.WithOpName("c"), {-5.0, -2.0, 0.0, -2.0}, {2, 2});
    Output s1 = ops::Add(s.WithOpName("s1"), a, b);
    Output s2 = ops::Add(s.WithOpName("s2"), b, c);
    Output int1, int2;
    if (forward) {
      int1 = ops::Identity(s.WithOpName("i1"), s1);
      int2 = ops::Identity(s.WithOpName("i2"), s2);
    } else {
      int1 = s1;
      int2 = s2;
    }
    Output a1 = ops::Abs(s.WithOpName("a1"), int1);
    Output a2 = ops::Abs(s.WithOpName("a2"), int2);
    Output r1 = ops::Reshape(s.WithOpName("r1"), a1, {1, 4});
    Output r2 = ops::Reshape(s.WithOpName("r2"), a2, {4, 1});
    TF_CHECK_OK(s.ToGraphDef(graph_def));
  }

  // Constructs the following graph.
  // (Flow is top to bottom, like nature intends.)
  //
  // a, b, and c are placeholders.  s is an Add op.  a1, a2, and a3 are Abs ops.
  // r1, r2, and r3 are Reshape ops.
  //
  // After this graph undergoes SA optimization, we expect a, b, and s to be
  // allocated from a new ScopedAllocator.  There will be control edges from the
  // ScopedAllocator node to a, b, and s, to ensure that we allocate the
  // backing tensor before we need it.  There will also be a control edge from c
  // to ScopedAllocator node, so that we delay allocation as much as possible.
  // There should be no edge from b to ScopedAllocator node, because that would
  // imply a cycle in the graph.
  /*
      a      b     c
      |     / \   /
      |    /   \ /
      |    |    s1
      |    |    |
      a1   a2   a3
      |    |    |
      r1   r2   r3
  */
  void BuildAbsGraphWithInputDependencies(GraphDef* graph_def) {
    Scope s = Scope::NewRootScope();
    s = s.WithDevice("/job:localhost/replica:0/task:0/device:CPU:0");

    Output a = ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                                ops::Placeholder::Shape({2, 2}));
    Output b = ops::Placeholder(s.WithOpName("b"), DT_FLOAT,
                                ops::Placeholder::Shape({2, 2}));
    Output c = ops::Placeholder(s.WithOpName("c"), DT_FLOAT,
                                ops::Placeholder::Shape({2, 2}));
    Output s1 = ops::Add(s.WithOpName("s1"), b, c);
    Output a1 = ops::Abs(s.WithOpName("a1"), a);
    Output a2 = ops::Abs(s.WithOpName("a2"), b);
    Output a3 = ops::Abs(s.WithOpName("a3"), s1);
    Output r1 = ops::Reshape(s.WithOpName("r1"), a1, {1, 4});
    Output r2 = ops::Reshape(s.WithOpName("r2"), a2, {4, 1});
    Output r3 = ops::Reshape(s.WithOpName("r3"), a3, {4, 1});
    TF_CHECK_OK(s.ToGraphDef(graph_def));
  }

  // Constructs the following graph.
  //
  // a and b are data inputs.  ctl1 and ctl2 are control inputs.  a1 and a2 are
  // Abs ops.  o1 and o2 are data outputs.  a1 -> ctl3 and a2 -> ctl4 are
  // control edges.
  //
  // After the optimizer runs, we expect the ctl1 and ctl2 to be connected to
  // the SAConcat node, and ctl3 and ctl4 to be connected to SASplit node.
  /*
     a  ctl1   b  ctl2
      \  /      \  /
       a1        a2
      /  \      /  \
     o1  ctl3  o2   ctl4
  */
  void BuildAbsGraphWithInputAndOutputControlEdges(GraphDef* graph_def) {
    Scope s = Scope::NewRootScope();
    s = s.WithDevice("/job:localhost/replica:0/task:0/device:CPU:0");

    Output a = ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                                ops::Placeholder::Shape({2, 2}));
    Output b = ops::Placeholder(s.WithOpName("b"), DT_FLOAT,
                                ops::Placeholder::Shape({2, 2}));
    Output ctl1 = ops::Placeholder(s.WithOpName("ctl1"), DT_FLOAT,
                                   ops::Placeholder::Shape({2, 2}));
    Output ctl2 = ops::Placeholder(s.WithOpName("ctl2"), DT_FLOAT,
                                   ops::Placeholder::Shape({2, 2}));
    Output a1 = ops::Abs(s.WithOpName("a1").WithControlDependencies({ctl1}), a);
    Output a2 = ops::Abs(s.WithOpName("a2").WithControlDependencies({ctl2}), b);
    Output o1 = ops::Reshape(s.WithOpName("o1"), a1, {1, 4});
    Output o2 = ops::Reshape(s.WithOpName("o2"), a2, {4, 1});
    Output ctl3 =
        ops::Const<float>(s.WithOpName("ctl3").WithControlDependencies({a1}),
                          {0.0, 0.0, 0.0, 0.0}, {2, 2});
    Output ctl4 =
        ops::Const<float>(s.WithOpName("ctl4").WithControlDependencies({a2}),
                          {0.0, 0.0, 0.0, 0.0}, {2, 2});
    TF_CHECK_OK(s.ToGraphDef(graph_def));
  }

  // Constructs the following graph.
  //
  // We have 2 different name scopes in this graph.  s3, a3, a4, r3, and r4 are
  // all under "sub" scope.  All other nodes are in the root scope.
  //
  // The intention is to test that ScopedAllocatorOptimizer works well with a
  // graph that has multiple name scopes.  In particular, it should work when a
  // node (in this case s2) is an input to two nodes in different name scopes
  // (a2 and sub/a3) which may be scope allocated.
  /*
        a    b    c         a    b
         \  / \  /           \  /
          s1   s2------      sub/s3
          |    |      |        |
          a1   a2   sub/a4   sub/a3
          |    |      |        |
          r1   r2   sub/r4   sub/r3
  */
  void BuildGraphWithMultipleScopes(GraphDef* graph_def) {
    Scope root_scope = Scope::NewRootScope();
    root_scope =
        root_scope.WithDevice("/job:localhost/replica:0/task:0/device:CPU:0");

    Output a = ops::Const<float>(root_scope.WithOpName("a"),
                                 {1.0, 0.0, 0.0, -1.0}, {2, 2});
    Output b = ops::Const<float>(root_scope.WithOpName("b"),
                                 {1.0, -2.0, 3.0, 4.0}, {2, 2});
    Output c = ops::Const<float>(root_scope.WithOpName("c"),
                                 {-5.0, -2.0, 0.0, -2.0}, {2, 2});

    // Root scope ops.
    Output s1 = ops::Add(root_scope.WithOpName("s1"), a, b);
    Output s2 = ops::Add(root_scope.WithOpName("s2"), b, c);
    Output a1 = ops::Abs(root_scope.WithOpName("a1"), s1);
    Output a2 = ops::Abs(root_scope.WithOpName("a2"), s2);
    Output r1 = ops::Reshape(root_scope.WithOpName("r1"), a1, {1, 4});
    Output r2 = ops::Reshape(root_scope.WithOpName("r2"), a2, {4, 1});

    // Sub scope ops.
    Scope sub_scope = root_scope.NewSubScope("sub");
    Output s3 = ops::Add(sub_scope.WithOpName("s3"), a, b);
    Output a3 = ops::Abs(sub_scope.WithOpName("a3"), s3);
    Output a4 = ops::Abs(sub_scope.WithOpName("a4"), s2);
    Output r3 = ops::Reshape(sub_scope.WithOpName("r3"), a3, {1, 4});
    Output r4 = ops::Reshape(sub_scope.WithOpName("r4"), a4, {4, 1});

    TF_CHECK_OK(root_scope.ToGraphDef(graph_def));
  }

  // Constructs the following graph.
  //
  // c1 and c2 are Const ops.  a1 and a2 are Abs ops.
  // We expect the optimizer to succeed and insert Identity between ci and ai.
  // This will ensure that we will still be able use ScopedAllocator with Const
  // inputs.
  /*
          c1   c2
          |    |
          a1   a2
          |    |
          r1   r2
  */
  void BuildConstGraph(GraphDef* graph_def, bool forward) {
    Scope s = Scope::NewRootScope();
    s = s.WithDevice("/job:localhost/replica:0/task:0/device:CPU:0");

    Output c1 =
        ops::Const<float>(s.WithOpName("c1"), {1.0, 0.0, 0.0, -1.0}, {2, 2});
    Output c2 =
        ops::Const<float>(s.WithOpName("c2"), {1.0, -2.0, 3.0, 4.0}, {2, 2});
    Output a1 = ops::Abs(s.WithOpName("a1"), c1);
    Output a2 = ops::Abs(s.WithOpName("a2"), c2);
    Output r1 = ops::Reshape(s.WithOpName("r1"), a1, {1, 4});
    Output r2 = ops::Reshape(s.WithOpName("r2"), a2, {4, 1});
    TF_CHECK_OK(s.ToGraphDef(graph_def));
  }

  void SetShapes(GraphDef* graph_def) {
    TensorShapeProto shape_proto;
    shape_proto.add_dim()->set_size(2);
    shape_proto.add_dim()->set_size(2);

    for (NodeDef& n : *graph_def->mutable_node()) {
      if (n.op() == "Add" || n.op() == "Abs") {
        AddNodeAttr("_output_shapes", {shape_proto}, &n);
      }
    }
  }

  // Invokes ScopedAllocatorOptimizer on `graph_def`, then executes it and
  // returns the outputs specified by `output_names` in `outputs`.
  void ExecuteGraph(const GraphDef& graph_def,
                    const std::vector<string>& output_names,
                    std::vector<Tensor>* outputs) {
    // Turn off all optimization except the ScopedAllocatorOptimizer
    // to avoid anything that would alter the expected graph input/output,
    // e.g. by constant folding away all calculations.
    ConfigProto config;
    GraphOptions* gopt = config.mutable_graph_options();
    OptimizerOptions* opts = gopt->mutable_optimizer_options();
    opts->set_do_common_subexpression_elimination(false);
    opts->set_do_constant_folding(false);
    opts->set_do_function_inlining(false);
    opts->set_opt_level(OptimizerOptions::L0);
    RewriterConfig* rwcfg = gopt->mutable_rewrite_options();
    rwcfg->clear_optimizers();
    (*rwcfg->add_optimizers()) = "scoped_allocator";
    rwcfg->mutable_scoped_allocator_opts()->add_enable_op("Abs");
    std::unique_ptr<Session> session(CreateSession(graph_def, config));

    std::vector<std::pair<string, Tensor>> inputs;
    std::vector<string> target_nodes = {};
    Status s = session->Run(inputs, output_names, target_nodes, outputs);
    TF_ASSERT_OK(s);
    ASSERT_EQ(outputs->size(), output_names.size());
  }

  // Validates that outputs match expected.
  void ValidateValues(const std::vector<Tensor>& outputs,
                      const std::vector<std::vector<float>>& expected) {
    for (int i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i].size(), outputs[i].NumElements());
      for (int j = 0; j < expected[i].size(); ++j) {
        EXPECT_EQ(expected[i][j], outputs[i].flat<float>()(j));
      }
    }
  }

  void GetNode(NodeMap* node_map, const string& node_name, NodeDef** node_def) {
    *node_def = node_map->GetNode(node_name);
    ASSERT_TRUE(*node_def);
  }

  // Validate that a node has a single control input from scoped allocator node.
  // Return the scoped allocator node.
  NodeDef* ValidateSAControlInput(GraphDef* graph, NodeMap* node_map,
                                  const string& node_name) {
    NodeDef* node = nullptr;
    GetNode(node_map, node_name, &node);
    int num_control_inputs = 0;
    string control_input_name;
    for (const auto& input : node->input()) {
      if (IsControlInput(input)) {
        ++num_control_inputs;
        control_input_name = input;
      }
    }
    EXPECT_EQ(num_control_inputs, 1);
    NodeDef* control_input_node = nullptr;
    GetNode(node_map, control_input_name, &control_input_node);
    EXPECT_EQ(control_input_node->op(), "_ScopedAllocator");
    return control_input_node;
  }

  int NumControlInputs(NodeMap* node_map, const string& node_name) {
    NodeDef* node = nullptr;
    GetNode(node_map, node_name, &node);
    int num_control_inputs = 0;
    for (const auto& input : node->input()) {
      if (IsControlInput(input)) {
        ++num_control_inputs;
      }
    }
    return num_control_inputs;
  }
};
#ifndef ENABLE_MKL

TEST_F(ScopedAllocatorOptimizerTest, UnaryRewriteOnly) {
  // Tests that Rewrite of program with parallel unary Ops is done as
  // anticipated.
  GrapplerItem item;
  BuildAbsGraph(&item.graph, false);
  SetShapes(&item.graph);

  ScopedAllocatorOptions opts;
  opts.add_enable_op("Abs");
  ScopedAllocatorOptimizer sao(RewriterConfig::ON, opts);
  ScopedAllocatorOptimizer::OpNameSet ons;
  ons.insert("Abs");

  GraphDef optimized_graph;
  TF_ASSERT_OK(sao.Optimize(nullptr /*cluster*/, item, &optimized_graph));

  // Examine the resulting graph def.
  NodeMap node_map(&optimized_graph);
  NodeDef* nd = nullptr;
  GetNode(&node_map, "scoped_allocator_1_1", &nd);
  {
    auto& nd_set = node_map.GetOutputs(nd->name());
    ASSERT_EQ(3, nd_set.size());
    std::unordered_set<string> expected = {"scoped_allocator_concat_1_1", "s1",
                                           "s2"};
    for (auto it : nd_set) {
      ASSERT_NE(expected.find(it->name()), expected.end())
          << "Failed to find " << it->name();
    }
  }
  {
    auto& nd_set = node_map.GetOutputs("scoped_allocator_concat_1_1");
    ASSERT_EQ(1, nd_set.size());
    for (auto it : nd_set) {
      ASSERT_EQ("scoped_allocator_1_1_Abs", it->name());
    }
  }
  {
    auto& nd_set = node_map.GetOutputs("scoped_allocator_1_1_Abs");
    ASSERT_EQ(1, nd_set.size());
    for (auto it : nd_set) {
      ASSERT_EQ("scoped_allocator_split_1_1", it->name());
    }
  }
  {
    auto& nd_set = node_map.GetOutputs("scoped_allocator_split_1_1");
    ASSERT_EQ(2, nd_set.size());
    std::unordered_set<string> name_set;
    for (auto it : nd_set) {
      name_set.insert(it->name());
    }
    ASSERT_TRUE(name_set.find("r1") != name_set.end());
    ASSERT_TRUE(name_set.find("r2") != name_set.end());
  }
}

TEST_F(ScopedAllocatorOptimizerTest, UnaryExecute) {
  // Builds the same graph as UnaryRewriteOnly but also executes it and
  // validates the output.
  GraphDef graph_def;
  BuildAbsGraph(&graph_def, /*forward=*/false);
  SetShapes(&graph_def);
  std::vector<Tensor> outputs;
  ExecuteGraph(graph_def,
               /*output_names=*/{"r1:0", "r2:0"}, &outputs);
  // a + b == 2, -2, 3, 3
  // b + c == -4, -4, 3, 2
  ValidateValues(outputs, /*expected=*/{{2, 2, 3, 3}, {4, 4, 3, 2}});
}

TEST_F(ScopedAllocatorOptimizerTest, MultipleScopes) {
  GraphDef graph_def;
  BuildGraphWithMultipleScopes(&graph_def);
  SetShapes(&graph_def);
  std::vector<Tensor> outputs;
  ExecuteGraph(graph_def,
               /*output_names=*/{"r1:0", "r2:0", "sub/r3:0", "sub/r4:0"},
               &outputs);
  ValidateValues(
      outputs,
      /*expected=*/{{2, 2, 3, 3}, {4, 4, 3, 2}, {2, 2, 3, 3}, {4, 4, 3, 2}});
}

// Tests static ScopedAllocatorOptimizer::ExtendNodeAttr.
// Maybe this should be moved elsewhere?
TEST_F(ScopedAllocatorOptimizerTest, Extend) {
  NodeDef nd;
  ScopedAllocatorOptimizer::ExtendNodeAttr("_scoped_allocator", {0, 2}, &nd);
  ScopedAllocatorOptimizer::ExtendNodeAttr("_scoped_allocator", {6, 7}, &nd);
  ScopedAllocatorOptimizer::ExtendNodeAttr("_scoped_allocator", {2, 3}, &nd);
  VLOG(0) << "nd: " << nd.DebugString();
  std::vector<int> scoped_allocator_attrs;
  AttrSlice slice(nd);
  Status sa_status =
      GetNodeAttr(slice, "_scoped_allocator", &scoped_allocator_attrs);
  for (int i : scoped_allocator_attrs) {
    VLOG(0) << "extracted: " << i;
  }
  NodeDef nd2;
  AddNodeAttr("_scoped_allocator", {0, 2}, &nd2);
  AddNodeAttr("_scoped_allocator", {6, 7}, &nd2);
  AddNodeAttr("_scoped_allocator", {2, 3}, &nd2);
  VLOG(0) << "nd2: " << nd2.DebugString();
}

TEST_F(ScopedAllocatorOptimizerTest, ForwardInputToOutput) {
  // Test that kernels that forward the input to output using `set_output` work
  // well with scoped allocator optimization.
  GraphDef graph_def;
  BuildAbsGraph(&graph_def, /*forward=*/true);
  SetShapes(&graph_def);
  std::vector<Tensor> outputs;
  ExecuteGraph(graph_def, /*output_names=*/{"r1:0", "r2:0"}, &outputs);
  // a + b == 2, -2, 3, 3
  // b + c == -4, -4, 3, 2
  ValidateValues(outputs, /*expected=*/{{2, 2, 3, 3}, {4, 4, 3, 2}});
}

// Test that graphs with a dependency upstream from the inputs, such as the one
// produced by `BuildAbsGraphWithInputDependencies`, are handled well by this
// optimizer.  In particular, the optimizer should not create cycles.
TEST_F(ScopedAllocatorOptimizerTest, InputDependencies) {
  GrapplerItem item;
  BuildAbsGraphWithInputDependencies(&item.graph);
  SetShapes(&item.graph);

  ScopedAllocatorOptions opts;
  opts.add_enable_op("Abs");
  ScopedAllocatorOptimizer sao(RewriterConfig::ON, opts);
  ScopedAllocatorOptimizer::OpNameSet ons;
  ons.insert("Add");

  GraphDef optimized_graph;
  TF_ASSERT_OK(sao.Optimize(/*cluster=*/nullptr, item, &optimized_graph));
  NodeMap node_map(&optimized_graph);

  // Check that all inputs to Abs ops have ScopedAllocator as a control
  // dependency.
  NodeDef* scoped_allocator_node =
      ValidateSAControlInput(&optimized_graph, &node_map, "a");
  VLOG(1) << scoped_allocator_node->DebugString();
  EXPECT_TRUE(ValidateSAControlInput(&optimized_graph, &node_map, "b"));
  EXPECT_TRUE(ValidateSAControlInput(&optimized_graph, &node_map, "s1"));

  // Check that ScopedAllocator node has a single input, which is a control edge
  // from c.
  EXPECT_EQ(scoped_allocator_node->input_size(), 1);
  EXPECT_EQ(scoped_allocator_node->input(0), "^c");
}

// Test that graphs with input and output control edges are rewired correctly by
// the optimizer.
TEST_F(ScopedAllocatorOptimizerTest, ControlEdgeRewire) {
  GrapplerItem item;
  BuildAbsGraphWithInputAndOutputControlEdges(&item.graph);
  SetShapes(&item.graph);
  LOG(INFO) << item.graph.DebugString();

  ScopedAllocatorOptions opts;
  opts.add_enable_op("Abs");
  ScopedAllocatorOptimizer sao(RewriterConfig::ON, opts);
  ScopedAllocatorOptimizer::OpNameSet ons;
  ons.insert("Const");

  GraphDef optimized_graph;
  TF_ASSERT_OK(sao.Optimize(/*cluster=*/nullptr, item, &optimized_graph));
  TF_ASSERT_OK(TopologicalSort(&optimized_graph));
  NodeMap node_map(&optimized_graph);
  LOG(INFO) << optimized_graph.DebugString();

  // Check that ctl1 and ctl2 are now connected only to SAConcat.
  NodeDef* ctl1 = nullptr;
  GetNode(&node_map, "ctl1", &ctl1);
  const auto& ctl1_outputs = node_map.GetOutputs("ctl1");
  EXPECT_EQ(ctl1_outputs.size(), 1);
  NodeDef* sa_concat = *ctl1_outputs.begin();
  EXPECT_EQ(sa_concat->op(), "_ScopedAllocatorConcat");
  NodeDef* ctl2 = nullptr;
  GetNode(&node_map, "ctl2", &ctl2);
  const auto& ctl2_outputs = node_map.GetOutputs("ctl2");
  EXPECT_EQ(ctl2_outputs.size(), 1);
  EXPECT_EQ(*ctl2_outputs.begin(), sa_concat);

  // Check that SAConcat has only 2 input control edges.
  EXPECT_EQ(NumControlInputs(&node_map, sa_concat->name()), 2);

  // Check that fused node, which conceptually used to have control inputs from
  // ctl1 and ctl2 respectively, no longer has any control inputs.
  const auto& sa_concat_outputs = node_map.GetOutputs(sa_concat->name());
  EXPECT_EQ(sa_concat_outputs.size(), 1);
  NodeDef* fused_abs = *sa_concat_outputs.begin();
  EXPECT_EQ(NumControlInputs(&node_map, fused_abs->name()), 0);

  // Check that SASplit node has control edges to ctl3, ctl4; also check that
  // those are the only control inputs on ctl3 and ctl4.
  const auto& fused_abs_outputs = node_map.GetOutputs(fused_abs->name());
  EXPECT_EQ(fused_abs_outputs.size(), 1);
  NodeDef* sa_split = *fused_abs_outputs.begin();
  EXPECT_EQ(NumControlOutputs(*sa_split, node_map), 2);
  EXPECT_EQ(NumControlInputs(&node_map, "ctl3"), 1);
  EXPECT_EQ(NumControlInputs(&node_map, "ctl4"), 1);
}

// Test that the optimization succeeds when any input is a Const op, and that it
// inserts Identity op between Const and Abs.
TEST_F(ScopedAllocatorOptimizerTest, ConstInput) {
  GrapplerItem item;
  BuildConstGraph(&item.graph, false);
  SetShapes(&item.graph);

  ScopedAllocatorOptions opts;
  opts.add_enable_op("Abs");
  ScopedAllocatorOptimizer sao(RewriterConfig::ON, opts);
  ScopedAllocatorOptimizer::OpNameSet ons;
  ons.insert("Abs");

  GraphDef optimized_graph;
  TF_ASSERT_OK(sao.Optimize(nullptr /*cluster*/, item, &optimized_graph));

  // Examine the resulting graphdef.
  const NodeDef* sa_node = nullptr;
  for (const NodeDef& node : optimized_graph.node()) {
    if (node.op() == "_ScopedAllocator") {
      sa_node = &node;
      break;
    }
  }
  ASSERT_NE(sa_node, nullptr);
  int num_identity_ops = 0;
  NodeMap node_map(&optimized_graph);
  for (NodeDef* sa_output : node_map.GetOutputs(sa_node->name())) {
    EXPECT_FALSE(IsConstant(*sa_output));
    if (IsIdentity(*sa_output)) {
      ++num_identity_ops;
    }
  }
  EXPECT_EQ(num_identity_ops, 2);
}
#endif  // ENABLE_MKL

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
