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

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"  // NOLINT
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
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
  // an new ScopedAllocator, then replace a1 and a2 with a3 that
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
  // returns the outputs specifed by `output_names` in `outputs`.
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
};

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
  NodeDef* nd = node_map.GetNode("scoped_allocator_1_1");
  ASSERT_TRUE(nd);
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
               /*output_names=*/{"r1:0", "r2:0", "scoped_allocator_1_2_Abs:0"},
               &outputs);
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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
