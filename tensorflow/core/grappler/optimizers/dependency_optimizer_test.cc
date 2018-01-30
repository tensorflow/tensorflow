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

#include "tensorflow/core/grappler/optimizers/dependency_optimizer.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class DependencyOptimizerTest : public ::testing::Test {};

void VerifyGraphsEqual(const GraphDef& original_graph,
                       const GraphDef& optimized_graph, const string& func) {
  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << func;
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = optimized_graph.node(i);
    EXPECT_EQ(original.name(), optimized.name()) << func;
    EXPECT_EQ(original.op(), optimized.op()) << func;
    EXPECT_EQ(original.input_size(), optimized.input_size()) << func;
    for (int j = 0; j < original.input_size(); ++j) {
      EXPECT_EQ(original.input(j), optimized.input(j)) << func;
    }
  }
}

TEST_F(DependencyOptimizerTest, NoOp) {
  // This trivial graph is so basic there's nothing to optimize.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  DependencyOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

TEST_F(DependencyOptimizerTest, DependenciesDrivenByConstants) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output y = ops::Const(s.WithOpName("y"), {1.0f, 2.0f}, {1, 2});
  Output z = ops::Const(s.WithOpName("z"), {1.0f, 2.0f}, {1, 2});
  Output add = ops::Add(s.WithOpName("add"), x, y);
  Output id1 =
      ops::Identity(s.WithOpName("id1").WithControlDependencies(x), add);
  Output id2 = ops::Identity(
      s.WithOpName("id2").WithControlDependencies(y).WithControlDependencies(z),
      add);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("id1");
  item.fetch.push_back("id2");

  DependencyOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // The 'z' node should have been optimized away leaving only 5 nodes.
  EXPECT_EQ(5, output.node_size());

  for (const NodeDef& node : item.graph.node()) {
    if (node.name() == "id1" || node.name() == "id2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("add", node.input(0));
    }
  }
}

TEST_F(DependencyOptimizerTest, ChangeToNoop) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::RandomUniform(s.WithOpName("x"), {1, 2}, DT_FLOAT);
  Output y = ops::RandomUniform(s.WithOpName("y"), {1, 2}, DT_FLOAT);
  Output add = ops::Add(s.WithOpName("add"), x, y);
  Output id1 =
      ops::Identity(s.WithOpName("id1").WithControlDependencies(add), x);
  Output id2 =
      ops::Identity(s.WithOpName("id2").WithControlDependencies(add), y);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("id1");
  item.fetch.push_back("id2");

  DependencyOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size(), output.node_size());
  for (int i = 0; i < item.graph.node_size(); ++i) {
    const NodeDef& node = item.graph.node(i);
    if (node.name() == "add") {
      EXPECT_EQ("NoOp", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^x", node.input(0));
      EXPECT_EQ("^y", node.input(1));
    } else if (node.name() == "id1") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^y", node.input(1));
    } else if (node.name() == "id2") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("^x", node.input(1));
    }
  }
}

TEST_F(DependencyOptimizerTest, ChangeToNoop_SwitchIdentity) {
  // This tests that we don't try to repeatedly add Identity nodes
  // with names like "ConstantFoldingCtrl/foo/bar/switch_$port" when
  // multiple nodes reading the same output of a Switch node get
  // optimized (e.g. constant folded or turned into NoOps).
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  ops::Variable v_in(scope.WithOpName("v_in"), {3}, DT_FLOAT);
  ops::Variable v_ctrl(scope.WithOpName("v_ctrl"), {}, DT_BOOL);
  ops::Switch s(scope.WithOpName("switch"), v_in, v_ctrl);
  // "neg" should be turned into a NoOp with a control dependency from
  // the existing Identity node "ConstantFoldingCtrl/switch_1" and
  // subsequently eliminated completely from the graph.
  Output neg = ops::Neg(scope.WithOpName("neg"), s.output_true);
  // c1 could be a result of constant folding some node fed by neg.
  Output c1 = ops::Const(scope.WithOpName("c1").WithControlDependencies(neg),
                         {1.0f, 2.0f}, {1, 2});
  Output ctrl_dep_id = ops::Identity(
      scope.WithOpName("ConstantFoldingCtrl/switch_1"), s.output_true);
  // c2 could be a result of constant folding a node fed by s, which also
  // added the ctrl_dep_id node.
  Output c2 =
      ops::Const(scope.WithOpName("c2").WithControlDependencies(ctrl_dep_id),
                 {1.0f, 2.0f}, {1, 2});
  Output neg1 = ops::Neg(scope.WithOpName("neg1"), s.output_false);
  Output neg2 = ops::Neg(scope.WithOpName("neg2"), ctrl_dep_id);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("c1");
  item.fetch.push_back("c2");
  item.fetch.push_back("neg1");
  item.fetch.push_back("neg2");

  DependencyOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size() - 1, output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    // "neg" should be eliminated.
    EXPECT_NE("neg", node.name());
    // A control dep from "^ConstantFoldingCtrl/switch_1"
    // should be attached to "c1".
    if (node.name() == "c1") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFoldingCtrl/switch_1", node.input(0));
    }
  }
}

// TODO(rmlarsen): Add test to make sure we skip Switch and Merge.
TEST_F(DependencyOptimizerTest, ChangeToNoop_NoFetch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::RandomUniform(s.WithOpName("x"), {1, 2}, DT_FLOAT);
  Output y = ops::RandomUniform(s.WithOpName("y"), {1, 2}, DT_FLOAT);
  Output add = ops::Add(s.WithOpName("add"), x, y);
  Output id1 =
      ops::Identity(s.WithOpName("id1").WithControlDependencies(add), x);
  Output id2 =
      ops::Identity(s.WithOpName("id2").WithControlDependencies(add), y);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  DependencyOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  TF_CHECK_OK(TopologicalSort(&item.graph));
  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

TEST_F(DependencyOptimizerTest, RemoveNoOps_EmptyInputOrOutput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::RandomUniform(s, {1, 2}, DT_FLOAT);
  auto noop1 = ops::NoOp(s);
  auto noop2 = ops::NoOp(s.WithControlDependencies(x));
  Output id = ops::Identity(s.WithControlDependencies({noop1.operation}), x);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("Identity");

  DependencyOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size(), output.node_size());
  for (const NodeDef& node : output.node()) {
    if (node.name() == "NoOp" || node.name() == "NoOp_1") {
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "Identity") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("RandomUniform", node.input(0));
    }
  }
}

TEST_F(DependencyOptimizerTest, RemoveNoOps_DeviceBoundaries) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::RandomUniform(s.WithOpName("x").WithDevice("/CPU:0"), {1, 2},
                                DT_FLOAT);
  Output y = ops::RandomUniform(s.WithOpName("y").WithDevice("/CPU:0"), {1, 2},
                                DT_FLOAT);
  // NoOp with a single input- and two output dependencies.
  auto noop = ops::NoOp(s.WithControlDependencies(x).WithDevice("/CPU:1"));
  // NoOp with a two input- and a single output dependency.
  auto noop_1 = ops::NoOp(
      s.WithControlDependencies(x).WithControlDependencies(y).WithDevice(
          "/CPU:0"));
  Output id = ops::Identity(
      s.WithControlDependencies({noop.operation}).WithDevice("/CPU:1"), x);
  Output id_1 = ops::Identity(
      s.WithControlDependencies({noop.operation, noop_1.operation})
          .WithDevice("/CPU:1"),
      y);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("Identity");
  item.fetch.push_back("Identity_1");

  DependencyOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // The optimization should be disabled to prevent increasing the number of
  // nodes crossing device boundaries.
  TF_CHECK_OK(TopologicalSort(&item.graph));
  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

TEST_F(DependencyOptimizerTest, RemoveNoOps_SingleInputOrOutput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::RandomUniform(s.WithOpName("x"), {1, 2}, DT_FLOAT);
  Output y = ops::RandomUniform(s.WithOpName("y"), {1, 2}, DT_FLOAT);
  // NoOp with a single input- and two output dependencies.
  auto noop = ops::NoOp(s.WithControlDependencies(x));
  // NoOp with a two input- and a single output dependency.
  auto noop_1 =
      ops::NoOp(s.WithControlDependencies(x).WithControlDependencies(y));
  Output id = ops::Identity(s.WithControlDependencies({noop.operation}), x);
  Output id_1 = ops::Identity(
      s.WithControlDependencies({noop.operation, noop_1.operation}), y);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("Identity");
  item.fetch.push_back("Identity_1");

  DependencyOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size(), output.node_size());
  for (const NodeDef& node : output.node()) {
    if (node.name() == "NoOp" || node.name() == "NoOp_1") {
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "Identity") {
      EXPECT_EQ("x", node.input(0));
    } else if (node.name() == "Identity_1") {
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("^x", node.input(1));
    }
  }
}

TEST_F(DependencyOptimizerTest, RemoveIdentity) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::RandomUniform(s.WithOpName("x"), {1, 2}, DT_FLOAT);
  Output y = ops::RandomUniform(s.WithOpName("y"), {1, 2}, DT_FLOAT);
  Output z = ops::RandomUniform(s.WithOpName("z"), {1, 2}, DT_FLOAT);

  // Identity nodes to be removed.
  // Case a) with a single input- and multiple outputs.
  auto id_a = ops::Identity(s.WithOpName("id_a"), x);
  // Case b) with multiple inputs and a single output.
  auto id_b = ops::Identity(
      s.WithOpName("id_b").WithControlDependencies(y).WithControlDependencies(
          z),
      x);
  // Case c) with two inputs and two outputs.
  auto id_c = ops::Identity(s.WithOpName("id_c").WithControlDependencies(y), x);

  // Output for Case a.
  Output a_a = ops::Identity(s.WithOpName("a_a"), id_a);
  Output a_b = ops::Identity(s.WithOpName("a_b"), id_a);
  Output a_c =
      ops::Identity(s.WithOpName("a_c").WithControlDependencies(id_a), z);
  Output a_d =
      ops::Identity(s.WithOpName("a_d").WithControlDependencies(id_a), z);
  // Output for Case b.
  Output b_a = ops::Identity(s.WithOpName("b_a"), id_b);
  // Output for Case c.
  Output c_a = ops::Identity(s.WithOpName("c_a"), id_c);
  Output c_b =
      ops::Identity(s.WithOpName("c_b").WithControlDependencies(id_c), z);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"a_a", "a_b", "a_c", "a_d", "b_a", "c_a", "c_b"};

  DependencyOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size() - 3, output.node_size());
  for (const NodeDef& node : output.node()) {
    EXPECT_NE("id_a", node.name());
    EXPECT_NE("id_b", node.name());
    EXPECT_NE("id_c", node.name());
    if (node.name() == "a_a" || node.name() == "a_b") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));
    }
    if (node.name() == "a_c" || node.name() == "a_d") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("z", node.input(0));
      EXPECT_EQ("^x", node.input(1));
    }
    if (node.name() == "b_a") {
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^y", node.input(1));
      EXPECT_EQ("^z", node.input(2));
    }
    if (node.name() == "c_a") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^y", node.input(1));
    }
    if (node.name() == "c_b") {
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("z", node.input(0));
      EXPECT_EQ("^x", node.input(1));
      EXPECT_EQ("^y", node.input(2));
    }
  }
}

TEST_F(DependencyOptimizerTest, RemoveIdentity_RepeatedInputs) {
  // Corner cases with repeated inputs.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  ops::Variable x(scope.WithOpName("x"), {}, DT_BOOL);
  ops::Variable y(scope.WithOpName("y"), {}, DT_BOOL);
  ops::Switch sw(scope.WithOpName("switch"), x, x);
  // id0 should be removed.
  Output id0 = ops::Identity(scope.WithOpName("id0"), sw.output_true);
  // id1 should not be removed, since it would anchor a control dependency
  // on the switch.
  Output id1 = ops::Identity(scope.WithOpName("id1"), sw.output_false);
  Output or0 = ops::LogicalOr(scope.WithOpName("or0"), id0, id0);
  Output or1 = ops::LogicalOr(scope.WithOpName("or1"), id0, y);
  Output or2 = ops::LogicalOr(
      scope.WithOpName("or2").WithControlDependencies(id1), y, y);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("or0");
  item.fetch.push_back("or1");
  item.fetch.push_back("or2");
  DependencyOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size() - 1, output.node_size());
  for (const NodeDef& node : output.node()) {
    EXPECT_NE("id0", node.name());
    if (node.name() == "or0") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("switch:1", node.input(0));
      EXPECT_EQ("switch:1", node.input(1));
    }
    if (node.name() == "or1") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("switch:1", node.input(0));
      EXPECT_EQ("y", node.input(1));
    }
    if (node.name() == "or2") {
      // or1 should be unchanged.
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^id1", node.input(2));
    }
  }
}

TEST_F(DependencyOptimizerTest, Transitive_Reduction_Simple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output x = ops::Square(s.WithOpName("x"), c);
  Output neg1 = ops::Neg(s.WithOpName("neg1"), x);
  Output neg2 =
      ops::Neg(s.WithOpName("neg2").WithControlDependencies({x}), neg1);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("neg2");
  DependencyOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(4, output.node_size());
  EXPECT_EQ("neg2", output.node(3).name());
  EXPECT_EQ(1, output.node(3).input_size());
  EXPECT_EQ("neg1", output.node(3).input(0));
}

TEST_F(DependencyOptimizerTest, ChangeToNoop_Identity) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  ops::Variable v_in(scope.WithOpName("v_in"), {3}, DT_FLOAT);
  Output id_after_var = ops::Identity(scope.WithOpName("id_after_var"), v_in);
  ops::Variable v_ctrl(scope.WithOpName("v_ctrl"), {}, DT_BOOL);
  ops::Switch s(
      scope.WithOpName("switch").WithControlDependencies(id_after_var), v_in,
      v_ctrl);
  Output id0 = ops::Identity(scope.WithOpName("id0"), s.output_true);
  Output grappler_added_id = ops::Identity(
      scope.WithOpName("ConstantFoldingCtrl/switch_1"), s.output_true);
  Output c1 = ops::Const(scope.WithOpName("c1")
                             .WithControlDependencies(id_after_var)
                             .WithControlDependencies(grappler_added_id),
                         {1.0f, 2.0f}, {1, 2});
  Output id1 = ops::Identity(scope.WithOpName("id1"), c1);
  Output id2 = ops::Identity(scope.WithOpName("id2"), id0);
  Output fetch =
      ops::Identity(scope.WithOpName("fetch").WithControlDependencies(id1), c1);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("c1");
  item.fetch.push_back("id2");
  item.fetch.push_back("fetch");

  DependencyOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size() - 2, output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    // "id0" and "id1" but neither "ConstantFoldingCtrl/switch_1",
    // "id_after_var, nor "id2"" should be eliminated.
    EXPECT_NE("id0", node.name());
    EXPECT_NE("id1", node.name());
    if (node.name() == "c1") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFoldingCtrl/switch_1", node.input(0));
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
