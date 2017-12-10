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

TEST_F(DependencyOptimizerTest, ChangeToNoop) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output y = ops::Const(s.WithOpName("y"), {1.0f, 2.0f}, {1, 2});
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
    const NodeDef& original = item.graph.node(i);
    const NodeDef& optimized = output.node(i);
    EXPECT_EQ(original.name(), optimized.name());
    if (original.name() == "add") {
      EXPECT_EQ("NoOp", optimized.op());
    } else {
      EXPECT_EQ(original.op(), optimized.op());
    }
    EXPECT_EQ(original.input_size(), optimized.input_size());
    for (int j = 0; j < original.input_size(); ++j) {
      if (original.name() == "add") {
        EXPECT_EQ(AsControlDependency(original.input(j)), optimized.input(j));
      } else {
        EXPECT_EQ(original.input(j), optimized.input(j));
      }
    }
  }
}

// TODO(rmlarsen): Add test to make sure we skip Switch and Merge.
TEST_F(DependencyOptimizerTest, ChangeToNoop_NoFetch) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output y = ops::Const(s.WithOpName("y"), {1.0f, 2.0f}, {1, 2});
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

  VerifyGraphsEqual(item.graph, output, __FUNCTION__);
}

TEST_F(DependencyOptimizerTest, RemoveNoOps_EmptyInputOrOutput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s, {1.0f, 2.0f}, {1, 2});
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
      EXPECT_EQ("Const", node.input(0));
    }
  }
}

TEST_F(DependencyOptimizerTest, RemoveNoOps_SingleInputOrOutput) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output y = ops::Const(s.WithOpName("y"), {1.0f, 2.0f}, {1, 2});
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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
