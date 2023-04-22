/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/common_subgraph_elimination.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

namespace {

void VerifyGraphsMatch(const GraphDef& original_graph,
                       const GraphDef& optimized_graph, int line) {
  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << line;
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = optimized_graph.node(i);
    EXPECT_EQ(original.name(), optimized.name()) << line;
    EXPECT_EQ(original.op(), optimized.op()) << line;
    EXPECT_EQ(original.input_size(), optimized.input_size()) << line;
    for (int j = 0; j < original.input_size(); ++j) {
      EXPECT_EQ(original.input(j), optimized.input(j)) << line;
    }
  }
}
}  // namespace

class CommonSubgraphEliminationTest : public ArithmeticOptimizerTest {};

TEST_F(CommonSubgraphEliminationTest, NoOp) {
  // This trivial graph is so basic there's nothing to optimize.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  CommonSubgraphElimination optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  VerifyGraphsMatch(item.graph, output, __LINE__);
}

TEST_F(CommonSubgraphEliminationTest, OpDedupping) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c1 = ops::Const(s.WithOpName("c1"), {3.14, 2.7}, {1, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), {3.14, 2.7}, {1, 2});
  Output div = ops::Div(s.WithOpName("div"), c1, c2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div"};

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  CommonSubgraphElimination optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);
  EXPECT_EQ(output.node_size(), 2);
  const NodeDef* new_c1 = node_map.GetNode("c1");
  ASSERT_NE(new_c1, nullptr);

  const NodeDef* new_div = node_map.GetNode("div");
  ASSERT_NE(new_div, nullptr);
  ASSERT_EQ(new_div->input_size(), 2);
  EXPECT_EQ(new_div->input(0), "c1");
  EXPECT_EQ(new_div->input(1), "c1");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<double>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(CommonSubgraphEliminationTest, OpDeduppingAssertAndCheckNumerics) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output p = ops::Placeholder(s, DT_BOOL, ops::Placeholder::Shape({}));
  Output c = ops::Const(s.WithOpName("c"), {3.14, 2.7}, {1, 2});
  auto check1 = ops::CheckNumerics(s.WithOpName("check1"), c, "foo");
  auto check2 = ops::CheckNumerics(s.WithOpName("check2"), c, "foo");
  auto assert1 = ops::Assert(s.WithOpName("assert1"), p, {c});
  auto assert2 = ops::Assert(s.WithOpName("assert2"), p, {c});
  Output div = ops::Div(s.WithOpName("div").WithControlDependencies(
                            {assert1.operation, assert2.operation}),
                        check1, check2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div"};
  Tensor bool_t(DT_BOOL, TensorShape({}));
  bool_t.scalar<bool>().setConstant(true);
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", bool_t}});
  ASSERT_EQ(tensors_expected.size(), 1);

  CommonSubgraphElimination optimizer;
  GraphDef output;

  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 6);
  const NodeDef* new_div = node_map.GetNode("div");
  ASSERT_NE(new_div, nullptr);
  ASSERT_EQ(new_div->input_size(), 3);
  EXPECT_EQ(new_div->input(0), "check1");
  EXPECT_EQ(new_div->input(1), "check2");
  EXPECT_EQ(new_div->input(2), "^assert1");

  auto tensors = EvaluateNodes(output, item.fetch, {{"Placeholder", bool_t}});
  EXPECT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<double>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(CommonSubgraphEliminationTest, OpDedupCommutative) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c1 = ops::Const(s.WithOpName("c1"), {1.0f, 2.0f}, {1, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), {3.0f, 4.0f}, {1, 2});
  Output mul1 = ops::Mul(s.WithOpName("mul1"), c1, c2);
  Output mul2 = ops::Mul(s.WithOpName("mul2"), c2, c1);
  Output div1 = ops::Div(s.WithOpName("div1"), mul1, mul2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div1"};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  CommonSubgraphElimination optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 4);
  const NodeDef* new_c1 = node_map.GetNode("c1");
  ASSERT_NE(new_c1, nullptr);
  const NodeDef* new_c2 = node_map.GetNode("c2");
  ASSERT_NE(new_c2, nullptr);
  const NodeDef* new_mul1 = node_map.GetNode("mul1");
  ASSERT_NE(new_mul1, nullptr);
  ASSERT_EQ(new_mul1->input_size(), 2);
  EXPECT_EQ(new_mul1->input(0), "c1");
  EXPECT_EQ(new_mul1->input(1), "c2");
  const NodeDef* new_div1 = node_map.GetNode("div1");
  ASSERT_NE(new_div1, nullptr);
  ASSERT_EQ(new_div1->input_size(), 2);
  EXPECT_EQ(new_div1->input(0), "mul1");
  EXPECT_EQ(new_div1->input(1), "mul1");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

}  // namespace grappler
}  // namespace tensorflow
