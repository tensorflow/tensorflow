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

#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer_test_utils.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kHoistFactorOptimizerDiv[] =
    "ArithmeticOptimizer/HoistCommonFactor_Div_";

constexpr char kHoistFactorOptimizerMul[] =
    "ArithmeticOptimizer/HoistCommonFactor_Mul_";

constexpr char kHoistFactorOptimizerAdd[] =
    "ArithmeticOptimizer/HoistCommonFactor_Add_";

constexpr char kSimplifyAggregationConst[] =
    "ArithmeticOptimizer/SimplifyAggregation_Const_";

constexpr char kSimplifyAggregationMul[] =
    "ArithmeticOptimizer/SimplifyAggregation_Mul_";

// Optimized name of outer Mul node by HoistCommonFactorOutOfAggregation.
string HoistMulName(const string& name) {
  return AddPrefixToNodeName(name, kHoistFactorOptimizerMul, "");
}

// Optimized name of outer Div node by HoistCommonFactorOutOfAggregation.
string HoistDivName(const string& name) {
  return AddPrefixToNodeName(name, kHoistFactorOptimizerDiv, "");
}

// Optimized name of inner Add node by HoistCommonFactorOutOfAggregation.
string HoistAddName(const string& name) {
  return AddPrefixToNodeName(name, kHoistFactorOptimizerAdd, "");
}

// Optimized name of Const node by SimplifyAggregation.
string AggregationConstName(const string& name) {
  return AddPrefixToNodeName(name, kSimplifyAggregationConst, "");
}

// Optimized name of Mul node by SimplifyAggregation.
string AggregationMulName(const string& name) {
  return AddPrefixToNodeName(name, kSimplifyAggregationMul, "");
}

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

TEST_F(ArithmeticOptimizerTest, NoOp) {
  // This trivial graph is so basic there's nothing to optimize.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  VerifyGraphsMatch(item.graph, output, __LINE__);
}

TEST_F(ArithmeticOptimizerTest, OpDedupping) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c1 = ops::Const(s.WithOpName("c1"), {3.14, 2.7}, {1, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), {3.14, 2.7}, {1, 2});
  Output div = ops::Div(s.WithOpName("div"), c1, c2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div"};

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  ArithmeticOptimizer optimizer;
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

TEST_F(ArithmeticOptimizerTest, OpDeduppingAssertAndCheckNumerics) {
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

  ArithmeticOptimizer optimizer;
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

TEST_F(ArithmeticOptimizerTest, OpDedupCommutative) {
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

  ArithmeticOptimizer optimizer;
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

TEST_F(ArithmeticOptimizerTest, ReplaceMulWithSquare) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output d = ops::Const(s.WithOpName("d"), {3.0f, 4.0f}, {1, 2});
  Output mul = ops::Mul(s.WithControlDependencies(d).WithOpName("mul"), c, c);
  Output mul_no_nan = ops::MulNoNan(s.WithOpName("mul_no_nan"), d, d);
  Output id = ops::Identity(s.WithOpName("id"), mul);
  Output id2 = ops::Identity(s.WithOpName("id2"), mul_no_nan);

  GrapplerItem item;
  item.fetch = {"id", "id2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 2);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyReplaceMulWithSquare(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(output.node_size(), 6);

  NodeMap node_map(&output);
  const string p = "ArithmeticOptimizer/ReplaceMulWithSquare";
  const NodeDef* square_node = node_map.GetNode(absl::StrCat(p, "_", "mul"));

  ASSERT_NE(square_node, nullptr);
  EXPECT_EQ(square_node->op(), "Square");
  ASSERT_EQ(square_node->input_size(), 2);
  EXPECT_EQ(square_node->input(0), "c");
  EXPECT_EQ(square_node->input(1), "^d");

  const NodeDef* square_node2 =
      node_map.GetNode(absl::StrCat(p, "_", "mul_no_nan"));
  ASSERT_NE(square_node2, nullptr);
  EXPECT_EQ(square_node2->op(), "Square");
  ASSERT_EQ(square_node2->input_size(), 1);
  EXPECT_EQ(square_node2->input(0), "d");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 2);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveInvolutionAdjacentNodes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  auto neg1 = ops::Neg(s.WithOpName("neg1"), c);
  auto neg2 = ops::Neg(s.WithOpName("neg2"), neg1);
  auto recip1 = ops::Reciprocal(s.WithOpName("recip1"), neg2);
  auto recip2 = ops::Reciprocal(s.WithOpName("recip2"), recip1);
  auto id = ops::Identity(s.WithOpName("id"), recip2);

  GrapplerItem item;
  item.fetch = {"id"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveInvolution(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  // Negation and Reciprocal nodes cancelled each other.
  ASSERT_EQ(output.node_size(), 2);
  EXPECT_EQ(output.node(1).name(), "id");
  ASSERT_EQ(output.node(1).input_size(), 1);
  EXPECT_EQ(output.node(1).input(0), "c");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveInvolutionAroundValuePreservingChain) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  auto recip1 = ops::Reciprocal(s.WithOpName("recip1"), c);
  auto id1 = ops::Identity(s.WithOpName("id1"), recip1);
  auto squeeze = ops::Squeeze(s.WithOpName("squeeze"), id1);
  auto recip2 = ops::Reciprocal(s.WithOpName("recip2"), squeeze);
  auto id2 = ops::Identity(s.WithOpName("id2"), recip2);

  std::vector<string> fetch = {"id2"};

  GrapplerItem item;
  item.fetch = fetch;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveInvolution(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  // Check that Reciprocal nodes were removed from the graph.
  EXPECT_EQ(output.node_size(), 3);

  // And const directly flows into squeeze.
  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "squeeze") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "c");
      found++;
    } else if (node.name() == "id2") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "squeeze");
      found++;
    }
  }
  EXPECT_EQ(found, 2);

  auto tensors = EvaluateNodes(output, fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveInvolutionSkipControlDependencies) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  auto recip1 = ops::Reciprocal(s.WithOpName("recip1"), c);
  auto id1 = ops::Identity(s.WithOpName("id1"), recip1);
  auto squeeze = ops::Squeeze(s.WithOpName("squeeze"), id1);
  auto recip2 = ops::Reciprocal(
      s.WithOpName("recip2").WithControlDependencies(squeeze), c);
  auto id2 = ops::Identity(s.WithOpName("id2"), recip2);

  std::vector<string> fetch = {"id2"};

  GrapplerItem item;
  item.fetch = fetch;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveInvolution(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);  // do not prune in this test

  // The optimizer should be a noop.
  VerifyGraphsMatch(item.graph, output, __LINE__);

  auto tensors = EvaluateNodes(output, fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, TrivialSumsSimple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output add = ops::Add(s.WithOpName("add"), x, x);
  Output id = ops::Identity(s.WithOpName("id"), add);

  GrapplerItem item;
  item.fetch = {"id"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 5);

  const string optimized_const_name = AggregationConstName("add");
  const string optimized_mul_name = AggregationMulName("add");

  const NodeDef* new_const = node_map.GetNode(optimized_const_name);
  ASSERT_NE(new_const, nullptr);
  ASSERT_EQ(new_const->input_size(), 1);
  EXPECT_EQ(new_const->input(0), "^x");
  EXPECT_EQ(new_const->attr().at("value").tensor().tensor_content(),
            string("\0\0\0@", 4));

  const NodeDef* new_mul = node_map.GetNode(optimized_mul_name);
  ASSERT_NE(new_mul, nullptr);
  ASSERT_EQ(new_mul->input_size(), 2);
  EXPECT_EQ(new_mul->input(0), optimized_const_name);
  EXPECT_EQ(new_mul->input(1), "x");

  const NodeDef* new_id = node_map.GetNode("id");
  ASSERT_NE(new_id, nullptr);
  ASSERT_EQ(new_id->input_size(), 1);
  EXPECT_EQ(new_id->input(0), optimized_mul_name);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, TrivialSumsSimpleWithControlDep) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output y = ops::Const(s.WithOpName("y"), {1.0f, 2.0f}, {1, 2});
  Output x = ops::Const(s.WithOpName("x"), {3.0f, 4.0f}, {1, 2});
  Output add = ops::Add(s.WithOpName("add").WithControlDependencies(y), x, x);
  Output id = ops::Identity(s.WithOpName("id"), add);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  std::vector<string> fetch = {"id"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 6);

  const string optimized_const_name = AggregationConstName("add");
  const string optimized_mul_name = AggregationMulName("add");

  const NodeDef* new_const = node_map.GetNode(optimized_const_name);
  ASSERT_NE(new_const, nullptr);
  ASSERT_EQ(new_const->input_size(), 1);
  EXPECT_EQ(new_const->input(0), "^x");
  EXPECT_EQ(new_const->attr().at("value").tensor().tensor_content(),
            string("\0\0\0@", 4));

  const NodeDef* new_mul = node_map.GetNode(optimized_mul_name);
  ASSERT_NE(new_mul, nullptr);
  ASSERT_EQ(new_mul->input_size(), 3);
  EXPECT_EQ(new_mul->input(0), optimized_const_name);
  EXPECT_EQ(new_mul->input(1), "x");
  EXPECT_EQ(new_mul->input(2), "^y");

  const NodeDef* new_id = node_map.GetNode("id");
  ASSERT_NE(new_id, nullptr);
  ASSERT_EQ(new_id->input_size(), 1);
  EXPECT_EQ(new_id->input(0), optimized_mul_name);

  auto tensors = EvaluateNodes(output, fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, TrivialSumsRepeatedAdd) {
  // Test case from b/69059093.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output p = ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({10, 10}));
  Output add = ops::Add(s.WithOpName("Add"), p, p);
  Output add1 = ops::Add(s.WithOpName("Add_1"), p, p);
  Output add4 = ops::Add(s.WithOpName("Add_4"), add, add1);
  Output add5 = ops::Add(s.WithOpName("Add_5"), add, add1);
  Output add6 = ops::Add(s.WithOpName("Add_6"), add4, add5);
  Output id = ops::Identity(s.WithOpName("id"), add6);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  const std::vector<string> devices{
      "/device:CPU:0", "/device:GPU:0", "/device:CPU:0", "/device:GPU:1",
      "/device:CPU:0", "/device:CPU:0", "/device:CPU:0",
  };
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device(devices[i]);
  }

  ArithmeticOptimizer optimizer;
  DisableAddToAddNCombining(&optimizer);

  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //
  // Mul(p,
  //     Add_6(Add_4(Const(2), Const(2)),
  //           Add_5(Const(2), Const(2))))
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 17);

  const NodeDef* id_node = node_map.GetNode("id");
  ASSERT_NE(id_node, nullptr);
  ASSERT_EQ(id_node->input_size(), 1);
  EXPECT_EQ(id_node->input(0), HoistMulName("Add_6"));

  const NodeDef* mul_node = node_map.GetNode(HoistMulName("Add_6"));
  ASSERT_NE(mul_node, nullptr);
  ASSERT_EQ(mul_node->input_size(), 2);
  EXPECT_EQ(mul_node->input(0), HoistAddName("Add_6"));
  EXPECT_EQ(mul_node->input(1), "Placeholder");

  const NodeDef* add_6_node = node_map.GetNode(HoistAddName("Add_6"));
  ASSERT_NE(add_6_node, nullptr);
  ASSERT_EQ(add_6_node->input_size(), 2);
  EXPECT_EQ(add_6_node->input(0), HoistAddName("Add_4"));
  EXPECT_EQ(add_6_node->input(1), HoistAddName("Add_5"));

  const NodeDef* add_4_node = node_map.GetNode(HoistAddName("Add_4"));
  ASSERT_NE(add_4_node, nullptr);
  EXPECT_EQ(add_4_node->op(), "Add");
  ASSERT_EQ(2, add_4_node->input_size());
  EXPECT_EQ(add_4_node->input(0), AggregationConstName("Add"));
  EXPECT_EQ(add_4_node->input(1), AggregationConstName("Add_1"));

  const NodeDef* add_5_node = node_map.GetNode(HoistAddName("Add_5"));
  ASSERT_NE(add_5_node, nullptr);
  EXPECT_EQ(add_5_node->op(), "Add");
  ASSERT_EQ(add_5_node->input_size(), 2);
  EXPECT_EQ(add_5_node->input(0), AggregationConstName("Add"));
  EXPECT_EQ(add_5_node->input(1), AggregationConstName("Add_1"));

  const NodeDef* add_const_node = node_map.GetNode(AggregationConstName("Add"));
  ASSERT_NE(add_const_node, nullptr);
  EXPECT_EQ(add_const_node->op(), "Const");
  ASSERT_EQ(add_const_node->input_size(), 1);
  EXPECT_EQ(add_const_node->input(0), "^Placeholder");

  const NodeDef* add_1_const_node =
      node_map.GetNode(AggregationConstName("Add_1"));
  ASSERT_NE(add_1_const_node, nullptr);
  EXPECT_EQ(add_1_const_node->op(), "Const");
  ASSERT_EQ(add_1_const_node->input_size(), 1);
  EXPECT_EQ(add_1_const_node->input(0), "^Placeholder");
}

TEST_F(ArithmeticOptimizerTest, HoistFactorMul) {
  for (bool matching_shapes : {true, false}) {
    for (bool use_addn : {true, false}) {
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();
      Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
      Output y1 = ops::Const(s.WithOpName("y1"), {3.0f, 4.0f}, {1, 2});
      Output y2 = matching_shapes
                      ? ops::Const(s.WithOpName("y2"), {5.0f, 6.0f}, {1, 2})
                      : ops::Const(s.WithOpName("y2"), {5.0f}, {1, 1});
      Output mul1 = ops::Mul(s.WithOpName("mul1"), x, y1);
      Output mul2 = ops::Mul(s.WithOpName("mul2"), y2, x);
      Output id =
          use_addn ? ops::Identity(s.WithOpName("id"),
                                   ops::AddN(s.WithOpName("add"), {mul1, mul2}))
                   : ops::Identity(s.WithOpName("id"),
                                   ops::Add(s.WithOpName("add"), mul1, mul2));

      GrapplerItem item;
      item.fetch = {"id"};
      TF_CHECK_OK(s.ToGraphDef(&item.graph));
      auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
      ASSERT_EQ(tensors_expected.size(), 1);
      ArithmeticOptimizer optimizer;
      EnableOnlyHoistCommonFactor(&optimizer);

      GraphDef output;
      OptimizeTwice(&optimizer, &item, &output);

      // We expect the following rewrite(s) to occur:
      //
      //        Add                 Mul
      //      /    \               /   \
      //    Mul    Mul       ->   x    Add
      //    / \    / \                 / \
      //   x  y1  y2  x              y1   y2
      //
      // If "root" op is AddN and shapes does not match, this rewrite is not
      // possible and graph should stay intact.
      NodeMap node_map(&output);

      if (use_addn && !matching_shapes) {
        VerifyGraphsMatch(item.graph, output, __LINE__);
      } else {
        EXPECT_EQ(output.node_size(), 9);

        const NodeDef* new_add_node = node_map.GetNode(HoistAddName("add"));
        ASSERT_NE(new_add_node, nullptr) << "Hoisted Add node not found";
        ASSERT_EQ(new_add_node->input_size(), 2);
        EXPECT_EQ(new_add_node->input(0), "y1");
        EXPECT_EQ(new_add_node->input(1), "y2");

        const NodeDef* new_mul_node = node_map.GetNode(HoistMulName("add"));
        ASSERT_NE(new_mul_node, nullptr) << "Hoisted Mul node not found";
        ASSERT_EQ(new_mul_node->input_size(), 2);
        EXPECT_EQ(new_mul_node->input(0), "x");
        EXPECT_EQ(new_mul_node->input(1), new_add_node->name());

        const NodeDef* id_node = node_map.GetNode("id");
        ASSERT_NE(id_node, nullptr) << "Id node not found";
        EXPECT_EQ(id_node->name(), "id");
        ASSERT_EQ(id_node->input_size(), 1);
        EXPECT_EQ(id_node->input(0), HoistMulName("add"));
      }
      auto tensors = EvaluateNodes(output, item.fetch);
      ASSERT_EQ(tensors.size(), 1);
      test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
    }
  }
}

TEST_F(ArithmeticOptimizerTest, HoistFactorDiv) {
  for (bool matching_shapes : {true, false}) {
    for (bool use_addn : {true, false}) {
      for (bool use_ints : {true, false}) {
        tensorflow::Scope s = tensorflow::Scope::NewRootScope();
        Output x = use_ints
                       ? ops::Const(s.WithOpName("x"), {1, 2}, {1, 2})
                       : ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
        Output y1 = use_ints
                        ? ops::Const(s.WithOpName("y1"), {3, 4}, {1, 2})
                        : ops::Const(s.WithOpName("y1"), {3.0f, 4.0f}, {1, 2});
        Output y2;
        if (matching_shapes) {
          y2 = use_ints ? ops::Const(s.WithOpName("y2"), {5, 6}, {1, 2})
                        : ops::Const(s.WithOpName("y2"), {5.0f, 6.0f}, {1, 2});
        } else {
          y2 = use_ints ? ops::Const(s.WithOpName("y2"), {5}, {1, 1})
                        : ops::Const(s.WithOpName("y2"), {5.0f}, {1, 1});
        }
        Output div1 = ops::Div(s.WithOpName("div1"), y1, x);
        Output div2 = ops::Div(s.WithOpName("div2"), y2, x);
        Output id =
            use_addn
                ? ops::Identity(s.WithOpName("id"),
                                ops::AddN(s.WithOpName("add"), {div1, div2}))
                : ops::Identity(s.WithOpName("id"),
                                ops::Add(s.WithOpName("add"), div1, div2));

        GrapplerItem item;
        item.fetch = {"id"};
        TF_CHECK_OK(s.ToGraphDef(&item.graph));

        auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
        ASSERT_EQ(tensors_expected.size(), 1);

        ArithmeticOptimizer optimizer;
        EnableOnlyHoistCommonFactor(&optimizer);

        GraphDef output;
        OptimizeTwice(&optimizer, &item, &output);

        // We expect the following rewrite(s) to occur:
        //
        //        Add                 Div
        //      /    \               /   \
        //    Div    Div       ->  Add    x
        //    / \    / \           / \
        //   y1  x  y2  x         y1  y2
        //
        // If "root" op is AddN and shapes does not match, this rewrite is not
        // possible and graph should stay intact.
        NodeMap node_map(&output);

        if ((use_addn && !matching_shapes) || use_ints) {
          VerifyGraphsMatch(item.graph, output, __LINE__);
        } else {
          EXPECT_EQ(output.node_size(), 9);

          const NodeDef* new_add_node = node_map.GetNode(HoistAddName("add"));
          ASSERT_TRUE(new_add_node != nullptr) << "Hoisted Add node not found";
          ASSERT_EQ(new_add_node->input_size(), 2);
          EXPECT_EQ(new_add_node->input(0), "y1");
          EXPECT_EQ(new_add_node->input(1), "y2");

          const NodeDef* new_div_node = node_map.GetNode(HoistDivName("add"));
          ASSERT_TRUE(new_div_node != nullptr) << "Hoisted Div node not found";
          ASSERT_EQ(new_div_node->input_size(), 2);
          EXPECT_EQ(new_div_node->input(0), new_add_node->name());
          EXPECT_EQ(new_div_node->input(1), "x");

          const NodeDef* id_node = node_map.GetNode("id");
          ASSERT_TRUE(id_node != nullptr) << "Id node not found";
          EXPECT_EQ("id", id_node->name());
          ASSERT_EQ(id_node->input_size(), 1);
          EXPECT_EQ(id_node->input(0), HoistDivName("add"));
        }
        auto tensors = EvaluateNodes(output, item.fetch);
        ASSERT_EQ(tensors.size(), 1);
        if (use_ints) {
          test::ExpectTensorEqual<int32>(tensors[0], tensors_expected[0]);
        } else {
          test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
        }
      }
    }
  }
}

TEST_F(ArithmeticOptimizerTest, FuseConjAndTranspose) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output re = ops::Const(s.WithOpName("re"), {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Output im = ops::Const(s.WithOpName("im"), {5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  Output z = ops::Complex(s.WithOpName("z"), re, im);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
  Output conj = ops::Conj(s.WithOpName("conj"), z);
  Output transp = ops::Transpose(s.WithOpName("trans"), conj, perm);

  GrapplerItem item;
  item.fetch = {"trans"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 7);

  const string p = "ArithmeticOptimizer/FoldConjugateIntoTranspose";
  const string optimized_name = absl::StrCat(p, "_", "trans");

  const NodeDef* trans_fused_node = node_map.GetNode(optimized_name);
  ASSERT_NE(trans_fused_node, nullptr);
  EXPECT_EQ(trans_fused_node->op(), "ConjugateTranspose");
  ASSERT_EQ(trans_fused_node->input_size(), 2);
  EXPECT_EQ(trans_fused_node->input(0), "z");
  EXPECT_EQ(trans_fused_node->input(1), "perm");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<complex64>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, FuseConjAndConjugateTranspose) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output re = ops::Const(s.WithOpName("re"), {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Output im = ops::Const(s.WithOpName("im"), {5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  Output z = ops::Complex(s.WithOpName("z"), re, im);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
  Output conj = ops::Conj(s.WithOpName("conj"), z);
  Output transp =
      ops::ConjugateTranspose(s.WithOpName("conjugate_trans"), conj, perm);

  GrapplerItem item;
  item.fetch = {"conjugate_trans"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 7);

  const string p = "ArithmeticOptimizer/FoldConjugateIntoTranspose";
  const string optimized_name = absl::StrCat(p, "_", "conjugate_trans");

  const NodeDef* conjugate_trans_fused_node = node_map.GetNode(optimized_name);
  ASSERT_NE(conjugate_trans_fused_node, nullptr);
  EXPECT_EQ(conjugate_trans_fused_node->op(), "Transpose");
  ASSERT_EQ(conjugate_trans_fused_node->input_size(), 2);
  EXPECT_EQ(conjugate_trans_fused_node->input(0), "z");
  EXPECT_EQ(conjugate_trans_fused_node->input(1), "perm");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<complex64>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, FuseTransposeAndConj) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output re = ops::Const(s.WithOpName("re"), {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Output im = ops::Const(s.WithOpName("im"), {5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  Output z = ops::Complex(s.WithOpName("z"), re, im);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
  Output trans = ops::Transpose(s.WithOpName("trans"), z, perm);
  Output conj = ops::Conj(s.WithOpName("conj"), trans);

  GrapplerItem item;
  item.fetch = {"conj"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(output.node_size(), 7);

  const string p = "ArithmeticOptimizer/FoldConjugateIntoTranspose";
  const string optimized_name = absl::StrCat(p, "_", "conj");

  const NodeDef* conj_fused_node = node_map.GetNode(optimized_name);
  ASSERT_NE(conj_fused_node, nullptr);
  EXPECT_EQ(conj_fused_node->op(), "ConjugateTranspose");
  ASSERT_EQ(conj_fused_node->input_size(), 2);
  EXPECT_EQ(conj_fused_node->input(0), "z");
  EXPECT_EQ(conj_fused_node->input(1), "perm");

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<complex64>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, FoldTransposeIntoMatMul) {
  for (const string matmul_type :
       {"MatMul", "SparseMatMul", "BatchMatMul", "BatchMatMulV2"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Output a = ops::Const(s.WithOpName("a"), {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Output b = ops::Const(s.WithOpName("b"), {5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
    Output trans_a = ops::Transpose(s.WithOpName("trans_a"), a, perm);
    Output trans_b = ops::Transpose(s.WithOpName("trans_b"), b, perm);

    Output matmul;
    auto matmul_op = s.WithOpName("matmul");
    if (matmul_type == "MatMul") {
      matmul = ops::MatMul(matmul_op, trans_a, trans_b);
    } else if (matmul_type == "SparseMatMul") {
      matmul = ops::SparseMatMul(matmul_op, trans_a, trans_b);
    } else if (matmul_type == "BatchMatMul") {
      matmul = ops::BatchMatMul(matmul_op, trans_a, trans_b);
    } else if (matmul_type == "BatchMatMulV2") {
      matmul = ops::BatchMatMulV2(matmul_op, trans_a, trans_b);
    }

    auto identity = ops::Identity(s.WithOpName("identity"), matmul);

    GrapplerItem item;
    item.fetch = {"matmul"};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
    ASSERT_EQ(tensors_expected.size(), 1);

    ArithmeticOptimizer optimizer;
    EnableOnlyFoldTransposeIntoMatMul(&optimizer);
    GraphDef output;
    OptimizeTwice(&optimizer, &item, &output);
    NodeMap node_map(&output);

    EXPECT_EQ(output.node_size(), 8);

    const string p = "ArithmeticOptimizer/FoldTransposeIntoMatMul";
    const string optimized_name = absl::StrCat(p, "_", "matmul");

    const NodeDef* matmul_fused_node = node_map.GetNode(optimized_name);
    ASSERT_NE(matmul_fused_node, nullptr);
    ASSERT_EQ(matmul_fused_node->input_size(), 2);
    EXPECT_EQ(matmul_fused_node->input(0), "a");
    EXPECT_EQ(matmul_fused_node->input(1), "b");

    if (matmul_type == "BatchMatMul" || matmul_type == "BatchMatMulV2") {
      EXPECT_TRUE(matmul_fused_node->attr().at("adj_x").b());
      EXPECT_TRUE(matmul_fused_node->attr().at("adj_y").b());
    } else {
      EXPECT_TRUE(matmul_fused_node->attr().at("transpose_a").b());
      EXPECT_TRUE(matmul_fused_node->attr().at("transpose_b").b());
    }

    const NodeDef* identity_node = node_map.GetNode("identity");
    ASSERT_NE(identity_node, nullptr);
    ASSERT_EQ(identity_node->input_size(), 1);
    EXPECT_EQ(identity_node->input(0), optimized_name);

    auto tensors = EvaluateNodes(output, item.fetch);
    ASSERT_EQ(tensors.size(), 1);
    test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  }
}

TEST_F(ArithmeticOptimizerTest, FoldConjugateTransposeIntoBatchMatMul) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output re_a =
      ops::Const(s.WithOpName("re_a"), {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  Output im_a =
      ops::Const(s.WithOpName("im_a"), {-1.0f, -2.0f, -3.0f, -4.0f}, {2, 2});
  Output re_b =
      ops::Const(s.WithOpName("re_b"), {5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  Output im_b =
      ops::Const(s.WithOpName("im_b"), {-5.0f, -6.0f, -7.0f, -8.0f}, {2, 2});
  Output a = ops::Complex(s.WithOpName("a"), re_a, im_a);
  Output b = ops::Complex(s.WithOpName("b"), re_b, im_b);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
  Output trans_a = ops::ConjugateTranspose(s.WithOpName("trans_a"), a, perm);
  Output trans_b = ops::ConjugateTranspose(s.WithOpName("trans_b"), b, perm);
  Output matmul = ops::BatchMatMul(s.WithOpName("matmul"), trans_a, trans_b);

  GrapplerItem item;
  item.fetch = {"matmul"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);

  NodeMap node_map(&output);
  EXPECT_EQ(output.node_size(), 11);

  const string p = "ArithmeticOptimizer/FoldTransposeIntoMatMul";
  const string optimized_name = absl::StrCat(p, "_", "matmul");

  const NodeDef* optimized_matmul = node_map.GetNode(optimized_name);
  ASSERT_NE(optimized_matmul, nullptr);
  ASSERT_EQ(optimized_matmul->input_size(), 2);
  EXPECT_EQ(optimized_matmul->input(0), "a");
  EXPECT_EQ(optimized_matmul->input(1), "b");
  EXPECT_TRUE(optimized_matmul->attr().at("adj_x").b());
  EXPECT_TRUE(optimized_matmul->attr().at("adj_y").b());

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<complex64>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshapeIdentityReshape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({-1, 3, 28, 28}));
  Output inputs_shape = ops::Shape(s, inputs);
  // The target shape of the reshape is the concatenation of `batch_size` and
  // [3,28,28].
  Output batch_size = ops::Slice(s, inputs_shape, ops::Const(s, {0}, {1}),
                                 ops::Const(s, {1}, {1}));
  Output target_shape = ops::Concat(
      s.WithOpName("target_shape"),
      {batch_size, ops::Const(s, {3, 28, 28}, {3})}, ops::Const(s, {0}, {}));
  Output reshape = ops::Reshape(s, inputs, target_shape);
  Output outputs = ops::Identity(s.WithOpName("outputs"), reshape);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 3, 28, 28}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", x_t}});
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(CountOpNodes(output, "Reshape"), 0);
  auto tensors = EvaluateNodes(output, item.fetch, {{"Placeholder", x_t}});
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest,
       RemoveRedundantReshapeIdentityReshapeBetweenSymbolicShapes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({-1, 3, -1, -1}));
  Output inputs_shape = ops::Shape(s, inputs);
  // The target shape of the reshape is the concatenation of `batch_size`, 3,
  // `height, and `width`.
  Output batch_size = ops::Slice(s, inputs_shape, ops::Const(s, {0}, {1}),
                                 ops::Const(s, {1}, {1}));
  Output height = ops::Slice(s, inputs_shape, ops::Const(s, {2}, {1}),
                             ops::Const(s, {1}, {1}));
  Output width = ops::Slice(s, inputs_shape, ops::Const(s, {3}, {1}),
                            ops::Const(s, {1}, {1}));
  Output target_shape =
      ops::Concat(s.WithOpName("target_shape"),
                  {batch_size, ops::Const(s, {3}, {1}), height, width},
                  ops::Const(s, {0}, {}));
  Output reshape = ops::Reshape(s, inputs, target_shape);
  Output outputs = ops::Identity(s.WithOpName("outputs"), reshape);

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 3, 28, 28}));
  GrapplerItem item;
  item.fetch = {"outputs"};
  item.feed = {{"Placeholder", x_t}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  // Assume valid feed shape in aggressive mode.
  ArithmeticOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(CountOpNodes(output, "Reshape"), 0);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshapeNotAssumeValidFeeds) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({4, 3, 28, 28}));
  Output target_shape = ops::Const(s, {4, 3, 28, 28}, {4});
  Output reshape = ops::Reshape(s, inputs, target_shape);
  Output outputs = ops::Identity(s.WithOpName("outputs"), reshape);

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4, 3, 28, 28}));
  GrapplerItem item;
  item.fetch = {"outputs"};
  item.feed = {{"Placeholder", x_t}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  // The reshape is preserved because the shape of the placeholder can be
  // different from the shape of the actual feed.
  EXPECT_EQ(CountOpNodes(output, "Reshape"), 1);

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest,
       RemoveRedundantReshapeAssumeValidFeedsInAggressiveMode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({4, 3, 28, 28}));
  Output target_shape = ops::Const(s, {4, 3, 28, 28}, {4});
  Output reshape = ops::Reshape(s, inputs, target_shape);
  Output outputs = ops::Identity(s.WithOpName("outputs"), reshape);

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4, 3, 28, 28}));
  GrapplerItem item;
  item.fetch = {"outputs"};
  item.feed = {{"Placeholder", x_t}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(CountOpNodes(output, "Reshape"), 0);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshapeNotIdentityReshape) {
  // Reshape from [-1,3,28,28] to [8,-1,28,28] is not identity, because it can
  // be from [4,3,28,28] to [8,6,28,28].
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({-1, 3, 28, 28}));
  Output reshape = ops::Reshape(s, inputs, ops::Const(s, {8, -1, 28, 28}, {4}));
  Output outputs = ops::Identity(s.WithOpName("outputs"), reshape);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({8, 3, 28, 28}));
  item.feed = {{"Placeholder", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(CountOpNodes(output, "Reshape"), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest,
       RemoveRedundantReshapeNotIdentityReshapeTooManyUnknownDimSizes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({4, 3}));
  Output reshape = ops::Reshape(s, inputs, ops::Const(s, {-1, -1}, {2}));
  Output outputs = ops::Identity(s.WithOpName("outputs"), reshape);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(CountOpNodes(output, "Reshape"), 1);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshapeCombineReshapes) {
  // Converts an NCHW_VECT_C tensor to NHWC and then flattens it to 2D. The two
  // reshapes should be combined.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output nchw_vect_c =
      ops::Placeholder(s.WithOpName("nchw_vect_c"), DT_INT8,
                       ops::Placeholder::Shape({8, 3, 28, 28, 4}));
  Output transpose =
      ops::Transpose(s.WithOpName("transpose"), nchw_vect_c,
                     ops::Const(s.WithOpName("perm"), {0, 2, 3, 1, 4}, {5}));
  Output nhwc = ops::Reshape(
      s.WithOpName("nhwc"), transpose,
      ops::Const(s.WithOpName("nhwc_shape"), {8, 28, 28, 12}, {4}));
  Output flatten = ops::Reshape(
      s.WithOpName("flatten"), nhwc,
      ops::Const(s.WithOpName("flatten_shape"), {8, 28 * 28 * 12}, {2}));
  Output outputs = ops::Identity(s.WithOpName("outputs"), flatten);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto x_t = GenerateRandomTensor<DT_INT8>(TensorShape({8, 3, 28, 28, 4}));
  item.feed = {{"nchw_vect_c", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(CountOpNodes(output, "Reshape"), 1);
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<int8>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, ReorderTransposeCastProducerIsCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/CPU:0");
  Output nhwc_uint8 =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output nhwc_fp32 = ops::Cast(s, nhwc_uint8, DT_FLOAT);
  Output nchw_fp32 =
      ops::Transpose(s, nhwc_fp32, ops::Const(s, {0, 3, 1, 2}, {4}));
  Output outputs = ops::Identity(s.WithOpName("outputs"), nchw_fp32);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto input_t = GenerateRandomTensor<DT_UINT8>(TensorShape({8, 28, 28, 3}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  OptimizeAndPrune(&optimizer, &item, &output);

  const NodeDef* transpose_node = nullptr;
  for (const NodeDef& node : output.node()) {
    if (node.op() == "Transpose") {
      EXPECT_EQ(transpose_node, nullptr);
      EXPECT_EQ(node.attr().at("T").type(), DT_UINT8);
      transpose_node = &node;
    }
  }
  ASSERT_NE(transpose_node, nullptr);

  for (const NodeDef& node : output.node()) {
    if (node.op() == "Cast") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(transpose_node->name(), NodeName(node.input(0)));
    }
  }

  auto tensors =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<float>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, ReorderS2DCastProducerIsCast) {
  // TODO(jingyue): Evaluate S2D+Cast on GPU as well. We can't simply put nodes
  // under a /GPU:0 scope, because this test would fail if the testing machine
  // doesn't have a GPU. Maybe EvaluateNodes should allow soft placement?
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/CPU:0");
  Output outputs =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({8, 28, 28, 3}));
  outputs = ops::Cast(s, outputs, DT_FLOAT);
  outputs = ops::SpaceToDepth(s, outputs, 2);
  outputs = ops::Identity(s.WithOpName("outputs"), outputs);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto input_t = GenerateRandomTensor<DT_UINT8>(TensorShape({8, 28, 28, 3}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  OptimizeAndPrune(&optimizer, &item, &output);

  const NodeDef* s2d_node = nullptr;
  for (const NodeDef& node : output.node()) {
    if (node.op() == "SpaceToDepth") {
      EXPECT_EQ(s2d_node, nullptr);
      EXPECT_EQ(node.attr().at("T").type(), DT_UINT8);
      s2d_node = &node;
    }
  }
  ASSERT_NE(s2d_node, nullptr);

  for (const NodeDef& node : output.node()) {
    if (node.op() == "Cast") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(s2d_node->name(), NodeName(node.input(0)));
    }
  }

  auto tensors =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<float>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, ReorderTransposeCastProducerIsTranspose) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/CPU:0");
  Output nhwc_fp32 =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output nchw_fp32 =
      ops::Transpose(s, nhwc_fp32, ops::Const(s, {0, 3, 1, 2}, {4}));
  Output nchw_uint8 = ops::Cast(s, nchw_fp32, DT_UINT8);
  Output outputs = ops::Identity(s.WithOpName("outputs"), nchw_uint8);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto input_t =
      GenerateConstantTensor<DT_FLOAT>(TensorShape({8, 28, 28, 3}), 42.0f);
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  OptimizeAndPrune(&optimizer, &item, &output);

  const NodeDef* cast_node = nullptr;
  for (const NodeDef& node : output.node()) {
    if (node.op() == "Cast") {
      EXPECT_EQ(cast_node, nullptr);
      cast_node = &node;
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(NodeName(node.input(0)), "Placeholder");
    }
  }
  ASSERT_NE(cast_node, nullptr);

  for (const NodeDef& node : output.node()) {
    if (node.op() == "Transpose") {
      EXPECT_EQ(node.attr().at("T").type(), DT_UINT8);
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(cast_node->name(), NodeName(node.input(0)));
    }
  }

  auto tensors =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<uint8>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, ReorderTransposeReverseCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/CPU:0");
  Output nhwc_uint8 =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output nhwc_fp32 = ops::Cast(s, nhwc_uint8, DT_FLOAT);
  Output nhwc_fp32_reversed =
      ops::Reverse(s, nhwc_fp32, ops::Const(s, {0}, {1}));
  Output nchw_fp32_reversed =
      ops::Transpose(s, nhwc_fp32_reversed, ops::Const(s, {0, 3, 1, 2}, {4}));

  Output outputs = ops::Identity(s.WithOpName("outputs"), nchw_fp32_reversed);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto input_t = GenerateRandomTensor<DT_UINT8>(TensorShape({8, 28, 28, 3}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  OptimizeAndPrune(&optimizer, &item, &output);

  const NodeDef* reverse_node = nullptr;
  const NodeDef* transpose_node = nullptr;
  const NodeDef* cast_node = nullptr;
  for (const NodeDef& node : output.node()) {
    if (node.op() == "Transpose") {
      EXPECT_EQ(transpose_node, nullptr);
      EXPECT_EQ(node.attr().at("T").type(), DT_UINT8);
      transpose_node = &node;
    } else if (node.op() == "ReverseV2") {
      EXPECT_EQ(reverse_node, nullptr);
      EXPECT_EQ(node.attr().at("T").type(), DT_UINT8);
      reverse_node = &node;
    } else if (node.op() == "Cast") {
      cast_node = &node;
    }
  }
  ASSERT_NE(cast_node, nullptr);
  ASSERT_NE(reverse_node, nullptr);
  ASSERT_NE(transpose_node, nullptr);
  ASSERT_EQ(reverse_node->input_size(), 2);
  EXPECT_EQ(NodeName(reverse_node->input(0)), "Placeholder");
  ASSERT_EQ(transpose_node->input_size(), 2);
  EXPECT_EQ(NodeName(transpose_node->input(0)), reverse_node->name());
  ASSERT_EQ(cast_node->input_size(), 1);
  EXPECT_EQ(NodeName(cast_node->input(0)), transpose_node->name());

  auto tensors =
      EvaluateNodes(item.graph, item.fetch, {{"Placeholder", input_t}});
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<float>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, ReorderTransposeCastCheckNumericsToIdentity) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/CPU:0");
  Output nhwc_uint8 =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output nhwc_fp32 = ops::Cast(s, nhwc_uint8, DT_FLOAT);
  Output nchw_fp32 = ops::CheckNumerics(s, nhwc_fp32, "foo");
  Output outputs = ops::Identity(s.WithOpName("outputs"), nchw_fp32);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));
  CompareGraphs(item.graph, output);
}

TEST_F(ArithmeticOptimizerTest, NoReorderTransposeCastProducerIsCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/CPU:0");
  Output nhwc_fp32 =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output nhwc_uint8 = ops::Cast(s, nhwc_fp32, DT_UINT8);
  Output nchw_uint8 =
      ops::Transpose(s, nhwc_uint8, ops::Const(s, {0, 3, 1, 2}, {4}));
  Output outputs = ops::Identity(s.WithOpName("outputs"), nchw_uint8);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));
  CompareGraphs(item.graph, output);
}

TEST_F(ArithmeticOptimizerTest, NoReorderTransposeCastProducerIsTranspose) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/CPU:0");
  Output nhwc_uint8 =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output nchw_uint8 =
      ops::Transpose(s, nhwc_uint8, ops::Const(s, {0, 3, 1, 2}, {4}));
  Output nchw_fp32 = ops::Cast(s, nchw_uint8, DT_FLOAT);
  Output outputs = ops::Identity(s.WithOpName("outputs"), nchw_fp32);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));
  CompareGraphs(item.graph, output);
}

TEST_F(ArithmeticOptimizerTest, RemoveIdentityTransposes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs_shape =
      ops::Const(s.WithOpName("inputs_shape"), {8, 3, 28, 28}, {4});
  Output inputs =
      ops::RandomUniform(s.WithOpName("inputs"), inputs_shape, DT_FLOAT);
  Output perm1 = ops::Const(s.WithOpName("perm1"), {0, 2, 3, 1}, {4});
  Output perm2 = ops::Const(s.WithOpName("perm2"), {0, 3, 1, 2}, {4});
  Output perm3 = ops::Const(s.WithOpName("perm3"), {0, 1, 2, 3}, {4});
  Output transpose1 = ops::Transpose(s.WithOpName("transpose1"), inputs, perm1);
  Output transpose2 =
      ops::Transpose(s.WithOpName("transpose2"), transpose1, perm2);
  Output transpose3 = ops::Transpose(s.WithOpName("transpose3"), inputs, perm3);
  Output id1 = ops::Identity(s.WithOpName("id1"), transpose2);
  Output id2 = ops::Identity(s.WithOpName("id2"), transpose3);

  GrapplerItem item;
  item.fetch = {"id1", "id2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  std::set<string> nodes_after_optimization;
  for (const NodeDef& node : output.node()) {
    nodes_after_optimization.insert(node.name());
  }
  EXPECT_EQ(nodes_after_optimization,
            std::set<string>({"id1", "id2", "inputs_shape", "inputs"}));
}

TEST_F(ArithmeticOptimizerTest, RemoveIdentityTransposesMultipleOutputs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs_shape =
      ops::Const(s.WithOpName("inputs_shape"), {8, 9, 28, 28}, {4});
  Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_FLOAT,
                                   ops::Placeholder::Shape({8, 12, 28, 28}));
  OutputList split = ops::Split(s, ops::Const(s, 1), inputs, 3).output;
  Output perm1 = ops::Const(s, {0, 2, 3, 1}, {4});
  Output perm2 = ops::Const(s, {0, 3, 1, 2}, {4});
  Output branch0 = split[0];
  Output branch1 = ops::Transpose(s, ops::Transpose(s, split[1], perm1), perm2);
  Output branch2 = split[2];
  Output concat = ops::Concat(s, {branch0, branch1, branch2}, ops::Const(s, 1));
  Output outputs = ops::Identity(s.WithOpName("outputs"), concat);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({8, 12, 28, 28}));
  item.feed = {{"inputs", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  for (const NodeDef& node : output.node()) {
    if (node.op() == "Concat") {
      ASSERT_EQ(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "Split");
      EXPECT_EQ(node.input(1), "Split:1");
      EXPECT_EQ(node.input(2), "Split:2");
    }
  }

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveTransposesWithControlDependency) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({2, 3}));
  Output transpose1 = ops::Transpose(s, inputs, ops::Const(s, {1, 0}));
  Output transpose2 = ops::Transpose(s, transpose1, ops::Const(s, {1, 0}));
  Output outputs =
      ops::Identity(s.WithOpName("outputs").WithControlDependencies(transpose2),
                    ops::Const(s.WithOpName("outputs_const"), 1.0f));

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 3}));
  item.feed = {{"Placeholder", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  NodeMap node_map(&output);
  const NodeDef* outputs_node = node_map.GetNode("outputs");
  ASSERT_EQ(outputs_node->input_size(), 2);
  EXPECT_EQ(outputs_node->input(0), "outputs_const");
  EXPECT_EQ(outputs_node->input(1), "^Placeholder");

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, NotRemoveTransposes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs_shape =
      ops::Const(s.WithOpName("inputs_shape"), {8, 3, 28, 28}, {4});
  Output inputs =
      ops::RandomUniform(s.WithOpName("inputs"), inputs_shape, DT_FLOAT);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 2, 3, 0}, {4});
  Output transpose1 = ops::Transpose(s.WithOpName("transpose1"), inputs, perm);
  Output transpose2 =
      ops::Transpose(s.WithOpName("transpose2"), transpose1, perm);
  Output outputs = ops::Identity(s.WithOpName("outputs"), transpose2);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(output.node_size(), 6);
}

TEST_F(ArithmeticOptimizerTest, RemoveIdentityTransposesThroughChain) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs_shape =
      ops::Const(s.WithOpName("inputs_shape"), {8, 3, 28, 28}, {4});
  Output inputs =
      ops::RandomUniform(s.WithOpName("inputs"), inputs_shape, DT_FLOAT);
  Output perm1 = ops::Const(s.WithOpName("perm1"), {0, 2, 3, 1}, {4});
  Output perm2 = ops::Const(s.WithOpName("perm2"), {0, 3, 1, 2}, {4});
  Output transpose1 = ops::Transpose(
      s.WithOpName("transpose1").WithControlDependencies(perm2), inputs, perm1);
  Output identity = ops::Identity(s.WithOpName("id"), transpose1);
  Output transpose2 =
      ops::Transpose(s.WithOpName("transpose2"), identity, perm2);
  Output id1 = ops::Identity(s.WithOpName("id1"), transpose2);

  GrapplerItem item;
  item.fetch = {"id1"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  ArithmeticOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  std::set<string> nodes_after_optimization;
  for (const NodeDef& node : output.node()) {
    nodes_after_optimization.insert(node.name());
    if (node.name() == "id") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "inputs");
    }
    if (node.name() == "id1") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "id");
    }
  }
  EXPECT_EQ(nodes_after_optimization,
            std::set<string>({"id", "id1", "inputs_shape", "inputs"}));
}

TEST_F(ArithmeticOptimizerTest, FoldMulToTransposeConv) {
  for (bool swap_inputs : {false, true}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_FLOAT,
                                     ops::Placeholder::Shape({1, 28, 28, 3}));
    Output scale = ops::Const(s.WithOpName("scale"), 1.0f / 255.0f, {});
    Output scaled_inputs = ops::Multiply(s.WithOpName("scaled_inputs"),
                                         swap_inputs ? scale : inputs,
                                         swap_inputs ? inputs : scale);
    Output perm_nhwc_to_nchw =
        ops::Const(s.WithOpName("perm_nhwc_to_nchw"), {0, 3, 1, 2}, {4});
    Output inputs_nchw = ops::Transpose(s.WithOpName("inputs_nchw"),
                                        scaled_inputs, perm_nhwc_to_nchw);
    Output weights = ops::Const(s.WithOpName("weights"),
                                Input::Initializer(127.0f, {5, 5, 3, 4}));
    Output conv =
        ops::Conv2D(s.WithOpName("conv"), inputs_nchw, weights, {1, 1, 1, 1},
                    "VALID", ops::Conv2D::DataFormat("NCHW"));
    Output outputs = ops::Identity(s.WithOpName("outputs"), conv);

    GrapplerItem item;
    item.fetch = {"outputs"};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    //    LOG(INFO) << "Before:\n" << item.graph.DebugString();
    GraphDef output;
    ArithmeticOptimizer optimizer;
    EnableOnlyFoldMultipleIntoConv(&optimizer);
    OptimizeTwiceAndPrune(&optimizer, &item, &output);

    //    LOG(INFO) << "After:\n"  << output.DebugString();
    NodeMap node_map(&output);
    // `conv` is now a folded convolution with scaled weights.
    const NodeDef* folded_conv = node_map.GetNode(conv.node()->name());
    ASSERT_NE(folded_conv, nullptr);

    const NodeDef* folded_conv_weights =
        node_map.GetNode(folded_conv->input(1));
    ASSERT_NE(folded_conv_weights, nullptr);
    EXPECT_EQ(folded_conv_weights->op(), "Mul");

    // Its input should be a transpose of `inputs`.
    const NodeDef* transpose =
        node_map.GetNode(NodeName(folded_conv->input(0)));
    ASSERT_NE(transpose, nullptr);
    ASSERT_EQ(transpose->input_size(), 2);
    EXPECT_EQ(transpose->input(0), "inputs");
  }
}

TEST_F(ArithmeticOptimizerTest, NotFoldMulAcrossPreservedTranspose) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_FLOAT,
                                   ops::Placeholder::Shape({8, 28, 28, 3}));
  Output scale = ops::Const(s.WithOpName("scale"), 1.0f / 255.0f, {});
  Output scaled_inputs =
      ops::Multiply(s.WithOpName("scaled_inputs"), inputs, scale);
  Output perm_nhwc_to_nchw =
      ops::Const(s.WithOpName("perm_nhwc_to_nchw"), {0, 3, 1, 2}, {4});
  Output inputs_nchw = ops::Transpose(s.WithOpName("inputs_nchw"),
                                      scaled_inputs, perm_nhwc_to_nchw);
  Output weights = ops::Const(s.WithOpName("weights"),
                              Input::Initializer(127.0f, {5, 5, 3, 16}));
  Output conv =
      ops::Conv2D(s.WithOpName("conv"), inputs_nchw, weights, {1, 1, 1, 1},
                  "VALID", ops::Conv2D::DataFormat("NCHW"));
  Output outputs = ops::Identity(s.WithOpName("outputs"), conv);

  Tensor inputs_nchw_tensor(DT_FLOAT, {8, 3, 28, 28});
  memset(const_cast<char*>(inputs_nchw_tensor.tensor_data().data()), 0,
         inputs_nchw_tensor.tensor_data().size());

  GrapplerItem item;
  item.fetch = {"outputs"};
  item.feed = {{"inputs_nchw", inputs_nchw_tensor}};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  NodeMap node_map(&output);
  const NodeDef* inputs_nchw_node_def =
      node_map.GetNode(inputs_nchw.node()->name());
  ASSERT_NE(inputs_nchw_node_def, nullptr);
  ASSERT_EQ(inputs_nchw_node_def->input_size(), 2);
  EXPECT_EQ(NodeName(inputs_nchw_node_def->input(0)),
            scaled_inputs.node()->name());
}

TEST_F(ArithmeticOptimizerTest, FoldMulToConv) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_FLOAT,
                                   ops::Placeholder::Shape({8, 28, 28, 28, 3}));
  Output scale = ops::Const(s.WithOpName("scale"), 1.0f / 255.0f, {});
  Output scaled_inputs =
      ops::Multiply(s.WithOpName("scaled_inputs"), inputs, scale);
  Output weights = ops::Const(s.WithOpName("weights"),
                              Input::Initializer(127.0f, {5, 5, 5, 3, 16}));
  Output conv = ops::Conv3D(s.WithOpName("conv"), scaled_inputs, weights,
                            {1, 1, 1, 1, 1}, "VALID");
  Output outputs = ops::Identity(s.WithOpName("outputs"), conv);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  NodeMap node_map(&output);
  // `conv` is now a folded convolution on `inputs` and scaled weights.
  const NodeDef* folded_conv = node_map.GetNode(conv.node()->name());
  ASSERT_NE(folded_conv, nullptr);
  ASSERT_EQ(folded_conv->input_size(), 2);
  CHECK_EQ(NodeName(folded_conv->input(0)), inputs.node()->name());
  const NodeDef* folded_conv_input_1 =
      node_map.GetNode(NodeName(folded_conv->input(1)));
  ASSERT_NE(folded_conv_input_1, nullptr);
  CHECK_EQ(folded_conv_input_1->op(), "Mul");
}

TEST_F(ArithmeticOptimizerTest, OptimizeCastMulTransposeConv) {
  // This unit test exercises two optimizations, folding mul into conv, and
  // reordering cast and transpose.
  //
  //   Conv2D(Transpose(Mul(Cast(I), S)), W)
  //     =>
  //   Conv2D(Transpose(Cast(I)), W*S)
  //     =>
  //   Conv2D(Cast(Transpose(I)), W*S)
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

  Output inputs =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output cast = ops::Cast(s, inputs, DT_FLOAT);
  Output mul = ops::Mul(s, cast, ops::Const(s, 1.0f / 255.0f));
  Output transpose =
      ops::Transpose(s, mul, ops::Const(s.WithOpName("perm"), {0, 3, 1, 2}));
  Output weights = ops::Const(s.WithOpName("weights"),
                              Input::Initializer(127.0f, {5, 5, 3, 16}));
  Output conv = ops::Conv2D(s, transpose, weights, {1, 1, 1, 1}, "VALID",
                            ops::Conv2D::DataFormat("NCHW"));
  Output outputs = ops::Identity(s.WithOpName("outputs"), conv);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  ArithmeticOptimizer optimizer;  // all optimization stages are on
  OptimizeTwiceAndPrune(&optimizer, &item, &output, /*const_folding=*/true);
  NodeMap node_map(&output);

  // Expected names for reordered cast and transpose.
  const string p = "ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_";
  const string optimized_cast_name = absl::StrCat(p, "float_Cast");
  const string optimized_transpose_name = absl::StrCat(p, "uint8_Transpose");

  // Expected names for folded multiply and conv.
  const string optimized_weights =
      "ArithmeticOptimizer/FoldMultiplyIntoConv_scaled_Conv2D_weights";

  const NodeDef* inputs_node = node_map.GetNode("Placeholder");
  const NodeDef* transpose_node = node_map.GetNode(optimized_transpose_name);
  const NodeDef* cast_node = node_map.GetNode(optimized_cast_name);

  const NodeDef* weights_node = node_map.GetNode(optimized_weights);
  const NodeDef* conv_node = node_map.GetNode("Conv2D");

  ASSERT_NE(inputs_node, nullptr);
  ASSERT_NE(transpose_node, nullptr);
  ASSERT_NE(cast_node, nullptr);
  ASSERT_NE(weights_node, nullptr);
  ASSERT_NE(conv_node, nullptr);

  EXPECT_EQ(output.node_size(), 7);
  ASSERT_EQ(transpose_node->input_size(), 2);
  EXPECT_EQ(transpose_node->input(0), inputs_node->name());
  ASSERT_EQ(cast_node->input_size(), 1);
  EXPECT_EQ(cast_node->input(0), transpose_node->name());
  ASSERT_EQ(conv_node->input_size(), 2);
  EXPECT_EQ(conv_node->input(0), cast_node->name());
  EXPECT_EQ(conv_node->input(1), weights_node->name());
}

TEST_F(ArithmeticOptimizerTest, OptimizeMultipleMulTransposeConv) {
  // This unit test exercises optimization of folding mul into conv for
  // multiple nodes in the graph.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/cpu:0");

  GrapplerItem item;
  Output conv[2];

  for (int i = 0; i < 2; ++i) {
    Output inputs =
        ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({8, 3, 28, 28}));
    Output mul = ops::Mul(s, inputs, ops::Const(s, 1.0f / 255.0f));
    Output weights = ops::Const(s.WithOpName("weights"),
                                Input::Initializer(127.0f, {5, 5, 3, 16}));
    conv[i] = ops::Conv2D(s, mul, weights, {1, 1, 1, 1}, "VALID",
                          ops::Conv2D::DataFormat("NCHW"));
  }
  Output outputs = ops::Add(s.WithOpName("outputs"), conv[0], conv[1]);

  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyFoldMultipleIntoConv(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output, /*const_folding=*/true);

  NodeMap node_map(&output);

  using absl::StrCat;
  const string p = "ArithmeticOptimizer/FoldMultiplyIntoConv_";
  const string optimized_weights = StrCat(p, "scaled_Conv2D_weights");
  const string optimized_weights_1 = StrCat(p, "scaled_Conv2D_1_weights_1");

  const NodeDef* weights_node = node_map.GetNode(optimized_weights);
  const NodeDef* weights_node_1 = node_map.GetNode(optimized_weights_1);
  const NodeDef* conv_node = node_map.GetNode("Conv2D");
  const NodeDef* conv_node_1 = node_map.GetNode("Conv2D_1");

  ASSERT_NE(weights_node, nullptr);
  ASSERT_NE(weights_node_1, nullptr);
  ASSERT_NE(conv_node, nullptr);
  ASSERT_NE(conv_node_1, nullptr);

  ASSERT_EQ(conv_node->input_size(), 2);
  ASSERT_EQ(conv_node_1->input_size(), 2);
  EXPECT_EQ(conv_node->input(1), weights_node->name());
  EXPECT_EQ(conv_node_1->input(1), weights_node_1->name());
}

TEST_F(ArithmeticOptimizerTest, CombineBitcasts) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_UINT8,
                                   ops::Placeholder::Shape({2, 3}));
  Output bc1 = ops::Bitcast(s.WithOpName("bc1"), inputs, DT_QINT8);
  Output bc2 = ops::Bitcast(s.WithOpName("bc2"), bc1, DT_INT8);
  Output outputs = ops::Identity(s.WithOpName("outputs"), bc2);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto x_t = GenerateRandomTensor<DT_UINT8>(TensorShape({2, 3}));
  item.feed = {{"inputs", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantBitcast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Bitcasts combined into a single op and inputs redirected to updated Bitcast
  EXPECT_EQ(output.node_size(), 3);
  EXPECT_EQ(CountOpNodes(output, "Bitcast"), 1);
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "bc2"));

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<int8>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, CombineAndRemoveBitcasts) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_INT8,
                                   ops::Placeholder::Shape({2, 3}));
  Output bc1 = ops::Bitcast(s, inputs, DT_QINT8);
  Output bc2 = ops::Bitcast(s, bc1, DT_INT8);
  Output outputs = ops::Identity(s.WithOpName("outputs"), bc2);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto x_t = GenerateRandomTensor<DT_INT8>(TensorShape({2, 3}));
  item.feed = {{"inputs", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantBitcast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Bitcasts removed and inputs redirected to outputs
  EXPECT_EQ(output.node_size(), 2);
  EXPECT_EQ(CountOpNodes(output, "Bitcast"), 0);
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "outputs"));

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<int8>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs = ops::Placeholder(s.WithOpName("inputs"), DT_INT8,
                                   ops::Placeholder::Shape({2, 3}));
  Output cast = ops::Cast(s, inputs, DT_INT8);
  Output outputs = ops::Identity(s.WithOpName("outputs"), cast);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto x_t = GenerateRandomTensor<DT_INT8>(TensorShape({2, 3}));
  item.feed = {{"inputs", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, item.feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantCast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Cast removed and inputs redirected to outputs
  EXPECT_EQ(output.node_size(), 2);
  EXPECT_EQ(CountOpNodes(output, "Cast"), 0);
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "outputs"));

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<int8>(tensors[0], tensors_expected[0]);
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewriteAddOpsOfIdenticalShape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  tensorflow::Scope sx = s.NewSubScope("x");
  tensorflow::Scope sy = s.NewSubScope("y");

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {2, 2}, DT_FLOAT);
  auto add_bc = ops::Add(sx.WithOpName("Add_bc"), b, c);
  auto add_abc = ops::Add(sy.WithOpName("Add_abc"), a, add_bc);

  auto outputs = ops::Identity(s.WithOpName("outputs"), add_abc);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyAddToAddNCombining(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //
  //     +
  //    / \
  //   a   +         -->    AddN(a, b, c)
  //      / \
  //     b   c
  EXPECT_EQ(output.node_size(), 5);

  NodeMap node_map(&output);

  // check add tree was replaced with AddN
  const NodeDef* collapsed_add =
      node_map.GetNode("y/ArithmeticOptimizer/AddOpsRewrite_Add_abc");
  ASSERT_NE(collapsed_add, nullptr);

  EXPECT_EQ(collapsed_add->op(), "AddN");
  ASSERT_EQ(collapsed_add->input_size(), 3);
  EXPECT_EQ(collapsed_add->input(0), "a");
  EXPECT_EQ(collapsed_add->input(1), "b");
  EXPECT_EQ(collapsed_add->input(2), "c");

  // check output was re-wired to new node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);
  ASSERT_EQ(updated_outputs->input_size(), 1);
  EXPECT_EQ(updated_outputs->input(0), collapsed_add->name());

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewriteMultiplePasses) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {2, 2}, DT_FLOAT);
  auto add_ab = ops::Add(s.WithOpName("Add_ab"), a, b);
  auto add_abc = ops::Add(s.WithOpName("Add_abc"), add_ab, c);

  auto x = ops::Variable(s.WithOpName("x"), {2, 2}, DT_FLOAT);
  auto y = ops::Variable(s.WithOpName("y"), {2, 2}, DT_FLOAT);
  auto z = ops::Variable(s.WithOpName("z"), {2, 2}, DT_FLOAT);
  auto add_xy = ops::Add(s.WithOpName("Add_xy"), x, y);
  auto add_xyz = ops::Add(s.WithOpName("Add_xyz"), add_xy, z);

  auto mul = ops::Multiply(s.WithOpName("Mul"), add_abc, add_xyz);
  auto outputs = ops::Identity(s.WithOpName("outputs"), mul);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto z_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}, {"x", x_t}, {"y", y_t}, {"z", z_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyAddToAddNCombining(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //
  //         *
  //      /     \
  //     +       +                        *
  //    / \     / \                    /     \
  //   +   c   x   + -->    AddN(a, b, c)  AddN(x, y, z))
  //  / \         / \
  // a   b       y   z
  EXPECT_EQ(output.node_size(), 10);

  NodeMap node_map(&output);

  // check left Add subtree replaced with AddN
  const NodeDef* collapsed_left =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_abc");
  ASSERT_NE(collapsed_left, nullptr);

  EXPECT_EQ(collapsed_left->op(), "AddN");
  ASSERT_EQ(collapsed_left->input_size(), 3);
  EXPECT_EQ(collapsed_left->input(0), "a");
  EXPECT_EQ(collapsed_left->input(1), "b");
  EXPECT_EQ(collapsed_left->input(2), "c");

  // check right Add subtree replaced with AddN
  const NodeDef* collapsed_right =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_xyz");
  ASSERT_NE(collapsed_right, nullptr);

  EXPECT_EQ(collapsed_right->op(), "AddN");
  ASSERT_EQ(collapsed_right->input_size(), 3);
  EXPECT_EQ(collapsed_right->input(0), "x");
  EXPECT_EQ(collapsed_right->input(1), "y");
  EXPECT_EQ(collapsed_right->input(2), "z");

  // check that Mul inputs re-wired to new Nodes
  const NodeDef* updated_mul = node_map.GetNode("Mul");
  ASSERT_NE(updated_mul, nullptr);

  EXPECT_EQ(updated_mul->op(), "Mul");
  ASSERT_EQ(updated_mul->input_size(), 2);
  EXPECT_EQ(updated_mul->input(0), collapsed_left->name());
  EXPECT_EQ(updated_mul->input(1), collapsed_right->name());

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewriteAddInputMultipleTimes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {2, 2}, DT_FLOAT);
  auto add_ab = ops::Add(s.WithOpName("Add_ab"), a, b);
  auto add_bc = ops::Add(s.WithOpName("Add_bc"), b, c);
  auto add_all = ops::Add(s.WithOpName("Add_all"), add_ab, add_bc);
  auto outputs = ops::Identity(s.WithOpName("outputs"), add_all);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyAddToAddNCombining(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //
  //     +
  //    / \
  //   +   +     -->    AddN(a, b, b, c)
  //  / \ / \                   ^
  // a   b   c                  b added twice!
  EXPECT_EQ(output.node_size(), 5);

  NodeMap node_map(&output);

  // check Add tree replaced with AddN
  const NodeDef* collapsed_add =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_all");
  ASSERT_NE(collapsed_add, nullptr);

  EXPECT_EQ(collapsed_add->op(), "AddN");
  ASSERT_EQ(collapsed_add->input_size(), 4);
  EXPECT_EQ(collapsed_add->input(0), "a");
  EXPECT_EQ(collapsed_add->input(1), "b");
  EXPECT_EQ(collapsed_add->input(2), "b");
  EXPECT_EQ(collapsed_add->input(3), "c");

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewriteAddOpsOfSymbolicallyEqualShape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  // unknown input shape propagated symbolically through the graph
  auto input = ops::Variable(s.WithOpName("input"), {-1, 2}, DT_FLOAT);

  // [a, b, c] have symbolically equal shapes
  auto a = ops::Sqrt(s.WithOpName("a"), input);
  auto b = ops::Square(s.WithOpName("b"), input);
  auto c = ops::Round(s.WithOpName("c"), input);

  // [add_ab, add_abc] shape must be inferred from inputs
  auto add_ab = ops::Add(s.WithOpName("Add_ab"), a, b);
  auto add_abc = ops::Add(s.WithOpName("Add_abc"), add_ab, c);

  auto outputs = ops::Identity(s.WithOpName("outputs"), add_abc);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {{"input", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyAddToAddNCombining(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //
  //     +
  //    / \
  //   +   c      -->    AddN(a, b, c)
  //  / \
  // a   b
  EXPECT_EQ(output.node_size(), 6);

  NodeMap node_map(&output);

  // check add tree was replaced with AddN
  const NodeDef* collapsed_add =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_abc");
  ASSERT_NE(collapsed_add, nullptr);
  EXPECT_EQ(collapsed_add->op(), "AddN");
  ASSERT_EQ(collapsed_add->input_size(), 3);
  EXPECT_EQ(collapsed_add->input(0), "a");
  EXPECT_EQ(collapsed_add->input(1), "b");
  EXPECT_EQ(collapsed_add->input(2), "c");

  // check output was re-wired to new node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);
  ASSERT_EQ(updated_outputs->input_size(), 1);
  EXPECT_EQ(updated_outputs->input(0), collapsed_add->name());

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewriteMinimizeBCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {32}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {32, 32}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {32, 32, 32}, DT_FLOAT);
  auto add_ab = ops::Add(s.WithOpName("Add_ab"), a, b);
  auto add_abc = ops::Add(s.WithOpName("Add_abc"), add_ab, c);

  auto x = ops::Variable(s.WithOpName("x"), {32}, DT_FLOAT);
  auto y = ops::Variable(s.WithOpName("y"), {32, 32}, DT_FLOAT);
  auto z = ops::Variable(s.WithOpName("z"), {32, 32, 32}, DT_FLOAT);
  auto add_xy = ops::Add(s.WithOpName("Add_xy"), x, y);
  auto add_xyz = ops::Add(s.WithOpName("Add_xyz"), add_xy, z);

  auto add_all = ops::Add(s.WithOpName("AddAll"), add_abc, add_xyz);
  auto outputs = ops::Identity(s.WithOpName("outputs"), add_all);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32, 32}));
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32}));
  auto z_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32, 32}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}, {"x", x_t}, {"y", y_t}, {"z", z_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyAddToAddNCombining(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //  1) [a, x], [b, y], [c, z] - aggregate same shapes first
  //  2) Build an aggregation tree minimizing cost of broadcast
  //
  //         +                              +
  //      /     \                       /       \
  //     +       +                     +       AddN(c, z)
  //    / \     / \                 /     \
  //   +   c   x   + -->    AddN(a, x)  AddN(b, y)
  //  / \         / \
  // a   b       y   z
  EXPECT_EQ(output.node_size(), 12);
  NodeMap node_map(&output);

  // expected names of outer and inner nodes
  string outer_add_name = "ArithmeticOptimizer/AddOpsRewrite_AddAll";
  string outer_0_add_name =
      "ArithmeticOptimizer/AddOpsRewrite_Internal_0_AddAll";
  string inner_0_add_name = "ArithmeticOptimizer/AddOpsRewrite_Leaf_0_AddAll";
  string inner_1_add_name = "ArithmeticOptimizer/AddOpsRewrite_Leaf_1_AddAll";
  string inner_2_add_name = "ArithmeticOptimizer/AddOpsRewrite_Leaf_2_AddAll";

  // Add [a, x] first
  const NodeDef* add_ax_node = node_map.GetNode(inner_0_add_name);
  ASSERT_NE(add_ax_node, nullptr);
  EXPECT_EQ(add_ax_node->op(), "AddN");
  ASSERT_EQ(add_ax_node->input_size(), 2);
  EXPECT_EQ(add_ax_node->input(0), "a");
  EXPECT_EQ(add_ax_node->input(1), "x");

  // Then add [b, y]
  const NodeDef* add_by_node = node_map.GetNode(inner_1_add_name);
  ASSERT_NE(add_by_node, nullptr);
  EXPECT_EQ(add_by_node->op(), "AddN");
  ASSERT_EQ(2, add_by_node->input_size());
  EXPECT_EQ(add_by_node->input(0), "b");
  EXPECT_EQ(add_by_node->input(1), "y");

  // Then add [c, z]
  const NodeDef* add_cz_node = node_map.GetNode(inner_2_add_name);
  ASSERT_NE(add_cz_node, nullptr);
  EXPECT_EQ(add_cz_node->op(), "AddN");
  ASSERT_EQ(add_cz_node->input_size(), 2);
  EXPECT_EQ(add_cz_node->input(0), "c");
  EXPECT_EQ(add_cz_node->input(1), "z");

  // Then add results together starting from smaller shapes [a, x] + [b, y]
  const NodeDef* outer_0_node = node_map.GetNode(outer_0_add_name);
  ASSERT_NE(outer_0_node, nullptr);
  EXPECT_EQ(outer_0_node->op(), "Add");
  ASSERT_EQ(outer_0_node->input_size(), 2);
  EXPECT_EQ(outer_0_node->input(0), inner_0_add_name);
  EXPECT_EQ(outer_0_node->input(1), inner_1_add_name);

  // And finally top level Add node
  const NodeDef* outer_node = node_map.GetNode(outer_add_name);
  ASSERT_NE(outer_node, nullptr);
  EXPECT_EQ(outer_node->op(), "Add");
  ASSERT_EQ(outer_node->input_size(), 2);
  EXPECT_EQ(outer_node->input(0), outer_0_add_name);
  EXPECT_EQ(outer_node->input(1), inner_2_add_name);

  // And outputs reading new top level Add node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);
  ASSERT_EQ(updated_outputs->input_size(), 1);
  EXPECT_EQ(updated_outputs->input(0), outer_add_name);

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewriteMinimizeBCastWithSymbolicShapes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  // We have a small input with one unknown dimension
  auto small = ops::Variable(s.WithOpName("small"), {-1, 1, 1}, DT_DOUBLE);

  // And second input which is larger, but has the same unknown dimension
  // device spec prevents this node from rewriting
  auto d = "/device:CPU:0";
  auto v = ops::Variable(s.WithOpName("v"), {1, 32, 32}, DT_DOUBLE);
  auto large = ops::Add(s.WithOpName("large").WithDevice(d), small, v);

  // [a, c] have {?, 1, 1} shape, [b] has {?, 32, 32}
  auto a = ops::Sqrt(s.WithOpName("a"), small);
  auto b = ops::Square(s.WithOpName("b"), large);
  auto c = ops::Round(s.WithOpName("c"), small);

  // [add_ab, add_abc] shape must be inferred from inputs
  auto add_ab = ops::Add(s.WithOpName("Add_ab"), a, b);
  auto add_abc = ops::Add(s.WithOpName("Add_abc"), add_ab, c);

  auto outputs = ops::Identity(s.WithOpName("outputs"), add_abc);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto s_t = GenerateRandomTensor<DT_DOUBLE>(TensorShape({8, 1, 1}));
  auto v_t = GenerateRandomTensor<DT_DOUBLE>(TensorShape({1, 32, 32}));
  std::vector<std::pair<string, Tensor>> feed = {{"small", s_t}, {"v", v_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyAddToAddNCombining(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur: it's much cheaper to add small
  // tensors, and do the broadcast just once
  //
  //     +                  +
  //    / \                / \
  //   +   c      -->     +   b
  //  / \                / \
  // a   b              a   c
  EXPECT_EQ(output.node_size(), 9);
  NodeMap node_map(&output);

  // expected names of outer and inner nodes
  string outer_add_name = "ArithmeticOptimizer/AddOpsRewrite_Add_abc";
  string inner_add_name = "ArithmeticOptimizer/AddOpsRewrite_Leaf_0_Add_abc";

  // outer Add node
  const NodeDef* outer_add = node_map.GetNode(outer_add_name);
  ASSERT_NE(outer_add, nullptr);
  EXPECT_EQ(outer_add->op(), "Add");
  ASSERT_EQ(outer_add->input_size(), 2);
  EXPECT_EQ(outer_add->input(0), inner_add_name);
  EXPECT_EQ(outer_add->input(1), "b");

  // inner AddN node
  const NodeDef* inner_add = node_map.GetNode(inner_add_name);
  ASSERT_NE(inner_add, nullptr);
  ASSERT_EQ(inner_add->input_size(), 2);
  EXPECT_EQ(inner_add->input(0), "a");
  EXPECT_EQ(inner_add->input(1), "c");

  // check output was re-wired to new node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);
  ASSERT_EQ(updated_outputs->input_size(), 1);
  EXPECT_EQ(updated_outputs->input(0), outer_add_name);

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<double>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveNegation) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Variable(s.WithOpName("x"), {2, 2}, DT_FLOAT);
  auto y = ops::Variable(s.WithOpName("y"), {2, 2}, DT_FLOAT);
  Output neg_x = ops::Neg(s.WithOpName("Neg_x"), x);
  Output neg_y = ops::Neg(s.WithOpName("Neg_y"), y);
  Output add_x_y = ops::Add(s.WithOpName("Add_x_y"), x, y);
  Output add_negx_y = ops::Add(s.WithOpName("Add_negx_y"), neg_x, y);
  Output add_x_negy = ops::Add(s.WithOpName("Add_x_negy"), x, neg_y);
  Output add_negx_negy = ops::Add(s.WithOpName("Add_negx_negy"), neg_x, neg_y);
  Output sub_x_y = ops::Sub(s.WithOpName("Sub_x_y"), x, y);
  Output sub_negx_y = ops::Sub(s.WithOpName("Sub_negx_y"), neg_x, y);
  Output sub_x_negy = ops::Sub(s.WithOpName("Sub_x_negy"), x, neg_y);
  Output sub_negx_negy = ops::Sub(s.WithOpName("Sub_negx_negy"), neg_x, neg_y);
  Output neg_x_with_dep = ops::Neg(
      s.WithOpName("Neg_x_with_dep").WithControlDependencies({add_x_y}), x);
  Output add_negx_with_dep_y =
      ops::Add(s.WithOpName("Add_negx_with_dep_y"), neg_x_with_dep, y);
  auto add_all =
      ops::AddN(s.WithOpName("add_all"),
                {add_x_y, add_negx_y, add_x_negy, add_negx_negy, sub_x_y,
                 sub_negx_y, sub_x_negy, sub_negx_negy, add_negx_with_dep_y});

  GrapplerItem item;
  item.fetch = {"add_all"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {{"x", x_t}, {"y", y_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveNegation(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);

  EXPECT_EQ(output.node_size(), item.graph.node_size());
  int found = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "Add_negx_y") {
      ++found;
      EXPECT_EQ(node.op(), "Sub");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "y");
      EXPECT_EQ(node.input(1), "x");
    } else if (node.name() == "Add_x_negy") {
      ++found;
      EXPECT_EQ(node.op(), "Sub");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      EXPECT_EQ(node.input(1), "y");
    } else if (node.name() == "Add_negx_negy") {
      ++found;
      EXPECT_EQ(node.op(), "Sub");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "Neg_x");
      EXPECT_EQ(node.input(1), "y");
    } else if (node.name() == "Sub_x_negy") {
      ++found;
      EXPECT_EQ(node.op(), "Add");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      EXPECT_EQ(node.input(1), "y");
    } else if (node.name() == "Sub_negx_negy") {
      ++found;
      EXPECT_EQ(node.op(), "Sub");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "y");
      EXPECT_EQ(node.input(1), "x");
    } else if (node.name() == "Add_negx_with_dep_y") {
      ++found;
      EXPECT_EQ(node.op(), "Sub");
      ASSERT_EQ(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "y");
      EXPECT_EQ(node.input(1), "x");
      EXPECT_EQ(node.input(2), "^Add_x_y");
    }
  }
  EXPECT_EQ(found, 6);

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, ConvertSqrtDivToRsqrtMul) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  auto y = ops::Const(s.WithOpName("y"), {3.0f, 4.0f}, {1, 2});
  Output sqrt_y = ops::Sqrt(s.WithOpName("sqrt_y"), y);
  Output div_x_sqrt_y = ops::Div(s.WithOpName("output"), x, sqrt_y);

  GrapplerItem item;
  item.fetch = {"output"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlySqrtDivToRsqrtMul(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "output") {
      EXPECT_EQ(node.op(), "Mul");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      EXPECT_EQ(node.input(1), "sqrt_y");
    } else if (node.name() == "sqrt_y") {
      EXPECT_EQ(node.op(), "Rsqrt");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "y");
    }
  }
}

TEST_F(ArithmeticOptimizerTest, DoNotConvertSqrtDivToRsqrtMulDivisorFetchNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output floats = ops::Const(s.WithOpName("floats"),
                             {0.7423212f, 0.19757693f, 0.53124744f}, {1, 3});
  Output output0 = ops::Sqrt(s.WithOpName("output0"), floats);
  Output const1 = ops::Const(s.WithOpName("const1"), 1.0f, {3});
  Output mul1 = ops::Multiply(s.WithOpName("mul1"), const1, 0.5f);
  Output grad = ops::Div(s.WithOpName("grad"), mul1, output0);

  GrapplerItem item;
  item.fetch = {"grad", "output0"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 2);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlySqrtDivToRsqrtMul(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 2);

  for (int i = 0; i < tensors.size(); i++) {
    EXPECT_EQ(tensors[i].NumElements(), tensors_expected[i].NumElements());
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "grad") {
      EXPECT_EQ(node.op(), "Div");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "mul1");
      EXPECT_EQ(node.input(1), "output0");
    } else if (node.name() == "output0") {
      EXPECT_EQ(node.op(), "Sqrt");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "floats");
    }
  }
}

TEST_F(ArithmeticOptimizerTest, ConvertSqrtDivToRsqrtMulExcludeFloorDiv) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  auto y = ops::Const(s.WithOpName("y"), {3.0f, 4.0f}, {1, 2});
  Output sqrt_y = ops::Sqrt(s.WithOpName("sqrt_y"), y);
  Output div_x_sqrt_y = ops::FloorDiv(s.WithOpName("output"), x, sqrt_y);

  GrapplerItem item;
  item.fetch = {"output"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlySqrtDivToRsqrtMul(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "output") {
      EXPECT_EQ(node.op(), "FloorDiv");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      EXPECT_EQ(node.input(1), "sqrt_y");
    } else if (node.name() == "sqrt_y") {
      EXPECT_EQ(node.op(), "Sqrt");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "y");
    }
  }
}

TEST_F(ArithmeticOptimizerTest, FuseSquaredDiff) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  auto y = ops::Const(s.WithOpName("y"), {3.0f, 4.0f}, {1, 2});
  Output sub_x_y = ops::Sub(s.WithOpName("sub_x_y"), x, y);
  Output square_sub_x_y = ops::Square(s.WithOpName("output"), sub_x_y);

  GrapplerItem item;
  item.fetch = {"output"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  const auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyFuseSquaredDiff(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  const auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "output") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "sub_x_y");
    } else if (node.name() == "sub_x_y") {
      EXPECT_EQ(node.op(), "SquaredDifference");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      EXPECT_EQ(node.input(1), "y");
    }
  }
}

TEST_F(ArithmeticOptimizerTest, DoNotFuseSquaredDiffFetchNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  auto y = ops::Const(s.WithOpName("y"), {3.0f, 4.0f}, {1, 2});
  Output sub_x_y = ops::Sub(s.WithOpName("sub_x_y"), x, y);
  Output square_sub_x_y = ops::Square(s.WithOpName("output"), sub_x_y);

  GrapplerItem item;
  item.fetch = {"output", "sub_x_y"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  const auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 2);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyFuseSquaredDiff(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  const auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 2);

  for (int i = 0; i < tensors.size(); i++) {
    EXPECT_EQ(tensors[i].NumElements(), tensors_expected[i].NumElements());
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "output") {
      EXPECT_EQ(node.op(), "Square");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "sub_x_y");
    } else if (node.name() == "sub_x_y") {
      EXPECT_EQ(node.op(), "Sub");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      EXPECT_EQ(node.input(1), "y");
    }
  }
}

TEST_F(ArithmeticOptimizerTest, ConvertLogSoftmax) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output softmax = ops::Softmax(s.WithOpName("softmax"), x);
  Output logsoftmax = ops::Log(s.WithOpName("output"), softmax);

  GrapplerItem item;
  item.fetch = {"output"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  const auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyLogSoftmax(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  const auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  EXPECT_EQ(output.node_size(), item.graph.node_size() - 1);
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "output") {
      EXPECT_EQ(node.op(), "LogSoftmax");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "x");
    }
  }
}

TEST_F(ArithmeticOptimizerTest, DoNotConvertLogSoftmaxArgFetchNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output floats = ops::Const(s.WithOpName("floats"),
                             {0.7423212f, 0.19757693f, 0.53124744f}, {1, 3});
  Output softmax = ops::Softmax(s.WithOpName("softmax"), floats);
  Output final_output = ops::Log(s.WithOpName("final_output"), softmax);

  GrapplerItem item;
  item.fetch = {"softmax", "final_output"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  const auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 2);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyLogSoftmax(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);
  const auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 2);

  // Should be a NoOp since we are not allowed to change the output of fetch
  // nodes.
  VerifyGraphsMatch(item.graph, output, __LINE__);

  for (int i = 0; i < tensors.size(); i++) {
    EXPECT_EQ(tensors[i].NumElements(), tensors_expected[i].NumElements());
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }
}

TEST_F(ArithmeticOptimizerTest, ConvertPow) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  auto y2 = ops::Const(s.WithOpName("y2"), {2.0f, 2.0f}, {1, 2});
  auto y3 = ops::Const(s.WithOpName("y3"), {3.0f, 3.0f}, {1, 2});
  auto y1 = ops::Const(s.WithOpName("y1"), {1.0f, 1.0f}, {1, 2});
  auto yPoint5 = ops::Const(s.WithOpName("y.5"), {0.5f, 0.5f}, {1, 2});
  auto y0 = ops::Const(s.WithOpName("y0"), {0.0f, 0.0f}, {1, 2});
  auto y_Point5 = ops::Const(s.WithOpName("y_.5"), {-0.5f, -0.5f}, {1, 2});
  auto y_1 = ops::Const(s.WithOpName("y_1"), {-1.0f, -1.0f}, {1, 2});
  auto y = ops::Const(s.WithOpName("y"), {3.0f, 4.0f}, {1, 2});
  auto z = ops::Const(s.WithOpName("z"), {42.0f}, {});
  auto ones = ops::Const(s.WithOpName("ones"), {1.0f, 1.0f, 1.0f}, {1, 3});
  auto zeros = ops::Const(s.WithOpName("zeros"), {0.0f, 0.0f, 0.0f}, {1, 3});
  Output out2 = ops::Pow(s.WithOpName("out2"), x, y2);
  Output out3 =
      ops::Pow(s.WithOpName("out3").WithDevice("/device:CPU:0"), x, y3);
  Output out1 = ops::Pow(s.WithOpName("out1"), x, y1);
  Output outPoint5 = ops::Pow(s.WithOpName("out.5"), x, yPoint5);
  Output out0 = ops::Pow(s.WithOpName("out0"), x, y0);
  Output out_Point5 = ops::Pow(s.WithOpName("out_.5"), x, y_Point5);
  Output out_1 = ops::Pow(s.WithOpName("out_1"), x, y_1);
  Output out = ops::Pow(s.WithOpName("out"), x, y);
  Output out_bcast1 = ops::Pow(s.WithOpName("out_bcast1"), z, ones);
  Output out_bcast2 = ops::Pow(s.WithOpName("out_bcast2"), z, zeros);

  GrapplerItem item;
  item.fetch = {"out2",   "out3",  "out1", "out.5",      "out0",
                "out_.5", "out_1", "out",  "out_bcast1", "out_bcast2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 10);

  GraphDef got;
  ArithmeticOptimizer optimizer;
  EnableOnlyConvertPow(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &got);
  auto tensors = EvaluateNodes(got, item.fetch);
  ASSERT_EQ(tensors.size(), 10);

  for (int i = 0; i < tensors.size(); ++i) {
    EXPECT_EQ(tensors[i].NumElements(), tensors_expected[i].NumElements());
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }

  GraphDef want;
  AddNode("x", "Const", {}, {}, &want);
  AddNode("y", "Const", {}, {}, &want);
  AddNode("z", "Const", {}, {}, &want);
  AddNode("ones", "Const", {}, {}, &want);
  AddNode("zeros", "Const", {}, {}, &want);
  AddNode("out2", "Square", {"x"}, {}, &want);
  AddNode("ArithmeticOptimizer/ConvertPow__inner_out3", "Square", {"x"}, {},
          &want)
      ->set_device("/device:CPU:0");
  AddNode("out3", "Mul", {"x", "ArithmeticOptimizer/ConvertPow__inner_out3"},
          {}, &want)
      ->set_device("/device:CPU:0");
  AddNode("out1", "Identity", {"x"}, {}, &want);
  AddNode("out.5", "Sqrt", {"x"}, {}, &want);
  AddNode("out0", "Const", {AsControlDependency("x")}, {}, &want);
  AddNode("out_.5", "Rsqrt", {"x"}, {}, &want);
  AddNode("out_1", "Reciprocal", {"x"}, {}, &want);
  AddNode("out", "Pow", {"x", "y"}, {}, &want);
  AddNode("out_bcast1", "Pow", {"z", "ones"}, {}, &want);
  AddNode("out_bcast2", "Pow", {"z", "zeros"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ArithmeticOptimizerTest, Log1p) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto x1 = ops::Const(s.WithOpName("x1"), {1.0f, 1.0f}, {1, 2});
  auto x2 = ops::Const(s.WithOpName("x2"), {2.0f, 2.0f}, {1, 2});
  auto x3 = ops::Const(s.WithOpName("x3"), {3.0f, 3.0f}, {1, 2});
  auto a12 = ops::Add(s.WithOpName("a12").WithControlDependencies(x3), x1, x2);
  auto a23 = ops::Add(s.WithOpName("a23"), x2, x3);
  Output out1 = ops::Log(s.WithOpName("out1"), a12);
  Output out2 = ops::Log(s.WithOpName("out2"), a23);

  GrapplerItem item;
  item.fetch = {"out1", "out2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 2);

  GraphDef got;
  ArithmeticOptimizer optimizer;
  EnableOnlyLog1p(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &got);
  auto tensors = EvaluateNodes(got, item.fetch);
  ASSERT_EQ(tensors.size(), 2);

  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(tensors[i].NumElements(), tensors_expected[i].NumElements());
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }

  GraphDef want;
  AddNode("x2", "Const", {}, {}, &want);
  AddNode("x3", "Const", {}, {}, &want);
  AddNode("a23", "Add", {"x2", "x3"}, {}, &want);
  AddNode("out1", "Log1p", {"x2", AsControlDependency("x3")}, {}, &want);
  AddNode("out2", "Log", {"a23"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ArithmeticOptimizerTest, Expm1) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto x1 = ops::Const(s.WithOpName("x1"), {2.0f, 2.0f}, {1, 2});
  auto x2 = ops::Const(s.WithOpName("x2"), {1.0f, 1.0f}, {1, 2});
  auto x3 = ops::Const(s.WithOpName("x3"), {3.0f, 3.0f}, {1, 2});
  auto exp1 = ops::Exp(s.WithOpName("exp1").WithControlDependencies(x3), x1);
  Output out1 = ops::Sub(s.WithOpName("out1"), exp1, x2);
  Output out2 = ops::Sub(s.WithOpName("out2"), exp1, x3);

  GrapplerItem item;
  item.fetch = {"out1", "out2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 2);

  GraphDef got;
  ArithmeticOptimizer optimizer;
  EnableOnlyExpm1(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &got);
  auto tensors = EvaluateNodes(got, item.fetch);
  ASSERT_EQ(tensors.size(), 2);

  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(tensors[i].NumElements(), tensors_expected[i].NumElements());
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }

  GraphDef want;
  AddNode("x1", "Const", {}, {}, &want);
  AddNode("x3", "Const", {}, {}, &want);
  AddNode("exp1", "Exp", {"x1", AsControlDependency("x3")}, {}, &want);
  AddNode("out1", "Expm1", {"x1", AsControlDependency("x3")}, {}, &want);
  AddNode("out2", "Sub", {"exp1", "x3"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ArithmeticOptimizerTest, MinimizeBroadcasts_SimpleSwap) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {32}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {32, 32}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {32}, DT_FLOAT);

  auto mul1 = ops::Mul(s.WithOpName("mul1"), a, b);
  auto mul2 = ops::Mul(s.WithOpName("mul2"), mul1, c);

  auto outputs = ops::Identity(s.WithOpName("outputs"), mul2);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyMinimizeBroadcasts(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //
  //     *                  *
  //    / \                / \
  //   *   c      -->     *   b
  //  / \                / \
  // a   b              a   c
  NodeMap node_map(&output);

  const NodeDef* mul1_node = node_map.GetNode("mul1");
  ASSERT_NE(mul1_node, nullptr);
  ASSERT_EQ(mul1_node->input_size(), 2);
  EXPECT_EQ(mul1_node->input(0), "a");
  EXPECT_EQ(mul1_node->input(1), "c");

  const NodeDef* mul2_node = node_map.GetNode("mul2");
  ASSERT_NE(mul2_node, nullptr);
  ASSERT_EQ(mul2_node->input_size(), 2);
  EXPECT_EQ(mul2_node->input(0), "mul1");
  EXPECT_EQ(mul2_node->input(1), "b");

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, MinimizeBroadcasts_FlattenTallGraph) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {32}, DT_DOUBLE);
  auto b = ops::Variable(s.WithOpName("b"), {32, 32}, DT_DOUBLE);
  auto c = ops::Variable(s.WithOpName("c"), {32}, DT_DOUBLE);
  auto d = ops::Variable(s.WithOpName("d"), {32}, DT_DOUBLE);
  auto e = ops::Variable(s.WithOpName("e"), {32}, DT_DOUBLE);

  auto mul1 = ops::Mul(s.WithOpName("mul1"), a, b);
  auto mul2 = ops::Mul(s.WithOpName("mul2"), mul1, c);
  auto mul3 = ops::Mul(s.WithOpName("mul3"), mul2, d);
  auto mul4 = ops::Mul(s.WithOpName("mul4"), mul3, e);

  auto outputs = ops::Identity(s.WithOpName("outputs"), mul4);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto a_t = GenerateRandomTensor<DT_DOUBLE>(TensorShape({32}));
  auto b_t = GenerateRandomTensor<DT_DOUBLE>(TensorShape({32, 32}));
  auto c_t = GenerateRandomTensor<DT_DOUBLE>(TensorShape({32}));
  auto d_t = GenerateRandomTensor<DT_DOUBLE>(TensorShape({32}));
  auto e_t = GenerateRandomTensor<DT_DOUBLE>(TensorShape({32}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}, {"d", d_t}, {"e", e_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyMinimizeBroadcasts(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur: Graph is "flattened" and
  // largest shape pushed to the top.
  //
  //          *
  //        /   \
  //       *     e                *
  //      /  \                  /   \
  //     *    d               *      b
  //    / \                 /  \
  //   *   c      -->     *      *
  //  / \                / \    / \
  // a   b              a   c  d   e
  NodeMap node_map(&output);

  const NodeDef* mul1_node = node_map.GetNode("mul1");
  ASSERT_NE(mul1_node, nullptr);
  ASSERT_EQ(mul1_node->input_size(), 2);
  EXPECT_EQ(mul1_node->input(0), "a");
  EXPECT_EQ(mul1_node->input(1), "c");

  const NodeDef* mul2_node = node_map.GetNode("mul2");
  ASSERT_NE(mul2_node, nullptr);
  ASSERT_EQ(mul2_node->input_size(), 2);
  EXPECT_EQ(mul2_node->input(0), "d");
  EXPECT_EQ(mul2_node->input(1), "e");

  const NodeDef* mul3_node = node_map.GetNode("mul3");
  ASSERT_NE(mul3_node, nullptr);
  ASSERT_EQ(mul3_node->input_size(), 2);
  EXPECT_EQ(mul3_node->input(0), "mul1");
  EXPECT_EQ(mul3_node->input(1), "mul2");

  const NodeDef* mul4_node = node_map.GetNode("mul4");
  ASSERT_NE(mul4_node, nullptr);
  ASSERT_EQ(mul4_node->input_size(), 2);
  EXPECT_EQ(mul4_node->input(0), "mul3");
  EXPECT_EQ(mul4_node->input(1), "b");

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<double>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, MinimizeBroadcasts_BuildTreeUp) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  // [a, b, c] - scalars, [d] - matrix
  auto a = ops::Variable(s.WithOpName("a"), {32}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {32}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {32}, DT_FLOAT);
  auto d = ops::Variable(s.WithOpName("D"), {32, 32}, DT_FLOAT);

  auto mul1 = ops::Mul(s.WithOpName("mul1"), a, b);
  auto mul2 = ops::Mul(s.WithOpName("mul2"), c, d);
  auto mul3 = ops::Mul(s.WithOpName("mul3"), mul1, mul2);

  auto outputs = ops::Identity(s.WithOpName("outputs"), mul3);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto d_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}, {"D", d_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyMinimizeBroadcasts(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);

  // We expect the following rewrite(s) to occur:
  //
  //                              *
  //                            /  \
  //       *                   *    D
  //     /   \                / \
  //    *     *      ->      *   c
  //   / \   / \            / \
  //  a   b c   D          a   b
  NodeMap node_map(&output);

  const NodeDef* mul1_node = node_map.GetNode("mul2");
  ASSERT_NE(mul1_node, nullptr);
  ASSERT_EQ(mul1_node->input_size(), 2);
  EXPECT_EQ(mul1_node->input(0), "a");
  EXPECT_EQ(mul1_node->input(1), "b");

  const NodeDef* mul2_node = node_map.GetNode("mul1");
  ASSERT_NE(mul2_node, nullptr);
  ASSERT_EQ(mul2_node->input_size(), 2);
  EXPECT_EQ(mul2_node->input(0), "mul2");
  EXPECT_EQ(mul2_node->input(1), "c");

  const NodeDef* mul3_node = node_map.GetNode("mul3");
  ASSERT_NE(mul3_node, nullptr);
  ASSERT_EQ(mul3_node->input_size(), 2);
  EXPECT_EQ(mul3_node->input(0), "D");
  EXPECT_EQ(mul3_node->input(1), "mul1");

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, DoNotHoistReluFromConcat) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output weights1 = ops::Const(s.WithOpName("weights1"),
                               Input::Initializer(1.0f, {5, 5, 3, 4}));
  Output weights2 = ops::Const(s.WithOpName("weights2"),
                               Input::Initializer(2.0f, {5, 5, 3, 4}));
  Output biases =
      ops::Const(s.WithOpName("biases"), Input::Initializer(2.0f, {4}));
  Output axis = ops::Const(s.WithOpName("axis"), 3, {});
  Output input = ops::Const(s.WithOpName("input"),
                            Input::Initializer(1.0f, {1, 28, 28, 3}));
  Output branch1 =
      ops::Conv2D(s.WithOpName("conv1"), input, weights1, {1, 1, 1, 1}, "SAME");
  branch1 = ops::BiasAdd(s.WithOpName("biasadd1"), branch1, biases);
  branch1 = ops::Relu(s.WithOpName("relu1"), branch1);
  Output branch2 =
      ops::Conv2D(s.WithOpName("conv2"), input, weights2, {1, 1, 1, 1}, "SAME");
  branch2 = ops::BiasAdd(s.WithOpName("biasadd2"), branch2, biases);
  branch2 = ops::Relu(s.WithOpName("relu2"), branch2);
  Output concat = ops::Concat(s.WithOpName("concat"), {branch1, branch2}, axis);
  Output output = ops::Identity(s.WithOpName("output"), concat);

  GrapplerItem item;
  item.fetch = {"output"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef new_graph;
  ArithmeticOptimizer optimizer;
  OptimizeAndPrune(&optimizer, &item, &new_graph);

  // Verify that the two Relus are not hoisted.
  EXPECT_EQ(CountOpNodes(new_graph, "Relu"), 2);

  auto tensors = EvaluateNodes(new_graph, item.fetch);
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }
}

TEST_F(ArithmeticOptimizerTest, HoistCWiseUnaryFromConcat) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 3.14f, {32});
  Output b = ops::Const(s.WithOpName("b"), 1.0f, {32});
  Output c = ops::Const(s.WithOpName("c"), 42.0f, {32});
  Output axis = ops::Const(s.WithOpName("axis"), 0, {});
  Output ctrl1 = ops::Const(s.WithOpName("ctrl1"), 1, {});
  Output ctrl2 = ops::Const(s.WithOpName("ctrl2"), 2, {});
  Output ctrl3 = ops::Const(s.WithOpName("ctrl3"), 3, {});
  // Test case with chains of length 1.
  // Rewrites
  //       Concat({Exp(a), Exp(b), Exp(c)})
  // into
  //       Exp(Concat({a, b, c})).
  Output sin_a =
      ops::Sin(s.WithOpName("sin_a").WithControlDependencies(ctrl3), a);
  Output exp_a =
      ops::Exp(s.WithOpName("exp_a").WithControlDependencies(ctrl1), sin_a);
  Output exp_b = ops::Exp(s.WithOpName("exp_b"), b);
  Output exp_c =
      ops::Exp(s.WithOpName("exp_c").WithControlDependencies(ctrl2), c);
  Output concat =
      ops::Concat(s.WithOpName("concat"), {exp_a, exp_b, exp_c}, axis);
  Output id = ops::Identity(s.WithOpName("id"), concat);

  // Test case with chains of length 2.
  // Rewrites
  //       Concat({Cos(Exp(a)), Cos(Exp(b)), Cos(Exp(c))})
  // into
  //       Cos(Exp(Concat({a, b, c}))).
  Output exp_a2 =
      ops::Exp(s.WithOpName("exp_a2").WithControlDependencies(ctrl1), sin_a);
  Output exp_b2 = ops::Exp(s.WithOpName("exp_b2"), b);
  Output exp_c2 =
      ops::Exp(s.WithOpName("exp_c2").WithControlDependencies(ctrl2), c);
  Output cos_exp_a2 = ops::Cos(
      s.WithOpName("cos_exp_a2").WithControlDependencies(ctrl1), exp_a2);
  Output cos_exp_b2 = ops::Cos(
      s.WithOpName("cos_exp_b2").WithControlDependencies(ctrl3), exp_b2);
  Output cos_exp_c2 = ops::Cos(s.WithOpName("cos_exp_c2"), exp_c2);
  Output concat2 = ops::Concat(s.WithOpName("concat2"),
                               {cos_exp_a2, cos_exp_b2, cos_exp_c2}, axis);
  Output id2 = ops::Identity(s.WithOpName("id2"), concat2);
  GrapplerItem item;
  item.fetch = {"id", "id2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyHoistCWiseUnaryChains(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "concat") {
      ASSERT_EQ(node.input_size(), 4);
      EXPECT_EQ(node.input(0), "sin_a");
      EXPECT_EQ(node.input(1), "b");
      EXPECT_EQ(node.input(2), "c");
      EXPECT_EQ(node.input(3), "axis");
      found++;
    }
    if (node.name() == "exp_a") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "concat");
      found++;
    }
    if (node.name() == "id") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "exp_a");
      found++;
    }

    if (node.name() == "concat2") {
      ASSERT_EQ(node.input_size(), 4);
      EXPECT_EQ(node.input(0), "sin_a");
      EXPECT_EQ(node.input(1), "b");
      EXPECT_EQ(node.input(2), "c");
      EXPECT_EQ(node.input(3), "axis");
      found++;
    }
    if (node.name() == "exp_a2") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "concat2");
      found++;
    }
    if (node.name() == "cos_exp_a2") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "exp_a2");
      found++;
    }
    if (node.name() == "id2") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "cos_exp_a2");
      found++;
    }
  }
  EXPECT_EQ(found, 7);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }
}

TEST_F(ArithmeticOptimizerTest, HoistCWiseUnaryIntoSplit) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), 3.1415f, {32});
  Output axis = ops::Const(s.WithOpName("axis"), 0, {});
  Output ctrl1 = ops::Const(s.WithOpName("ctrl1"), 1, {});
  Output ctrl2 = ops::Const(s.WithOpName("ctrl2"), 2, {});
  Output ctrl3 = ops::Const(s.WithOpName("ctrl3"), 3, {});
  // Test case with chains of length 1.
  // Rewrites
  //          [Sin(y) for y in Split(x)]
  // into
  //          [y for y in Split(Sin(x))].
  ops::Split split1(s.WithOpName("split1"), axis, x, 2);
  Output sin_a =
      ops::Sin(s.WithOpName("sin_a").WithControlDependencies(ctrl1), split1[0]);
  Output id_a = ops::Identity(s.WithOpName("id_a"), sin_a);
  Output sin_b = ops::Sin(s.WithOpName("sin_b"), split1[1]);
  Output exp_b = ops::Exp(s.WithOpName("exp_b"), sin_b);
  Output id_b = ops::Identity(s.WithOpName("id_b"), exp_b);

  // Test case with SplitV and chains of length 2.
  // Rewrites
  //          [Cos(Exp(y)) for y in Split(x)]
  // into
  //          [y for y in Split(Cos(Exp(x)))].
  Output size_splits2 = ops::Const(s.WithOpName("size_splits2"), {20, 12}, {2});
  ops::SplitV split2(s.WithOpName("split2"), x, size_splits2, axis, 2);
  Output exp_a2 = ops::Exp(
      s.WithOpName("exp_a2").WithControlDependencies(ctrl1), split2[0]);
  Output exp_b2 = ops::Exp(s.WithOpName("exp_b2"), split2[1]);
  Output cos_exp_a2 = ops::Cos(
      s.WithOpName("cos_exp_a2").WithControlDependencies(ctrl2), exp_a2);
  Output cos_exp_b2 = ops::Cos(
      s.WithOpName("cos_exp_b2").WithControlDependencies(ctrl3), exp_b2);
  Output id_a2 = ops::Identity(s.WithOpName("id_a2"), cos_exp_a2);
  Output id_b2 = ops::Identity(s.WithOpName("id_b2"), cos_exp_b2);

  GrapplerItem item;
  item.fetch = {"id_a", "id_b", "id_a2", "id_b2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyHoistCWiseUnaryChains(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  int found = 0;
  for (const NodeDef& node : output.node()) {
    // The following 6 nodes should be pruned.
    EXPECT_NE(node.name(), "sin_a");
    EXPECT_NE(node.name(), "sin_b");
    EXPECT_NE(node.name(), "exp_a2");
    EXPECT_NE(node.name(), "exp_b2");
    EXPECT_NE(node.name(), "cos_exp_a2");
    EXPECT_NE(node.name(), "cos_exp_b2");

    if (node.name() == "split1") {
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "axis");
      EXPECT_EQ(node.input(1), "ArithmeticOptimizer/_sin_a_split1");
      found++;
    }
    if (node.name() == "ArithmeticOptimizer/_sin_a_split1") {
      EXPECT_EQ(node.op(), "Sin");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "x");
      found++;
    }
    if (node.name() == "id_a") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "split1");
      found++;
    }
    if (node.name() == "exp_b") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "split1:1");
      found++;
    }
    if (node.name() == "id_b") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "exp_b");
      found++;
    }
    if (node.name() == "ArithmeticOptimizer/_exp_a2_split2") {
      EXPECT_EQ(node.op(), "Exp");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "x");
      found++;
    }
    if (node.name() == "ArithmeticOptimizer/_cos_exp_a2_split2") {
      EXPECT_EQ(node.op(), "Cos");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "ArithmeticOptimizer/_exp_a2_split2");
      found++;
    }
    if (node.name() == "split2") {
      ASSERT_EQ(node.input_size(), 3);
      EXPECT_EQ(node.input(0), "ArithmeticOptimizer/_cos_exp_a2_split2");
      EXPECT_EQ(node.input(1), "size_splits2");
      EXPECT_EQ(node.input(2), "axis");
      found++;
    }
    if (node.name() == "id_a2") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "split2");
      found++;
    }
    if (node.name() == "id_b2") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "split2:1");
      found++;
    }
  }
  EXPECT_EQ(found, 10);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }
}

TEST_F(ArithmeticOptimizerTest, RemoveIdempotent) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 3.14f, {32});
  Output sn1 = ops::Snapshot(s.WithOpName("sn1"), a);
  Output sn2 = ops::Snapshot(s.WithOpName("sn2"), sn1);
  Output out1 = ops::Identity(s.WithOpName("out1"), sn2);
  Output id1 = ops::Identity(s.WithOpName("id1"), a);
  Output id2 = ops::Identity(s.WithOpName("id2"), id1);
  Output out2 = ops::Identity(s.WithOpName("out2"), id2);
  GrapplerItem item;
  item.fetch = {"out1", "out2"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdempotent(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);

  EXPECT_EQ(7, output.node_size());
  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "out1") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "sn1");
      found++;
    } else if (node.name() == "out2") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "id1");
      found++;
    } else if (node.name() == "sn1") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "a");
      found++;
    }
  }
  EXPECT_EQ(found, 3);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors[i], tensors_expected[i], 1e-6);
  }
}

TEST_F(ArithmeticOptimizerTest, RemoveLogicalNot) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 3.14f, {32});
  Output b = ops::Const(s.WithOpName("b"), -3.14f, {32});
  Output eq = ops::Equal(s.WithOpName("eq"), a, b);
  Output neq = ops::NotEqual(s.WithOpName("neq"), a, b);
  Output lt = ops::Less(s.WithOpName("lt"), a, b);
  Output le = ops::LessEqual(s.WithOpName("le"), a, b);
  Output gt = ops::Greater(s.WithOpName("gt"), a, b);
  Output ge = ops::GreaterEqual(s.WithOpName("ge"), a, b);
  // not_eq is reserved
  Output not_eq1 = ops::LogicalNot(s.WithOpName("not_eq1"), eq);
  Output not_neq = ops::LogicalNot(s.WithOpName("not_neq"), neq);
  Output not_lt = ops::LogicalNot(s.WithOpName("not_lt"), lt);
  Output not_le = ops::LogicalNot(s.WithOpName("not_le"), le);
  Output not_gt = ops::LogicalNot(s.WithOpName("not_gt"), gt);
  Output not_ge = ops::LogicalNot(s.WithOpName("not_ge"), ge);
  Output id_not_eq = ops::Identity(s.WithOpName("id_not_eq"), not_eq1);
  Output id_not_neq = ops::Identity(s.WithOpName("id_not_neq"), not_neq);
  Output id_not_lt = ops::Identity(s.WithOpName("id_not_lt"), not_lt);
  Output id_not_le = ops::Identity(s.WithOpName("id_not_le"), not_le);
  Output id_not_gt = ops::Identity(s.WithOpName("id_not_gt"), not_gt);
  Output id_not_ge = ops::Identity(s.WithOpName("id_not_ge"), not_ge);

  GrapplerItem item;
  item.fetch = {"id_not_eq", "id_not_neq", "id_not_lt",
                "id_not_le", "id_not_gt",  "id_not_ge"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveLogicalNot(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);

  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "id_not_eq") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "eq");
      ++found;
    }
    if (node.name() == "id_not_neq") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "neq");
      ++found;
    }
    if (node.name() == "id_not_lt") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "lt");
      ++found;
    }
    if (node.name() == "id_not_le") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "le");
      ++found;
    }
    if (node.name() == "id_not_gt") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "gt");
      ++found;
    }
    if (node.name() == "id_not_ge") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "ge");
      ++found;
    }

    if (node.name() == "eq") {
      EXPECT_EQ(node.op(), "NotEqual");
      ++found;
    }
    if (node.name() == "neq") {
      EXPECT_EQ(node.op(), "Equal");
      ++found;
    }
    if (node.name() == "lt") {
      EXPECT_EQ(node.op(), "GreaterEqual");
      ++found;
    }
    if (node.name() == "le") {
      EXPECT_EQ(node.op(), "Greater");
      ++found;
    }
    if (node.name() == "gt") {
      EXPECT_EQ(node.op(), "LessEqual");
      ++found;
    }
    if (node.name() == "ge") {
      EXPECT_EQ(node.op(), "Less");
      ++found;
    }
  }
  EXPECT_EQ(found, 12);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorEqual<bool>(tensors[i], tensors_expected[i]);
  }
}

TEST_F(ArithmeticOptimizerTest, OptimizeMaxOrMinOfMonotonicElementWise) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output sqrt = ops::Sqrt(s.WithOpName("sqrt"), x);
  Output reduce_max = ops::Max(s.WithOpName("reduce_max"), sqrt, {0});
  Output final_out = ops::Identity(s.WithOpName("final_out"), reduce_max);

  GrapplerItem item;
  item.fetch = {"final_out"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  // Check if the inputs are switched
  int required_node_count = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "sqrt") {
      EXPECT_EQ(node.op(), "Sqrt");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "reduce_max");
      ++required_node_count;
    } else if (node.name() == "reduce_max") {
      EXPECT_EQ(node.op(), "Max");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      ++required_node_count;
    }
  }
  EXPECT_EQ(required_node_count, 2);
}

TEST_F(ArithmeticOptimizerTest, OptimizeArgMaxOrArgMinOfMonotonicElementWise) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  const auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output sqrt = ops::Sqrt(s.WithOpName("sqrt"), x);
  Output arg_max = ops::ArgMax(s.WithOpName("arg_max"), sqrt, 1);
  Output final_out = ops::Identity(s.WithOpName("final_out"), arg_max);

  GrapplerItem item;
  item.fetch = {"final_out"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  const auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  const auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorEqual<int64>(tensors[0], tensors_expected[0]);
  EXPECT_EQ(output.node_size(), item.graph.node_size() - 1);
  // Check if the inputs are switched
  int required_node_count = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "final_out") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "arg_max");
      ++required_node_count;
    } else if (node.name() == "arg_max") {
      EXPECT_EQ(node.op(), "ArgMax");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      ++required_node_count;
    }
  }
  EXPECT_EQ(required_node_count, 2);
}

TEST_F(ArithmeticOptimizerTest,
       OptimizeMaxOrMinOfMonotonicElementWiseDoNotChangeFetchNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output sqrt = ops::Sqrt(s.WithOpName("sqrt"), x);
  Output reduce_max = ops::Max(s.WithOpName("reduce_max"), sqrt, {0});
  Output final_out = ops::Identity(s.WithOpName("final_out"), reduce_max);

  GrapplerItem item;
  item.fetch = {"sqrt", "final_out"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(tensors_expected.size(), 2);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);

  // Should be a NoOp since we are not allowed to change the output of fetch
  // nodes.
  VerifyGraphsMatch(item.graph, output, __LINE__);
}

TEST_F(ArithmeticOptimizerTest,
       OptimizeMaxOrMinOfMonotonicElementWiseDoNotChangeFetchNodeReduction) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {2, 3}, {1, 2});
  Output reshape = ops::Reshape(s.WithOpName("reshape"), x, {-1});
  Output y = ops::Neg(s.WithOpName("y"), reshape);
  Output z = ops::Max(s.WithOpName("z"), y, {0});

  GrapplerItem item;
  item.fetch = {"z"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);

  // Should be a NoOp since we are not allowed to change the output of fetch
  // nodes.
  VerifyGraphsMatch(item.graph, output, __LINE__);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<int>(tensors[0], tensors_expected[0]);
  test::ExpectTensorEqual<int>(tensors[0], Tensor(-2));
}

TEST_F(ArithmeticOptimizerTest,
       OptimizeMaxOrMinOfMonotonicElementWiseNonIncreasing) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output neg = ops::Neg(s.WithOpName("neg"), x);
  Output reduce_max = ops::Max(s.WithOpName("reduce_max"), neg, {0});
  Output final_out = ops::Identity(s.WithOpName("final_out"), reduce_max);

  GrapplerItem item;
  item.fetch = {"final_out"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  // Check if the inputs are switched
  int required_node_count = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "neg") {
      EXPECT_EQ(node.op(), "Neg");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "reduce_max");
      ++required_node_count;
    } else if (node.name() == "reduce_max") {
      EXPECT_EQ(node.op(), "Min");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "x");
      ++required_node_count;
    }
  }
  EXPECT_EQ(2, required_node_count);
}

TEST_F(ArithmeticOptimizerTest,
       OptimizeMaxOrMinOfMonotonicElementWiseNonIncreasingDoNotChangeMaxPool) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), 1.5f, {3, 3, 3, 1});
  Output neg = ops::Neg(s.WithOpName("neg"), x);
  Output max_pool = ops::MaxPool(s.WithOpName("max_pool"), neg, {1, 2, 2, 1},
                                 {1, 2, 2, 1}, "VALID");

  GrapplerItem item;
  item.fetch = {"max_pool"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);

  // Should be a NoOp
  VerifyGraphsMatch(item.graph, output, __LINE__);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, OptimizeMaxOrMinOfMonotonicElementWiseMaxPool) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), 1.5f, {3, 3, 3, 1});
  Output sqrt = ops::Sqrt(s.WithOpName("sqrt"), x);
  Output max_pool = ops::MaxPool(s.WithOpName("max_pool"), sqrt, {1, 2, 2, 1},
                                 {1, 2, 2, 1}, "VALID");
  Output final_out = ops::Identity(s.WithOpName("final_out"), max_pool);

  GrapplerItem item;
  item.fetch = {"final_out"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);

  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
  EXPECT_EQ(output.node_size(), item.graph.node_size());
  // Check if the inputs are switched
  int required_node_count = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "sqrt") {
      EXPECT_EQ(node.op(), "Sqrt");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "max_pool");
      ++required_node_count;
    } else if (node.name() == "max_pool") {
      EXPECT_EQ(node.op(), "MaxPool");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "x");
      ++required_node_count;
    }
  }
  EXPECT_EQ(required_node_count, 2);
}

TEST_F(ArithmeticOptimizerTest, UnaryOpsComposition) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output sqrt = ops::Sqrt(s.WithOpName("sqrt"), x);
  Output log = ops::Log(s.WithOpName("log"), sqrt);
  Output relu = ops::Relu(s.WithOpName("relu"), log);
  Output final_out = ops::Identity(s.WithOpName("final_out"), relu);

  GrapplerItem item;
  item.fetch = {"final_out"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Place all nodes on CPU.
  for (int i = 0; i < item.graph.node_size(); ++i) {
    item.graph.mutable_node(i)->set_device("/device:CPU:0");
  }

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyUnaryOpsComposition(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(output.node_size(), 3);

  // Check that Sqrt/Log/Relu were replaced with a single op.
  int required_node_count = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "final_out") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "relu/unary_ops_composition");
      ++required_node_count;
    } else if (node.name() == "relu/unary_ops_composition") {
      EXPECT_EQ(node.op(), "_UnaryOpsComposition");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "x");

      auto op_names = node.attr().at("op_names").list().s();
      ASSERT_EQ(op_names.size(), 3);
      EXPECT_EQ(op_names[0], "Sqrt");
      EXPECT_EQ(op_names[1], "Log");
      EXPECT_EQ(op_names[2], "Relu");
      ++required_node_count;
    }
  }
  EXPECT_EQ(required_node_count, 2);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorNear<float>(tensors[0], tensors_expected[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveStackStridedSliceSameAxis) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto a_in =
      ops::Const(s.WithOpName("a_in"), {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  auto b_in =
      ops::Const(s.WithOpName("b_in"), {-1.0f, -2.0f, -3.0f, -4.0f}, {2, 2});
  auto c_in =
      ops::Const(s.WithOpName("c_in"), {5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
  auto a = ops::PlaceholderWithDefault(s.WithOpName("a"), a_in,
                                       PartialTensorShape({-1, -1}));
  auto b = ops::PlaceholderWithDefault(s.WithOpName("b"), b_in,
                                       PartialTensorShape({-1, -1}));
  auto c = ops::PlaceholderWithDefault(s.WithOpName("c"), c_in,
                                       PartialTensorShape({-1, -1}));
  // stacked = tf.stack((a, b, c), axis=1).
  // stacked.shape == [2, 3, 2] (a, b, c are stacked along new axis 1)
  auto stacked =
      ops::Stack(s.WithOpName("stacked"), {a.output, b.output, c.output},
                 ops::Stack::Axis(1));
  auto expanded_a = ops::ExpandDims(s.WithOpName("expanded_a"), a, {1});
  auto expanded_b = ops::ExpandDims(s.WithOpName("expanded_b"), b, {1});
  auto expanded_c = ops::ExpandDims(s.WithOpName("expanded_c"), c, {1});
  auto begin_a = ops::Const(s.WithOpName("begin_a"), {0, 0, 0}, {3});
  auto end_a = ops::Const(s.WithOpName("end_a"), {0, 1, 0}, {3});
  auto begin_b = ops::Const(s.WithOpName("begin_b"), {0, 1, 0}, {3});
  auto end_b = ops::Const(s.WithOpName("end_b"), {0, 2, 0}, {3});
  auto begin_c = ops::Const(s.WithOpName("begin_c"), {0, 2, 0}, {3});
  auto end_c = ops::Const(s.WithOpName("end_c"), {0, 3, 0}, {3});
  auto end_c_1to = ops::Const(s.WithOpName("begin_c_2to"), {0, 0, 0}, {3});
  auto strides = ops::Const(s.WithOpName("strides"), {1, 1, 1}, {3});

  // stacked[:, 0]
  using SS = ops::StridedSlice;
  auto pa_slice = ops::Identity(
      s.WithOpName("pa_slice_out"),
      SS(s.WithOpName("pa_slice"), stacked, begin_a, end_a, strides,
         SS::BeginMask(0b0101)  // 5
             .EllipsisMask(0)
             .EndMask(0b0101)  // 5
             .NewAxisMask(0)
             .ShrinkAxisMask(0b0010)));  // 2

  // stacked[:, 1]
  auto pb_slice = ops::Identity(
      s.WithOpName("pb_slice_out"),
      SS(s.WithOpName("pb_slice"), stacked, begin_b, end_b, strides,
         SS::BeginMask(0b0101)  // 5
             .EllipsisMask(0)
             .EndMask(0b0101)  // 5
             .NewAxisMask(0)
             .ShrinkAxisMask(0b0010)));  // 2

  // stacked[:, 2]
  auto pc_slice = ops::Identity(
      s.WithOpName("pc_slice_out"),
      SS(s.WithOpName("pc_slice"), stacked, begin_c, end_c, strides,
         SS::BeginMask(0b0101)  // 5
             .EllipsisMask(0)
             .EndMask(0b0101)  // 5
             .NewAxisMask(0)
             .ShrinkAxisMask(0b0010)));  // 2

  // stacked[:, 0:1, :]
  auto pa_slice_01 = ops::Identity(
      s.WithOpName("pa_slice_01_out"),
      SS(s.WithOpName("pa_slice_01"), stacked, begin_a, end_a, strides,
         SS::BeginMask(0b0101)  // 5
             .EllipsisMask(0)
             .EndMask(0b0101)  // 5
             .NewAxisMask(0)
             .ShrinkAxisMask(0)));

  // stacked[:, :1, :]
  auto pa_slice_to1 = ops::Identity(
      s.WithOpName("pa_slice_to1_out"),
      SS(s.WithOpName("pa_slice_to1"), stacked, begin_a, end_a, strides,
         SS::BeginMask(0b0111)  // 7
             .EllipsisMask(0)
             .EndMask(0b0101)  // 5
             .NewAxisMask(0)
             .ShrinkAxisMask(0)));

  // stacked[:, 1:2, :]
  auto pb_slice_12 = ops::Identity(
      s.WithOpName("pb_slice_12_out"),
      SS(s.WithOpName("pb_slice_12"), stacked, begin_b, end_b, strides,
         SS::BeginMask(0b0101)  // 5
             .EllipsisMask(0)
             .EndMask(0b0101)  // 5
             .NewAxisMask(0)
             .ShrinkAxisMask(0)));

  // stacked[:, 2:, :].
  auto pc_slice_2to = ops::Identity(
      s.WithOpName("pc_slice_2to_out"),
      SS(s.WithOpName("pc_slice_2to"), stacked, begin_c, end_c_1to, strides,
         SS::BeginMask(0b0101)  // 5
             .EllipsisMask(0)
             .EndMask(0b0111)  // 7
             .NewAxisMask(0)
             .ShrinkAxisMask(0)));

  GrapplerItem item;
  item.fetch = {"a",
                "b",
                "c",
                "pa_slice_out",
                "pb_slice_out",
                "pc_slice_out",
                "expanded_a",
                "expanded_b",
                "expanded_c",
                "pa_slice_01_out",
                "pa_slice_to1_out",
                "pb_slice_12_out",
                "pc_slice_2to_out"};
  enum FetchItem {
    fA,
    fB,
    fC,
    fASliceOut,
    fBSliceOut,
    fCSliceOut,
    fExpandedA,
    fExpandedB,
    fExpandedC,
    fASlice01Out,
    fASliceTo1Out,
    fBSlice12Out,
    fCSlice2ToOut,
  };
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);

  // stacked[:, 0, :] == a.
  test::ExpectTensorEqual<float>(tensors_expected[fASliceOut],
                                 tensors_expected[fA]);
  // stacked[:, 1, :] == b.
  test::ExpectTensorEqual<float>(tensors_expected[fBSliceOut],
                                 tensors_expected[fB]);
  // stacked[:, 2, :] == c.
  test::ExpectTensorEqual<float>(tensors_expected[fCSliceOut],
                                 tensors_expected[fC]);

  // stacked[:, 0:1, :] == expand_dims(a, 1).
  test::ExpectTensorEqual<float>(tensors_expected[fASlice01Out],
                                 tensors_expected[fExpandedA]);

  // stacked[:, :1, :] == expand_dims(a, 1).
  test::ExpectTensorEqual<float>(tensors_expected[fASliceTo1Out],
                                 tensors_expected[fExpandedA]);

  // stacked[:, 1:2, :] == expand_dims(b, 1).
  test::ExpectTensorEqual<float>(tensors_expected[fBSlice12Out],
                                 tensors_expected[fExpandedB]);
  // stacked[:, 2:, :] == expand_dims(c, 1).
  test::ExpectTensorEqual<float>(tensors_expected[fCSlice2ToOut],
                                 tensors_expected[fExpandedC]);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveStackStridedSliceSameAxis(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  for (const auto& node : output.node()) {
    if (node.name() == "pa_slice_out") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "a");
    } else if (node.name() == "pb_slice_out") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "b");
    } else if (node.name() == "pc_slice_out") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "c");
    } else if (str_util::EndsWith(node.name(), "_out")) {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(
          absl::StrCat(node.input(0), "_out"),
          absl::StrCat("ArithmeticOptimizer/RemoveStackStridedSliceSameAxis_",
                       node.name()));
    }
  }

  auto tensors = EvaluateNodes(output, item.fetch);

  // stacked[:, 0, :] == a.
  test::ExpectTensorEqual<float>(tensors[fASliceOut], tensors_expected[fA]);

  // stacked[:, 1, :] == b.
  test::ExpectTensorEqual<float>(tensors[fBSliceOut], tensors_expected[fB]);
  // stacked[:, 2, :] == c.
  test::ExpectTensorEqual<float>(tensors[fCSliceOut], tensors_expected[fC]);

  // stacked[:, 0:1, :] == expand_dims(a, 1).
  test::ExpectTensorEqual<float>(tensors[fASlice01Out],
                                 tensors_expected[fExpandedA]);

  // stacked[:, :1, :] == expand_dims(a, 1).
  test::ExpectTensorEqual<float>(tensors[fASliceTo1Out],
                                 tensors_expected[fExpandedA]);

  // stacked[:, 1:2, :] == expand_dims(b, 1).
  test::ExpectTensorEqual<float>(tensors[fBSlice12Out],
                                 tensors_expected[fExpandedB]);
  // stacked[:, 2:, :] == expand_dims(c, 1).
  test::ExpectTensorEqual<float>(tensors[fCSlice2ToOut],
                                 tensors_expected[fExpandedC]);
}

TEST_F(ArithmeticOptimizerTest, SimplifyAggregationBFloat16) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output cast = ops::Cast(s.WithOpName("cast"), x, DT_BFLOAT16);
  Output add = ops::AddN(s.WithOpName("add"), {cast, cast});
  Output id = ops::Identity(s.WithOpName("id"), add);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"id"};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlySimplifyAggregation(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  // Extra node created for multiplier.
  EXPECT_EQ(output.node_size(), 5);

  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
  test::ExpectTensorEqual<bfloat16>(tensors[0], tensors_expected[0]);
}

}  // namespace grappler
}  // namespace tensorflow
