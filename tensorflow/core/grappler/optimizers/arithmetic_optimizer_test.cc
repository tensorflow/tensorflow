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
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
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

// Optimized name of outer Mul node by HoistCommonFactorOutOfAggregation
string HoistMulName(const string& name) {
  return AddPrefixToNodeName(name, kHoistFactorOptimizerMul, "");
}

// Optimized name of outer Div node by HoistCommonFactorOutOfAggregation
string HoistDivName(const string& name) {
  return AddPrefixToNodeName(name, kHoistFactorOptimizerDiv, "");
}

// Optimized name of inner Add node by HoistCommonFactorOutOfAggregation
string HoistAddName(const string& name) {
  return AddPrefixToNodeName(name, kHoistFactorOptimizerAdd, "");
}

string OptimizedName(const string& name) {
  return AddPrefixToNodeName(name, kArithmeticOptimizer);
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

class ArithmeticOptimizerTest : public GrapplerTest {
 protected:
  // Optimize a graph using ArithmeticOptimizer and prune all the nodes that no
  // longer have any output consumers.
  void OptimizeAndPrune(ArithmeticOptimizer* optimizer, GrapplerItem* item,
                        GraphDef* output) {
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  // Run ArithmeticOptimizer twice to make sure the rewrite is idempotent.
  void OptimizeTwice(ArithmeticOptimizer* optimizer, GrapplerItem* item,
                     GraphDef* output) {
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
  }

  // TODO(ezhulenev): Make private. After migration to stages each test
  // should explicitly enable required optimization for tests isolation
  void DisableAllStages(ArithmeticOptimizer* optimizer) {
    ArithmeticOptimizer::ArithmeticOptimizerOptions options;
    options.enable_try_simplify_and_replace = false;
    options.combine_add_to_addn = false;
    options.hoist_common_factor_out_of_aggregation = false;
    options.minimize_broadcasts = false;
    options.remove_identity_transpose = false;
    options.remove_redundant_bitcast = false;
    options.remove_redundant_cast = false;
    options.remove_negation = false;
    optimizer->options_ = options;
  }

  void DisableAddToAddNCombining(ArithmeticOptimizer* optimizer) {
    optimizer->options_.combine_add_to_addn = false;
  }

  void EnableOnlyAddToAddNCombining(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.combine_add_to_addn = true;
  }

  void EnableOnlyHoistCommonFactor(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.hoist_common_factor_out_of_aggregation = true;
  }

  void EnableOnlyMinimizeBroadcasts(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.minimize_broadcasts = true;
  }

  void EnableOnlyRemoveIdentityTranspose(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_identity_transpose = true;
  }

  void EnableOnlyRemoveRedundantBitcast(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_bitcast = true;
  }

  void EnableOnlyRemoveRedundantCast(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_cast = true;
  }

  void EnableOnlyRemoveNegation(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_negation = true;
  }
};

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
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);
  EXPECT_EQ(2, output.node_size());
  const NodeDef* new_c1 = node_map.GetNode("c1");
  ASSERT_NE(new_c1, nullptr);

  const NodeDef* new_div = node_map.GetNode("div");
  ASSERT_NE(new_div, nullptr);
  EXPECT_EQ(2, new_div->input_size());
  EXPECT_EQ("c1", new_div->input(0));
  EXPECT_EQ("c1", new_div->input(1));

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<double>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;

  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(5, output.node_size());
  const NodeDef* new_div = node_map.GetNode("div");
  ASSERT_NE(new_div, nullptr);
  EXPECT_EQ(4, new_div->input_size());
  EXPECT_EQ("check1", new_div->input(0));
  EXPECT_EQ("check1", new_div->input(1));
  EXPECT_EQ("^assert1", new_div->input(2));
  EXPECT_EQ("^assert1", new_div->input(3));

  auto tensors = EvaluateNodes(output, item.fetch, {{"Placeholder", bool_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<double>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(4, output.node_size());
  const NodeDef* new_c1 = node_map.GetNode("c1");
  ASSERT_NE(new_c1, nullptr);
  const NodeDef* new_c2 = node_map.GetNode("c2");
  ASSERT_NE(new_c2, nullptr);
  const NodeDef* new_mul1 = node_map.GetNode("mul1");
  ASSERT_NE(new_mul1, nullptr);
  EXPECT_EQ(2, new_mul1->input_size());
  EXPECT_EQ("c1", new_mul1->input(0));
  EXPECT_EQ("c2", new_mul1->input(1));
  const NodeDef* new_div1 = node_map.GetNode("div1");
  ASSERT_NE(new_div1, nullptr);
  EXPECT_EQ(2, new_div1->input_size());
  EXPECT_EQ("mul1", new_div1->input(0));
  EXPECT_EQ("mul1", new_div1->input(1));

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, MulToSquare) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output d = ops::Const(s.WithOpName("d"), {3.0f, 4.0f}, {1, 2});
  Output mul = ops::Mul(s.WithControlDependencies(d).WithOpName("mul"), c, c);
  Output id = ops::Identity(s.WithOpName("id"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  std::vector<string> fetch = {"id"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(5, output.node_size());
  EXPECT_EQ("id", output.node(3).name());
  EXPECT_EQ(OptimizedName("mul_square"), output.node(3).input(0));
  EXPECT_EQ("Square", output.node(4).op());
  EXPECT_EQ(OptimizedName("mul_square"), output.node(4).name());
  EXPECT_EQ(2, output.node(4).input_size());
  EXPECT_EQ("c", output.node(4).input(0));
  EXPECT_EQ("^d", output.node(4).input(1));

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, SimplifyInvolutionsReal) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output neg1 = ops::Neg(s.WithOpName("neg1"), c);
  Output neg2 = ops::Neg(s.WithOpName("neg2"), neg1);
  Output recip1 = ops::Reciprocal(s.WithOpName("recip1"), neg2);
  Output recip2 = ops::Reciprocal(s.WithOpName("recip2"), recip1);
  Output id = ops::Identity(s.WithOpName("id"), recip2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  std::vector<string> fetch = {"id"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(6, output.node_size());
  EXPECT_EQ("c", output.node(1).input(0));
  EXPECT_EQ("c", output.node(3).input(0));
  EXPECT_EQ("c", output.node(5).input(0));

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, SimplifyInvolutionsWithChain) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output recip1 = ops::Reciprocal(s.WithOpName("recip1"), c);
  Output id1 = ops::Identity(s.WithOpName("id1"), recip1);
  Output squeeze = ops::Squeeze(s.WithOpName("squeeze"), id1);
  Output recip2 = ops::Reciprocal(s.WithOpName("recip2"), squeeze);
  Output id2 = ops::Identity(s.WithOpName("id2"), recip2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  std::vector<string> fetch = {"id2"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(6, output.node_size());
  EXPECT_EQ("squeeze", output.node(5).input(0));
  EXPECT_EQ("c", output.node(2).input(0));

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, SimplifyInvolutionsWithControlChain) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output recip1 = ops::Reciprocal(s.WithOpName("recip1"), c);
  Output id1 = ops::Identity(s.WithOpName("id1"), recip1);
  Output squeeze = ops::Squeeze(s.WithOpName("squeeze"), id1);
  Output recip2 = ops::Reciprocal(
      s.WithOpName("recip2").WithControlDependencies(squeeze), c);
  Output id2 = ops::Identity(s.WithOpName("id2"), recip2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  std::vector<string> fetch = {"id2"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // The optimizer should be a noop.
  EXPECT_EQ(item.graph.node_size(), output.node_size());
  for (int i = 0; i < item.graph.node_size(); ++i) {
    const NodeDef& original = item.graph.node(i);
    const NodeDef& optimized = output.node(i);
    EXPECT_EQ(original.name(), optimized.name());
    EXPECT_EQ(original.op(), optimized.op());
    EXPECT_EQ(original.input_size(), optimized.input_size());
    for (int j = 0; j < original.input_size(); ++j) {
      EXPECT_EQ(original.input(j), optimized.input(j));
    }
  }

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, TrivialSumsSimple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output add = ops::Add(s.WithOpName("add"), x, x);
  Output id = ops::Identity(s.WithOpName("id"), add);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  std::vector<string> fetch = {"id"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(5, output.node_size());

  const NodeDef* new_const = node_map.GetNode(OptimizedName("add_const"));
  ASSERT_NE(new_const, nullptr);
  EXPECT_EQ("^x", new_const->input(0));
  EXPECT_EQ(std::string("\0\0\0@", 4),
            new_const->attr().at("value").tensor().tensor_content());

  const NodeDef* new_mul = node_map.GetNode(OptimizedName("add_mul"));
  ASSERT_NE(new_mul, nullptr);
  EXPECT_EQ(OptimizedName("add_const"), new_mul->input(0));
  EXPECT_EQ("x", new_mul->input(1));

  const NodeDef* new_id = node_map.GetNode("id");
  ASSERT_NE(new_id, nullptr);
  EXPECT_EQ(OptimizedName("add_mul"), new_id->input(0));

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(6, output.node_size());

  const NodeDef* new_const = node_map.GetNode(OptimizedName("add_const"));
  ASSERT_NE(new_const, nullptr);
  EXPECT_EQ("^x", new_const->input(0));
  EXPECT_EQ(std::string("\0\0\0@", 4),
            new_const->attr().at("value").tensor().tensor_content());

  const NodeDef* new_mul = node_map.GetNode(OptimizedName("add_mul"));
  ASSERT_NE(new_mul, nullptr);
  EXPECT_EQ(OptimizedName("add_const"), new_mul->input(0));
  EXPECT_EQ("x", new_mul->input(1));
  EXPECT_EQ("^y", new_mul->input(2));

  const NodeDef* new_id = node_map.GetNode("id");
  ASSERT_NE(new_id, nullptr);
  EXPECT_EQ(OptimizedName("add_mul"), new_id->input(0));

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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

  EXPECT_EQ(17, output.node_size());

  const NodeDef* id_node = node_map.GetNode("id");
  ASSERT_NE(id_node, nullptr);
  EXPECT_EQ(1, id_node->input_size());
  EXPECT_EQ(HoistMulName("Add_6"), id_node->input(0));

  const NodeDef* mul_node = node_map.GetNode(HoistMulName("Add_6"));
  ASSERT_NE(mul_node, nullptr);
  EXPECT_EQ(2, mul_node->input_size());
  EXPECT_EQ("Placeholder", mul_node->input(0));
  EXPECT_EQ(HoistAddName("Add_6"), mul_node->input(1));

  const NodeDef* add_6_node = node_map.GetNode(HoistAddName("Add_6"));
  ASSERT_NE(add_6_node, nullptr);
  EXPECT_EQ(2, add_6_node->input_size());
  EXPECT_EQ(HoistAddName("Add_4"), add_6_node->input(0));
  EXPECT_EQ(HoistAddName("Add_5"), add_6_node->input(1));

  const NodeDef* add_4_node = node_map.GetNode(HoistAddName("Add_4"));
  ASSERT_NE(add_4_node, nullptr);
  EXPECT_EQ("Add", add_4_node->op());
  EXPECT_EQ(2, add_4_node->input_size());
  EXPECT_EQ(OptimizedName("Add_const"), add_4_node->input(0));
  EXPECT_EQ(OptimizedName("Add_1_const"), add_4_node->input(1));

  const NodeDef* add_5_node = node_map.GetNode(HoistAddName("Add_5"));
  ASSERT_NE(add_5_node, nullptr);
  EXPECT_EQ("Add", add_5_node->op());
  EXPECT_EQ(2, add_5_node->input_size());
  EXPECT_EQ(OptimizedName("Add_const"), add_5_node->input(0));
  EXPECT_EQ(OptimizedName("Add_1_const"), add_5_node->input(1));

  const NodeDef* add_const_node = node_map.GetNode(OptimizedName("Add_const"));
  ASSERT_NE(add_const_node, nullptr);
  EXPECT_EQ("Const", add_const_node->op());
  EXPECT_EQ(1, add_const_node->input_size());
  EXPECT_EQ("^Placeholder", add_const_node->input(0));

  const NodeDef* add_1_const_node =
      node_map.GetNode(OptimizedName("Add_1_const"));
  ASSERT_NE(add_1_const_node, nullptr);
  EXPECT_EQ("Const", add_1_const_node->op());
  EXPECT_EQ(1, add_1_const_node->input_size());
  EXPECT_EQ("^Placeholder", add_1_const_node->input(0));
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
      EXPECT_EQ(1, tensors_expected.size());
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
        EXPECT_EQ(9, output.node_size());

        const NodeDef* new_add_node = node_map.GetNode(HoistAddName("add"));
        ASSERT_NE(new_add_node, nullptr) << "Hoisted Add node not found";
        EXPECT_EQ("y1", new_add_node->input(0));
        EXPECT_EQ("y2", new_add_node->input(1));

        const NodeDef* new_mul_node = node_map.GetNode(HoistMulName("add"));
        ASSERT_NE(new_mul_node, nullptr) << "Hoisted Mul node not found";
        EXPECT_EQ("x", new_mul_node->input(0));
        EXPECT_EQ(new_add_node->name(), new_mul_node->input(1));

        const NodeDef* id_node = node_map.GetNode("id");
        ASSERT_NE(id_node, nullptr) << "Id node not found";
        EXPECT_EQ("id", id_node->name());
        EXPECT_EQ(HoistMulName("add"), id_node->input(0));
      }
      auto tensors = EvaluateNodes(output, item.fetch);
      EXPECT_EQ(1, tensors.size());
      test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
          EXPECT_EQ(9, output.node_size());

          const NodeDef* new_add_node = node_map.GetNode(HoistAddName("add"));
          ASSERT_TRUE(new_add_node != nullptr) << "Hoisted Add node not found";
          EXPECT_EQ("y1", new_add_node->input(0));
          EXPECT_EQ("y2", new_add_node->input(1));

          const NodeDef* new_div_node = node_map.GetNode(HoistDivName("add"));
          ASSERT_TRUE(new_div_node != nullptr) << "Hoisted Div node not found";
          EXPECT_EQ(new_add_node->name(), new_div_node->input(0));
          EXPECT_EQ("x", new_div_node->input(1));

          const NodeDef* id_node = node_map.GetNode("id");
          ASSERT_TRUE(id_node != nullptr) << "Id node not found";
          EXPECT_EQ("id", id_node->name());
          EXPECT_EQ(HoistDivName("add"), id_node->input(0));
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
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  std::vector<string> fetch = {"trans"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());
  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(7, output.node_size());

  const NodeDef* trans_fused_node =
      node_map.GetNode(OptimizedName("trans_fused"));
  ASSERT_NE(trans_fused_node, nullptr);
  EXPECT_EQ("ConjugateTranspose", trans_fused_node->op());
  EXPECT_EQ("z", trans_fused_node->input(0));
  EXPECT_EQ("perm", trans_fused_node->input(1));

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<complex64>(tensors_expected[0], tensors[0]);
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
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  std::vector<string> fetch = {"conjugate_trans"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(7, output.node_size());

  const NodeDef* conjugate_trans_fused_node =
      node_map.GetNode(OptimizedName("conjugate_trans_fused"));
  EXPECT_EQ("Transpose", conjugate_trans_fused_node->op());
  EXPECT_EQ("z", conjugate_trans_fused_node->input(0));
  EXPECT_EQ("perm", conjugate_trans_fused_node->input(1));
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<complex64>(tensors_expected[0], tensors[0]);
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
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  std::vector<string> fetch = {"conj"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(7, output.node_size());

  const NodeDef* conj_fused_node =
      node_map.GetNode(OptimizedName("conj_fused"));
  EXPECT_EQ("ConjugateTranspose", conj_fused_node->op());
  EXPECT_EQ("z", conj_fused_node->input(0));
  EXPECT_EQ("perm", conj_fused_node->input(1));
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<complex64>(tensors_expected[0], tensors[0]);
}

TEST_F(ArithmeticOptimizerTest, FoldTransposeIntoMatMul) {
  for (const string matmul_type : {"MatMul", "SparseMatMul", "BatchMatMul"}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output a = ops::Const(s.WithOpName("a"), {1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    Output b = ops::Const(s.WithOpName("b"), {5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});
    Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
    Output trans_a = ops::Transpose(s.WithOpName("trans_a"), a, perm);
    Output trans_b = ops::Transpose(s.WithOpName("trans_b"), b, perm);
    if (matmul_type == "MatMul") {
      Output matmul = ops::MatMul(s.WithOpName("matmul"), trans_a, trans_b);
    } else if (matmul_type == "SparseMatMul") {
      Output matmul =
          ops::SparseMatMul(s.WithOpName("matmul"), trans_a, trans_b);
    } else if (matmul_type == "BatchMatMul") {
      Output matmul =
          ops::BatchMatMul(s.WithOpName("matmul"), trans_a, trans_b);
    }
    GrapplerItem item;
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    std::vector<string> fetch = {"matmul"};
    auto tensors_expected = EvaluateNodes(item.graph, fetch);
    EXPECT_EQ(1, tensors_expected.size());

    ArithmeticOptimizer optimizer;
    GraphDef output;
    OptimizeTwice(&optimizer, &item, &output);
    NodeMap node_map(&output);

    EXPECT_EQ(7, output.node_size());

    const NodeDef* matmul_fused_node =
        node_map.GetNode(OptimizedName("matmul_fused"));
    ASSERT_NE(matmul_fused_node, nullptr);
    EXPECT_EQ("a", matmul_fused_node->input(0));
    EXPECT_EQ("b", matmul_fused_node->input(1));
    if (matmul_type == "BatchMatMul") {
      EXPECT_TRUE(matmul_fused_node->attr().at("adj_x").b());
      EXPECT_TRUE(matmul_fused_node->attr().at("adj_y").b());
    } else {
      EXPECT_TRUE(matmul_fused_node->attr().at("transpose_a").b());
      EXPECT_TRUE(matmul_fused_node->attr().at("transpose_b").b());
    }
    auto tensors = EvaluateNodes(output, fetch);
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  std::vector<string> fetch = {"matmul"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(11, output.node_size());
  EXPECT_EQ(OptimizedName("matmul_fused"), output.node(10).name());
  EXPECT_EQ("a", output.node(10).input(0));
  EXPECT_EQ("b", output.node(10).input(1));
  EXPECT_TRUE(output.node(10).attr().at("adj_x").b());
  EXPECT_TRUE(output.node(10).attr().at("adj_y").b());
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<complex64>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, IdentityReshape) {
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
  EXPECT_EQ(1, tensors_expected.size());
  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(0, CountOpNodes(output, "Reshape"));
  auto tensors = EvaluateNodes(output, item.fetch, {{"Placeholder", x_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, NotIdentityReshape) {
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
  EXPECT_EQ(1, tensors_expected.size());
  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(1, CountOpNodes(output, "Reshape"));
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, NotIdentityReshapeTooManyUnknownDimSizes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs =
      ops::Placeholder(s, DT_FLOAT, ops::Placeholder::Shape({4, 3}));
  Output reshape = ops::Reshape(s, inputs, ops::Const(s, {-1, -1}, {2}));
  Output outputs = ops::Identity(s.WithOpName("outputs"), reshape);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(1, CountOpNodes(output, "Reshape"));
}

TEST_F(ArithmeticOptimizerTest, CombineReshapes) {
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
  EXPECT_EQ(1, tensors_expected.size());
  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(1, CountOpNodes(output, "Reshape"));
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int8>(tensors_expected[0], tensors[0]);
}

TEST_F(ArithmeticOptimizerTest, ReorderTransposeCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/gpu:0");
  Output nhwc_uint8 =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({8, 28, 28, 3}));
  Output nhwc_fp32 = ops::Cast(s, nhwc_uint8, DT_FLOAT);
  Output nchw_fp32 =
      ops::Transpose(s, nhwc_fp32, ops::Const(s, {0, 3, 1, 2}, {4}));
  Output outputs = ops::Identity(s.WithOpName("outputs"), nchw_fp32);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  const NodeDef* transpose_node = nullptr;
  for (const NodeDef& node : output.node()) {
    if (node.op() == "Transpose") {
      EXPECT_EQ(transpose_node, nullptr);
      EXPECT_EQ(DT_UINT8, node.attr().at("T").type());
      transpose_node = &node;
    }
  }
  EXPECT_NE(transpose_node, nullptr);

  for (const NodeDef& node : output.node()) {
    if (node.op() == "Cast") {
      EXPECT_EQ(NodeName(node.input(0)), transpose_node->name());
    }
  }
}

TEST_F(ArithmeticOptimizerTest, NoReorderTransposeCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/gpu:0");
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

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  int num_transposes = 0;
  for (const NodeDef& node : output.node()) {
    if (node.op() == "Transpose") {
      EXPECT_EQ(DT_UINT8, node.attr().at("T").type());
      EXPECT_EQ(node.input(0), "Cast");
      ++num_transposes;
    }
  }
  EXPECT_EQ(1, num_transposes);
}

TEST_F(ArithmeticOptimizerTest, RemoveIdentityTransposes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs_shape =
      ops::Const(s.WithOpName("inputs_shape"), {8, 3, 28, 28}, {4});
  Output inputs =
      ops::RandomUniform(s.WithOpName("inputs"), inputs_shape, DT_FLOAT);
  Output perm1 = ops::Const(s.WithOpName("perm1"), {0, 2, 3, 1}, {4});
  Output perm2 = ops::Const(s.WithOpName("perm2"), {0, 3, 1, 2}, {4});
  Output perm3 = ops::Const(s.WithOpName("perm2"), {0, 1, 2, 3}, {4});
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

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  for (const NodeDef& node : output.node()) {
    if (node.op() == "Concat") {
      EXPECT_EQ(node.input(0), "Split");
      EXPECT_EQ(node.input(1), "Split:1");
      EXPECT_EQ(node.input(2), "Split:2");
    }
  }
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

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  NodeMap node_map(&output);
  const NodeDef* outputs_node = node_map.GetNode("outputs");
  EXPECT_EQ(2, outputs_node->input_size());
  EXPECT_EQ(outputs_node->input(0), "outputs_const");
  EXPECT_EQ(outputs_node->input(1), "^Placeholder");
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

  EXPECT_EQ(6, output.node_size());
}

TEST_F(ArithmeticOptimizerTest, FoldMulToTransposeConv) {
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

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  NodeMap node_map(&output);
  // `conv` is now a folded convolution with scaled weights.
  const NodeDef* folded_conv = node_map.GetNode(conv.node()->name());
  CHECK_EQ(node_map.GetNode(NodeName(folded_conv->input(1)))->op(), "Mul");
  // Its input should be a transpose of `inputs`.
  const NodeDef* transpose = node_map.GetNode(NodeName(folded_conv->input(0)));
  CHECK_EQ(NodeName(transpose->input(0)), inputs.node()->name());
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
  CHECK_EQ(inputs.node()->name(), NodeName(folded_conv->input(0)));
  CHECK_EQ(node_map.GetNode(NodeName(folded_conv->input(1)))->op(), "Mul");
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
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/gpu:0");
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
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(
      ConstantFolding(/*cpu_device=*/nullptr).Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  NodeMap node_map(&output);
  const NodeDef* inputs_node = CHECK_NOTNULL(node_map.GetNode("Placeholder"));
  const NodeDef* transpose_node =
      CHECK_NOTNULL(node_map.GetNode(OptimizedName("Transpose_uint8")));
  const NodeDef* cast_node =
      CHECK_NOTNULL(node_map.GetNode(OptimizedName("Cast_float")));
  const NodeDef* weights_node =
      CHECK_NOTNULL(node_map.GetNode(OptimizedName("weights_scaled_Conv2D")));
  const NodeDef* conv_node = CHECK_NOTNULL(node_map.GetNode("Conv2D"));

  EXPECT_EQ(output.node_size(), 7);
  EXPECT_EQ(transpose_node->input(0), inputs_node->name());
  EXPECT_EQ(cast_node->input(0), transpose_node->name());
  EXPECT_EQ(conv_node->input(0), cast_node->name());
  EXPECT_EQ(conv_node->input(1), weights_node->name());
}

TEST_F(ArithmeticOptimizerTest, OptimizeMultipleMulTransposeConv) {
  // This unit test exercises optimization of folding mul into conv for
  // multiple nodes in the graph.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice("/gpu:0");

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
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(
      ConstantFolding(/*cpu_device=*/nullptr).Optimize(nullptr, item, &output));

  item.graph.Swap(&output);
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  NodeMap node_map(&output);
  const NodeDef* weights_node =
      CHECK_NOTNULL(node_map.GetNode(OptimizedName("weights_scaled_Conv2D")));
  const NodeDef* conv_node = CHECK_NOTNULL(node_map.GetNode("Conv2D"));

  const NodeDef* weights_node_1 =
      CHECK_NOTNULL(node_map.GetNode(OptimizedName("weights_scaled_Conv2D_1")));
  const NodeDef* conv_node_1 = CHECK_NOTNULL(node_map.GetNode("Conv2D_1"));
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

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantBitcast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Bitcasts combined into a single op and inputs redirected to updated Bitcast
  EXPECT_EQ(3, output.node_size());
  EXPECT_EQ(1, CountOpNodes(output, "Bitcast"));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "bc2"));
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

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantBitcast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Bitcasts removed and inputs redirected to outputs
  EXPECT_EQ(2, output.node_size());
  EXPECT_EQ(0, CountOpNodes(output, "Bitcast"));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "outputs"));
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

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantCast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Cast removed and inputs redirected to outputs
  EXPECT_EQ(2, output.node_size());
  EXPECT_EQ(0, CountOpNodes(output, "Cast"));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "outputs"));
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewrite_AddOpsOfIdenticalShape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  tensorflow::Scope sx = s.NewSubScope("x");
  tensorflow::Scope sy = s.NewSubScope("y");

  auto a = ops::Variable(s.WithOpName("a"), {2, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {2, 2}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {2, 2}, DT_FLOAT);
  auto add_ab = ops::Add(sx.WithOpName("Add_ab"), a, b);
  auto add_abc = ops::Add(sy.WithOpName("Add_abc"), add_ab, c);

  auto outputs = ops::Identity(s.WithOpName("outputs"), add_abc);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

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
  EXPECT_EQ(5, output.node_size());

  NodeMap node_map(&output);

  // check add tree was replaced with AddN
  const NodeDef* collapsed_add =
      node_map.GetNode("y/ArithmeticOptimizer/AddOpsRewrite_Add_abc");
  ASSERT_NE(collapsed_add, nullptr);

  EXPECT_EQ("AddN", collapsed_add->op());
  EXPECT_EQ(3, collapsed_add->input_size());
  EXPECT_EQ("a", collapsed_add->input(0));
  EXPECT_EQ("b", collapsed_add->input(1));
  EXPECT_EQ("c", collapsed_add->input(2));

  // check output was re-wired to new node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);

  EXPECT_EQ(collapsed_add->name(), updated_outputs->input(0));
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewrite_MultiplePasses) {
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
  EXPECT_EQ(10, output.node_size());

  NodeMap node_map(&output);

  // check left Add subtree replaced with AddN
  const NodeDef* collapsed_left =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_abc");
  ASSERT_NE(collapsed_left, nullptr);

  EXPECT_EQ("AddN", collapsed_left->op());
  EXPECT_EQ(3, collapsed_left->input_size());
  EXPECT_EQ("a", collapsed_left->input(0));
  EXPECT_EQ("b", collapsed_left->input(1));
  EXPECT_EQ("c", collapsed_left->input(2));

  // check right Add subtree replaced with AddN
  const NodeDef* collapsed_right =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_xyz");
  ASSERT_NE(collapsed_right, nullptr);

  EXPECT_EQ("AddN", collapsed_right->op());
  EXPECT_EQ(3, collapsed_right->input_size());
  EXPECT_EQ("x", collapsed_right->input(0));
  EXPECT_EQ("y", collapsed_right->input(1));
  EXPECT_EQ("z", collapsed_right->input(2));

  // check that Mul inputs re-wired to new Nodes
  const NodeDef* updated_mul = node_map.GetNode("Mul");
  ASSERT_NE(updated_mul, nullptr);

  EXPECT_EQ("Mul", updated_mul->op());
  EXPECT_EQ(2, updated_mul->input_size());
  EXPECT_EQ(collapsed_left->name(), updated_mul->input(0));
  EXPECT_EQ(collapsed_right->name(), updated_mul->input(1));
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewrite_AddInputMultipleTimes) {
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
  EXPECT_EQ(5, output.node_size());

  NodeMap node_map(&output);

  // check Add tree replaced with AddN
  const NodeDef* collapsed_add =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_all");
  ASSERT_NE(collapsed_add, nullptr);

  EXPECT_EQ("AddN", collapsed_add->op());
  EXPECT_EQ(4, collapsed_add->input_size());
  EXPECT_EQ("a", collapsed_add->input(0));
  EXPECT_EQ("b", collapsed_add->input(1));
  EXPECT_EQ("b", collapsed_add->input(2));
  EXPECT_EQ("c", collapsed_add->input(3));
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewrite_AddOpsOfSymbolicallyEqualShape) {
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
  EXPECT_EQ(6, output.node_size());

  NodeMap node_map(&output);

  // check add tree was replaced with AddN
  const NodeDef* collapsed_add =
      node_map.GetNode("ArithmeticOptimizer/AddOpsRewrite_Add_abc");
  ASSERT_NE(collapsed_add, nullptr);
  EXPECT_EQ("AddN", collapsed_add->op());
  EXPECT_EQ(3, collapsed_add->input_size());
  EXPECT_EQ("a", collapsed_add->input(0));
  EXPECT_EQ("b", collapsed_add->input(1));
  EXPECT_EQ("c", collapsed_add->input(2));

  // check output was re-wired to new node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);
  EXPECT_EQ(collapsed_add->name(), updated_outputs->input(0));
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewrite_MinimizeBCast) {
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
  EXPECT_EQ(12, output.node_size());
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
  EXPECT_EQ("AddN", add_ax_node->op());
  EXPECT_EQ(2, add_ax_node->input_size());
  EXPECT_EQ("a", add_ax_node->input(0));
  EXPECT_EQ("x", add_ax_node->input(1));

  // Then add [b, y]
  const NodeDef* add_by_node = node_map.GetNode(inner_1_add_name);
  ASSERT_NE(add_by_node, nullptr);
  EXPECT_EQ("AddN", add_by_node->op());
  EXPECT_EQ(2, add_by_node->input_size());
  EXPECT_EQ("b", add_by_node->input(0));
  EXPECT_EQ("y", add_by_node->input(1));

  // Then add [c, z]
  const NodeDef* add_cz_node = node_map.GetNode(inner_2_add_name);
  ASSERT_NE(add_cz_node, nullptr);
  EXPECT_EQ("AddN", add_cz_node->op());
  EXPECT_EQ(2, add_cz_node->input_size());
  EXPECT_EQ("c", add_cz_node->input(0));
  EXPECT_EQ("z", add_cz_node->input(1));

  // Then add results together starting from smaller shapes [a, x] + [b, y]
  const NodeDef* outer_0_node = node_map.GetNode(outer_0_add_name);
  ASSERT_NE(outer_0_node, nullptr);
  EXPECT_EQ("Add", outer_0_node->op());
  EXPECT_EQ(2, outer_0_node->input_size());
  EXPECT_EQ(inner_0_add_name, outer_0_node->input(0));
  EXPECT_EQ(inner_1_add_name, outer_0_node->input(1));

  // And finally top level Add node
  const NodeDef* outer_node = node_map.GetNode(outer_add_name);
  ASSERT_NE(outer_node, nullptr);
  EXPECT_EQ("Add", outer_node->op());
  EXPECT_EQ(2, outer_node->input_size());
  EXPECT_EQ(outer_0_add_name, outer_node->input(0));
  EXPECT_EQ(inner_2_add_name, outer_node->input(1));

  // And outputs reading new top level Add node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);
  EXPECT_EQ(outer_add_name, updated_outputs->input(0));
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewrite_MinimizeBCastWithSymbolicShapes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  // We have a small input with one unknown dimension
  auto small = ops::Variable(s.WithOpName("small"), {-1, 1, 1}, DT_FLOAT);

  // And second input which is larger, but has the same unknown dimension
  // device spec prevents this node from rewriting
  auto d = "/job:do_not_rewrite_me";
  auto v = ops::Variable(s.WithOpName("v"), {1, 32, 32}, DT_FLOAT);
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
  EXPECT_EQ(9, output.node_size());
  NodeMap node_map(&output);

  // expected names of outer and inner nodes
  string outer_add_name = "ArithmeticOptimizer/AddOpsRewrite_Add_abc";
  string inner_add_name = "ArithmeticOptimizer/AddOpsRewrite_Leaf_0_Add_abc";

  // outer Add node
  const NodeDef* outer_add = node_map.GetNode(outer_add_name);
  ASSERT_NE(outer_add, nullptr);
  EXPECT_EQ("Add", outer_add->op());
  EXPECT_EQ(inner_add_name, outer_add->input(0));
  EXPECT_EQ("b", outer_add->input(1));

  // inner AddN node
  const NodeDef* inner_add = node_map.GetNode(inner_add_name);
  ASSERT_NE(inner_add, nullptr);
  EXPECT_EQ(2, inner_add->input_size());
  EXPECT_EQ("a", inner_add->input(0));
  EXPECT_EQ("c", inner_add->input(1));

  // check output was re-wired to new node
  const NodeDef* updated_outputs = node_map.GetNode("outputs");
  ASSERT_NE(updated_outputs, nullptr);
  EXPECT_EQ(outer_add_name, updated_outputs->input(0));
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
  auto add_all = ops::AddN(s.WithOpName("add_all"),
                           {add_x_y, add_negx_y, add_x_negy, add_negx_negy,
                            sub_x_y, sub_negx_y, sub_x_negy, sub_negx_negy});

  GrapplerItem item;
  item.fetch = {"add_all"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveNegation(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(item.graph.node_size(), output.node_size());
  int found = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "Add_negx_y") {
      ++found;
      EXPECT_EQ("Sub", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("x", node.input(1));
      EXPECT_EQ("^Neg_x", node.input(2));
    } else if (node.name() == "Add_x_negy") {
      ++found;
      EXPECT_EQ("Sub", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^Neg_y", node.input(2));
    } else if (node.name() == "Add_negx_negy") {
      ++found;
      EXPECT_EQ("Sub", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("Neg_y", node.input(0));
      EXPECT_EQ("x", node.input(1));
      EXPECT_EQ("^Neg_x", node.input(2));
    } else if (node.name() == "Sub_x_negy") {
      ++found;
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^Neg_y", node.input(2));
    } else if (node.name() == "Sub_negx_negy") {
      ++found;
      EXPECT_EQ("Sub", node.op());
      EXPECT_EQ(4, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("x", node.input(1));
      EXPECT_EQ("^Neg_y", node.input(2));
      EXPECT_EQ("^Neg_x", node.input(3));
    }
  }
  EXPECT_EQ(5, found);
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
  EXPECT_EQ("a", mul1_node->input(0));
  EXPECT_EQ("c", mul1_node->input(1));

  const NodeDef* mul2_node = node_map.GetNode("mul2");
  ASSERT_NE(mul2_node, nullptr);
  EXPECT_EQ("mul1", mul2_node->input(0));
  EXPECT_EQ("b", mul2_node->input(1));
}

TEST_F(ArithmeticOptimizerTest, MinimizeBroadcasts_FlattenTallGraph) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  auto a = ops::Variable(s.WithOpName("a"), {32}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {32, 32}, DT_FLOAT);
  auto c = ops::Variable(s.WithOpName("c"), {32}, DT_FLOAT);
  auto d = ops::Variable(s.WithOpName("d"), {32}, DT_FLOAT);
  auto e = ops::Variable(s.WithOpName("e"), {32}, DT_FLOAT);

  auto mul1 = ops::Mul(s.WithOpName("mul1"), a, b);
  auto mul2 = ops::Mul(s.WithOpName("mul2"), mul1, c);
  auto mul3 = ops::Mul(s.WithOpName("mul3"), mul2, d);
  auto mul4 = ops::Mul(s.WithOpName("mul4"), mul3, e);

  auto outputs = ops::Identity(s.WithOpName("outputs"), mul4);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

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
  EXPECT_EQ("a", mul1_node->input(0));
  EXPECT_EQ("c", mul1_node->input(1));

  const NodeDef* mul2_node = node_map.GetNode("mul2");
  ASSERT_NE(mul2_node, nullptr);
  EXPECT_EQ("d", mul2_node->input(0));
  EXPECT_EQ("e", mul2_node->input(1));

  const NodeDef* mul3_node = node_map.GetNode("mul3");
  ASSERT_NE(mul3_node, nullptr);
  EXPECT_EQ("mul1", mul3_node->input(0));
  EXPECT_EQ("mul2", mul3_node->input(1));

  const NodeDef* mul4_node = node_map.GetNode("mul4");
  ASSERT_NE(mul4_node, nullptr);
  EXPECT_EQ("mul3", mul4_node->input(0));
  EXPECT_EQ("b", mul4_node->input(1));
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
  EXPECT_EQ("a", mul1_node->input(0));
  EXPECT_EQ("b", mul1_node->input(1));

  const NodeDef* mul2_node = node_map.GetNode("mul1");
  ASSERT_NE(mul2_node, nullptr);
  EXPECT_EQ("mul2", mul2_node->input(0));
  EXPECT_EQ("c", mul2_node->input(1));

  const NodeDef* mul3_node = node_map.GetNode("mul3");
  ASSERT_NE(mul3_node, nullptr);
  EXPECT_EQ("D", mul3_node->input(0));
  EXPECT_EQ("mul1", mul3_node->input(1));
}

}  // namespace grappler
}  // namespace tensorflow
