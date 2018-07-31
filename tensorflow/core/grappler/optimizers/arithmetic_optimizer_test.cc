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
    output->Clear();
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  // Run ArithmeticOptimizer twice to make sure the rewrite is idempotent.
  void OptimizeTwice(ArithmeticOptimizer* optimizer, GrapplerItem* item,
                     GraphDef* output) {
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
  }

  // Run ArithmeticOptimizer twice to make sure the rewrite is idempotent.
  // Optionally run a constant folding pass before pruning.
  void OptimizeTwiceAndPrune(ArithmeticOptimizer* optimizer, GrapplerItem* item,
                             GraphDef* output, bool const_folding = false) {
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));

    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));

    if (const_folding) {
      item->graph.Swap(output);
      output->Clear();
      TF_EXPECT_OK(ConstantFolding(/*cpu_device=*/nullptr)
                       .Optimize(nullptr, *item, output));
    }

    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  // TODO(ezhulenev): Make private. After migration to stages each test
  // should explicitly enable required optimization for tests isolation
  void DisableAllStages(ArithmeticOptimizer* optimizer) {
    ArithmeticOptimizer::ArithmeticOptimizerOptions options;
    options.dedup_computations = false;
    options.combine_add_to_addn = false;
    options.convert_sqrt_div_to_rsqrt_mul = false;
    options.convert_pow = false;
    options.convert_log1p = false;
    options.optimize_max_or_min_of_monotonic = false;
    options.fold_conjugate_into_transpose = false;
    options.fold_multiply_into_conv = false;
    options.fold_transpose_into_matmul = false;
    options.hoist_common_factor_out_of_aggregation = false;
    options.hoist_cwise_unary_chains = false;
    options.minimize_broadcasts = false;
    options.remove_identity_transpose = false;
    options.remove_involution = false;
    options.remove_idempotent = false;
    options.remove_redundant_bitcast = false;
    options.remove_redundant_cast = false;
    options.remove_redundant_reshape = false;
    options.remove_negation = false;
    options.remove_logical_not = false;
    options.reorder_cast_and_transpose = false;
    options.replace_mul_with_square = false;
    options.simplify_aggregation = false;
    options.unary_ops_composition = false;
    optimizer->options_ = options;
  }

  void DisableAddToAddNCombining(ArithmeticOptimizer* optimizer) {
    optimizer->options_.combine_add_to_addn = false;
  }

  void EnableOnlyAddToAddNCombining(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.combine_add_to_addn = true;
  }

  void EnableOnlyFoldConjugateIntoTranspose(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.fold_conjugate_into_transpose = true;
  }

  void EnableOnlyFoldMultipleIntoConv(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.fold_multiply_into_conv = true;
  }

  void EnableOnlyFoldTransposeIntoMatMul(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.fold_transpose_into_matmul = true;
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

  void EnableOnlyRemoveInvolution(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_involution = true;
  }

  void EnableOnlyRemoveRedundantBitcast(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_bitcast = true;
  }

  void EnableOnlyRemoveRedundantCast(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_cast = true;
  }

  void EnableOnlyRemoveRedundantReshape(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_reshape = true;
  }

  void EnableOnlyRemoveNegation(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_negation = true;
  }

  void EnableOnlyReorderCastAndTranspose(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.reorder_cast_and_transpose = true;
  }

  void EnableOnlyReplaceMulWithSquare(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.replace_mul_with_square = true;
  }

  void EnableOnlyHoistCWiseUnaryChains(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.hoist_cwise_unary_chains = true;
  }

  void EnableOnlySqrtDivToRsqrtMul(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_sqrt_div_to_rsqrt_mul = true;
  }

  void EnableOnlyConvertPow(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_pow = true;
  }

  void EnableOnlyRemoveIdempotent(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_idempotent = true;
  }

  void EnableOnlyRemoveLogicalNot(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_logical_not = true;
  }

  void EnableOnlySimplifyAggregation(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.simplify_aggregation = true;
  }

  void EnableOnlyLog1p(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_log1p = true;
  }

  void EnableOnlyOptimizeMaxOrMinOfMonotonic(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.optimize_max_or_min_of_monotonic = true;
  }

  void EnableOnlyUnaryOpsComposition(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.unary_ops_composition = true;
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

TEST_F(ArithmeticOptimizerTest, ReplaceMulWithSquare) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output d = ops::Const(s.WithOpName("d"), {3.0f, 4.0f}, {1, 2});
  Output mul = ops::Mul(s.WithControlDependencies(d).WithOpName("mul"), c, c);
  Output id = ops::Identity(s.WithOpName("id"), mul);

  GrapplerItem item;
  item.fetch = {"id"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyReplaceMulWithSquare(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(4, output.node_size());

  NodeMap node_map(&output);
  const string p = "ArithmeticOptimizer/ReplaceMulWithSquare";
  const NodeDef* square_node = node_map.GetNode(strings::StrCat(p, "_", "mul"));

  ASSERT_NE(square_node, nullptr);
  EXPECT_EQ("Square", square_node->op());
  EXPECT_EQ("c", square_node->input(0));
  EXPECT_EQ("^d", square_node->input(1));

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveInvolution_AdjacentNodes) {
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveInvolution(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  // Negation and Reciprocal nodes cancelled each other.
  EXPECT_EQ(2, output.node_size());
  EXPECT_EQ("id", output.node(1).name());
  EXPECT_EQ("c", output.node(1).input(0));

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveInvolution_AroundValuePreservingChain) {
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveInvolution(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  // Check that Reciprocal nodes were removed from the graph.
  EXPECT_EQ(3, output.node_size());

  // And const directly flows into squeeze.
  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "squeeze") {
      EXPECT_EQ("c", node.input(0));
      found++;
    } else if (node.name() == "id2") {
      EXPECT_EQ("squeeze", node.input(0));
      found++;
    }
  }
  EXPECT_EQ(2, found);

  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveInvolution_SkipControlDependencies) {
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveInvolution(&optimizer);
  OptimizeTwice(&optimizer, &item, &output);  // do not prune in this test

  // The optimizer should be a noop.
  VerifyGraphsMatch(item.graph, output, __LINE__);

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
  item.fetch = {"id"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(5, output.node_size());

  const string optimized_const_name = AggregationConstName("add");
  const string optimized_mul_name = AggregationMulName("add");

  const NodeDef* new_const = node_map.GetNode(optimized_const_name);
  ASSERT_NE(new_const, nullptr);
  EXPECT_EQ("^x", new_const->input(0));
  EXPECT_EQ(std::string("\0\0\0@", 4),
            new_const->attr().at("value").tensor().tensor_content());

  const NodeDef* new_mul = node_map.GetNode(optimized_mul_name);
  ASSERT_NE(new_mul, nullptr);
  EXPECT_EQ(optimized_const_name, new_mul->input(0));
  EXPECT_EQ("x", new_mul->input(1));

  const NodeDef* new_id = node_map.GetNode("id");
  ASSERT_NE(new_id, nullptr);
  EXPECT_EQ(optimized_mul_name, new_id->input(0));

  auto tensors = EvaluateNodes(output, item.fetch);
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

  const string optimized_const_name = AggregationConstName("add");
  const string optimized_mul_name = AggregationMulName("add");

  const NodeDef* new_const = node_map.GetNode(optimized_const_name);
  ASSERT_NE(new_const, nullptr);
  EXPECT_EQ("^x", new_const->input(0));
  EXPECT_EQ(std::string("\0\0\0@", 4),
            new_const->attr().at("value").tensor().tensor_content());

  const NodeDef* new_mul = node_map.GetNode(optimized_mul_name);
  ASSERT_NE(new_mul, nullptr);
  EXPECT_EQ(optimized_const_name, new_mul->input(0));
  EXPECT_EQ("x", new_mul->input(1));
  EXPECT_EQ("^y", new_mul->input(2));

  const NodeDef* new_id = node_map.GetNode("id");
  ASSERT_NE(new_id, nullptr);
  EXPECT_EQ(optimized_mul_name, new_id->input(0));

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
  EXPECT_EQ(AggregationConstName("Add"), add_4_node->input(0));
  EXPECT_EQ(AggregationConstName("Add_1"), add_4_node->input(1));

  const NodeDef* add_5_node = node_map.GetNode(HoistAddName("Add_5"));
  ASSERT_NE(add_5_node, nullptr);
  EXPECT_EQ("Add", add_5_node->op());
  EXPECT_EQ(2, add_5_node->input_size());
  EXPECT_EQ(AggregationConstName("Add"), add_5_node->input(0));
  EXPECT_EQ(AggregationConstName("Add_1"), add_5_node->input(1));

  const NodeDef* add_const_node = node_map.GetNode(AggregationConstName("Add"));
  ASSERT_NE(add_const_node, nullptr);
  EXPECT_EQ("Const", add_const_node->op());
  EXPECT_EQ(1, add_const_node->input_size());
  EXPECT_EQ("^Placeholder", add_const_node->input(0));

  const NodeDef* add_1_const_node =
      node_map.GetNode(AggregationConstName("Add_1"));
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

        auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
        EXPECT_EQ(1, tensors_expected.size());

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
        auto tensors = EvaluateNodes(output, item.fetch);
        EXPECT_EQ(1, tensors.size());
        if (use_ints) {
          test::ExpectTensorEqual<int32>(tensors_expected[0], tensors[0]);
        } else {
          test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(7, output.node_size());

  const string p = "ArithmeticOptimizer/FoldConjugateIntoTranspose";
  const string optimized_name = strings::StrCat(p, "_", "trans");

  const NodeDef* trans_fused_node = node_map.GetNode(optimized_name);
  ASSERT_NE(trans_fused_node, nullptr);
  EXPECT_EQ("ConjugateTranspose", trans_fused_node->op());
  EXPECT_EQ("z", trans_fused_node->input(0));
  EXPECT_EQ("perm", trans_fused_node->input(1));

  auto tensors = EvaluateNodes(output, item.fetch);
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
  item.fetch = {"conjugate_trans"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(7, output.node_size());

  const string p = "ArithmeticOptimizer/FoldConjugateIntoTranspose";
  const string optimized_name = strings::StrCat(p, "_", "conjugate_trans");

  const NodeDef* conjugate_trans_fused_node = node_map.GetNode(optimized_name);
  ASSERT_NE(conjugate_trans_fused_node, nullptr);
  EXPECT_EQ("Transpose", conjugate_trans_fused_node->op());
  EXPECT_EQ("z", conjugate_trans_fused_node->input(0));
  EXPECT_EQ("perm", conjugate_trans_fused_node->input(1));

  auto tensors = EvaluateNodes(output, item.fetch);
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
  item.fetch = {"conj"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);
  NodeMap node_map(&output);

  EXPECT_EQ(7, output.node_size());

  const string p = "ArithmeticOptimizer/FoldConjugateIntoTranspose";
  const string optimized_name = strings::StrCat(p, "_", "conj");

  const NodeDef* conj_fused_node = node_map.GetNode(optimized_name);
  ASSERT_NE(conj_fused_node, nullptr);
  EXPECT_EQ("ConjugateTranspose", conj_fused_node->op());
  EXPECT_EQ("z", conj_fused_node->input(0));
  EXPECT_EQ("perm", conj_fused_node->input(1));

  auto tensors = EvaluateNodes(output, item.fetch);
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

    auto matmul_op = s.WithOpName("matmul");
    if (matmul_type == "MatMul") {
      Output matmul = ops::MatMul(matmul_op, trans_a, trans_b);
    } else if (matmul_type == "SparseMatMul") {
      Output matmul = ops::SparseMatMul(matmul_op, trans_a, trans_b);
    } else if (matmul_type == "BatchMatMul") {
      Output matmul = ops::BatchMatMul(matmul_op, trans_a, trans_b);
    }

    GrapplerItem item;
    item.fetch = {"matmul"};
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
    EXPECT_EQ(1, tensors_expected.size());

    ArithmeticOptimizer optimizer;
    EnableOnlyFoldTransposeIntoMatMul(&optimizer);
    GraphDef output;
    OptimizeTwice(&optimizer, &item, &output);
    NodeMap node_map(&output);

    EXPECT_EQ(7, output.node_size());

    const string p = "ArithmeticOptimizer/FoldTransposeIntoMatMul";
    const string optimized_name = strings::StrCat(p, "_", "matmul");

    const NodeDef* matmul_fused_node = node_map.GetNode(optimized_name);
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

    auto tensors = EvaluateNodes(output, item.fetch);
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
  item.fetch = {"matmul"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());

  ArithmeticOptimizer optimizer;
  GraphDef output;
  OptimizeTwice(&optimizer, &item, &output);

  NodeMap node_map(&output);
  ASSERT_EQ(11, output.node_size());

  const string p = "ArithmeticOptimizer/FoldTransposeIntoMatMul";
  const string optimized_name = strings::StrCat(p, "_", "matmul");

  const NodeDef* optimized_matmul = node_map.GetNode(optimized_name);
  ASSERT_NE(optimized_matmul, nullptr);
  EXPECT_EQ("a", optimized_matmul->input(0));
  EXPECT_EQ("b", optimized_matmul->input(1));
  EXPECT_TRUE(optimized_matmul->attr().at("adj_x").b());
  EXPECT_TRUE(optimized_matmul->attr().at("adj_y").b());

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<complex64>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshape_IdentityReshape) {
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
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(0, CountOpNodes(output, "Reshape"));
  auto tensors = EvaluateNodes(output, item.fetch, {{"Placeholder", x_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest,
       RemoveRedundantReshape_IdentityReshapeBetweenSymbolicShapes) {
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  // Assume valid feed shape in aggressive mode.
  ArithmeticOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(0, CountOpNodes(output, "Reshape"));
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshape_NotAssumeValidFeeds) {
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  // The reshape is preserved because the shape of the placeholder can be
  // different from the shape of the actual feed.
  EXPECT_EQ(1, CountOpNodes(output, "Reshape"));

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest,
       RemoveRedundantReshape_AssumeValidFeedsInAggressiveMode) {
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(0, CountOpNodes(output, "Reshape"));
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshape_NotIdentityReshape) {
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
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(1, CountOpNodes(output, "Reshape"));
  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest,
       RemoveRedundantReshape_NotIdentityReshapeTooManyUnknownDimSizes) {
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

  EXPECT_EQ(1, CountOpNodes(output, "Reshape"));
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantReshape_CombineReshapes) {
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
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantReshape(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

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
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveIdentityTranspose(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  NodeMap node_map(&output);
  const NodeDef* outputs_node = node_map.GetNode("outputs");
  EXPECT_EQ(2, outputs_node->input_size());
  EXPECT_EQ(outputs_node->input(0), "outputs_const");
  EXPECT_EQ(outputs_node->input(1), "^Placeholder");

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("inputs", node.input(0));
      EXPECT_EQ("^perm2", node.input(1));
    }
    if (node.name() == "id1") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("id", node.input(0));
    }
  }
  EXPECT_EQ(nodes_after_optimization,
            std::set<string>({"id", "id1", "inputs_shape", "inputs", "perm2"}));
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
  ArithmeticOptimizer optimizer;
  EnableOnlyFoldMultipleIntoConv(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output);

  NodeMap node_map(&output);

  // `conv` is now a folded convolution with scaled weights.
  const NodeDef* folded_conv = node_map.GetNode(conv.node()->name());
  ASSERT_NE(folded_conv, nullptr);

  const NodeDef* folded_conv_weights = node_map.GetNode(folded_conv->input(1));
  ASSERT_NE(folded_conv_weights, nullptr);
  EXPECT_EQ("Mul", folded_conv_weights->op());

  // Its input should be a transpose of `inputs`.
  const NodeDef* transpose = node_map.GetNode(NodeName(folded_conv->input(0)));
  ASSERT_NE(transpose, nullptr);
  EXPECT_EQ("inputs", transpose->input(0));
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
  ArithmeticOptimizer optimizer;  // all optimization stages are on
  OptimizeTwiceAndPrune(&optimizer, &item, &output, /*const_folding=*/true);

  NodeMap node_map(&output);

  // Expected names for reordered cast and transpose.
  const string p = "ArithmeticOptimizer/ReorderCastAndTranspose_";
  const string optimized_cast_name = strings::StrCat(p, "float_Cast");
  const string optimized_transpose_name = strings::StrCat(p, "uint8_Transpose");

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
  ArithmeticOptimizer optimizer;
  EnableOnlyFoldMultipleIntoConv(&optimizer);
  OptimizeTwiceAndPrune(&optimizer, &item, &output, /*const_folding=*/true);

  NodeMap node_map(&output);

  using strings::StrCat;
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantBitcast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Bitcasts combined into a single op and inputs redirected to updated Bitcast
  EXPECT_EQ(3, output.node_size());
  EXPECT_EQ(1, CountOpNodes(output, "Bitcast"));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "bc2"));

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int8>(tensors_expected[0], tensors[0]);
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantBitcast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Bitcasts removed and inputs redirected to outputs
  EXPECT_EQ(2, output.node_size());
  EXPECT_EQ(0, CountOpNodes(output, "Bitcast"));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "outputs"));

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int8>(tensors_expected[0], tensors[0]);
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyRemoveRedundantCast(&optimizer);

  OptimizeAndPrune(&optimizer, &item, &output);
  NodeMap node_map(&output);

  // Cast removed and inputs redirected to outputs
  EXPECT_EQ(2, output.node_size());
  EXPECT_EQ(0, CountOpNodes(output, "Cast"));
  EXPECT_TRUE(IsNodesDirectlyConnected(node_map, "inputs", "outputs"));

  auto tensors = EvaluateNodes(output, item.fetch, item.feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int8>(tensors_expected[0], tensors[0]);
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

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto z_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}, {"x", x_t}, {"y", y_t}, {"z", z_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {{"input", x_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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

  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32}));
  auto c_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32, 32}));
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32}));
  auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32}));
  auto z_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({32, 32, 32}));
  std::vector<std::pair<string, Tensor>> feed = {
      {"a", a_t}, {"b", b_t}, {"c", c_t}, {"x", x_t}, {"y", y_t}, {"z", z_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ArithmeticOptimizerTest, AddOpsRewrite_MinimizeBCastWithSymbolicShapes) {
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
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<double>(tensors_expected[0], tensors[0], 1e-6);
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

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  std::vector<std::pair<string, Tensor>> feed = {{"x", x_t}, {"y", y_t}};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, feed);
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlySqrtDivToRsqrtMul(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());

  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  EXPECT_EQ(item.graph.node_size(), output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "output") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("sqrt_y", node.input(1));
    } else if (node.name() == "sqrt_y") {
      EXPECT_EQ("Rsqrt", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
    }
  }
}

TEST_F(ArithmeticOptimizerTest, ConvertPow) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  auto x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  auto y2 = ops::Const(s.WithOpName("y2"), {2.0f, 2.0f}, {1, 2});
  auto y1 = ops::Const(s.WithOpName("y1"), {1.0f, 1.0f}, {1, 2});
  auto yPoint5 = ops::Const(s.WithOpName("y.5"), {0.5f, 0.5f}, {1, 2});
  auto y0 = ops::Const(s.WithOpName("y0"), {0.0f, 0.0f}, {1, 2});
  auto y_Point5 = ops::Const(s.WithOpName("y_.5"), {-0.5f, -0.5f}, {1, 2});
  auto y_1 = ops::Const(s.WithOpName("y_1"), {-1.0f, -1.0f}, {1, 2});
  auto y = ops::Const(s.WithOpName("y"), {3.0f, 4.0f}, {1, 2});
  Output out2 = ops::Pow(s.WithOpName("out2"), x, y2);
  Output out1 = ops::Pow(s.WithOpName("out1"), x, y1);
  Output outPoint5 = ops::Pow(s.WithOpName("out.5"), x, yPoint5);
  Output out0 = ops::Pow(s.WithOpName("out0"), x, y0);
  Output out_Point5 = ops::Pow(s.WithOpName("out_.5"), x, y_Point5);
  Output out_1 = ops::Pow(s.WithOpName("out_1"), x, y_1);
  Output out = ops::Pow(s.WithOpName("out"), x, y);

  GrapplerItem item;
  item.fetch = {"out2", "out1", "out.5", "out0", "out_.5", "out_1", "out"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(7, tensors_expected.size());

  GraphDef got;
  ArithmeticOptimizer optimizer;
  EnableOnlyConvertPow(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &got);
  auto tensors = EvaluateNodes(got, item.fetch);
  EXPECT_EQ(7, tensors.size());

  GraphDef want;
  AddNode("x", "Const", {}, {}, &want);
  AddNode("y2", "Const", {}, {}, &want);
  AddNode("y1", "Const", {}, {}, &want);
  AddNode("y.5", "Const", {}, {}, &want);
  AddNode("y0", "Const", {}, {}, &want);
  AddNode("y_.5", "Const", {}, {}, &want);
  AddNode("y_1", "Const", {}, {}, &want);
  AddNode("y", "Const", {}, {}, &want);
  AddNode("out2", "Square", {"x", AsControlDependency("y2")}, {}, &want);
  AddNode("out1", "Identity", {"x", AsControlDependency("y1")}, {}, &want);
  AddNode("out.5", "Sqrt", {"x", AsControlDependency("y.5")}, {}, &want);
  AddNode("out0", "Const",
          {AsControlDependency("x"), AsControlDependency("y0")}, {}, &want);
  AddNode("out_.5", "Rsqrt", {"x", AsControlDependency("y_.5")}, {}, &want);
  AddNode("out_1", "Reciprocal", {"x", AsControlDependency("y_1")}, {}, &want);
  AddNode("out", "Pow", {"x", "y"}, {}, &want);

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
  EXPECT_EQ(2, tensors_expected.size());

  GraphDef got;
  ArithmeticOptimizer optimizer;
  EnableOnlyLog1p(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &got);
  auto tensors = EvaluateNodes(got, item.fetch);
  EXPECT_EQ(2, tensors.size());

  GraphDef want;
  AddNode("x1", "Const", {}, {}, &want);
  AddNode("x2", "Const", {}, {}, &want);
  AddNode("x3", "Const", {}, {}, &want);
  AddNode("a23", "Add", {"x2", "x3"}, {}, &want);
  AddNode("out1", "Log1p",
          {"x2", AsControlDependency("x1"), AsControlDependency("x3")}, {},
          &want);
  AddNode("out2", "Log", {"a23"}, {}, &want);

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
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<double>(tensors_expected[0], tensors[0], 1e-6);
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
  EXPECT_EQ(1, tensors_expected.size());

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

  auto tensors = EvaluateNodes(output, item.fetch, feed);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
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
      EXPECT_EQ(6, node.input_size());
      EXPECT_EQ("sin_a", node.input(0));
      EXPECT_EQ("b", node.input(1));
      EXPECT_EQ("c", node.input(2));
      EXPECT_EQ("axis", node.input(3));
      EXPECT_EQ("^ctrl1", node.input(4));
      EXPECT_EQ("^ctrl2", node.input(5));
      found++;
    }
    if (node.name() == "exp_a") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("concat", node.input(0));
      EXPECT_EQ("^ctrl1", node.input(1));
      found++;
    }
    if (node.name() == "id") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("exp_a", node.input(0));
      found++;
    }

    if (node.name() == "concat2") {
      EXPECT_EQ(7, node.input_size());
      EXPECT_EQ("sin_a", node.input(0));
      EXPECT_EQ("b", node.input(1));
      EXPECT_EQ("c", node.input(2));
      EXPECT_EQ("axis", node.input(3));
      EXPECT_EQ("^ctrl1", node.input(4));
      EXPECT_EQ("^ctrl2", node.input(5));
      EXPECT_EQ("^ctrl3", node.input(6));
      found++;
    }
    if (node.name() == "exp_a2") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("concat2", node.input(0));
      EXPECT_EQ("^ctrl1", node.input(1));
      found++;
    }
    if (node.name() == "cos_exp_a2") {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("exp_a2", node.input(0));
      EXPECT_EQ("^ctrl1", node.input(1));
      found++;
    }
    if (node.name() == "id2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("cos_exp_a2", node.input(0));
      found++;
    }
  }
  EXPECT_EQ(7, found);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
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
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("axis", node.input(0));
      EXPECT_EQ("ArithmeticOptimizer/_sin_a_split1", node.input(1));
      found++;
    }
    if (node.name() == "ArithmeticOptimizer/_sin_a_split1") {
      EXPECT_EQ("Sin", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^ctrl1", node.input(1));
      found++;
    }
    if (node.name() == "id_a") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("split1", node.input(0));
      found++;
    }
    if (node.name() == "exp_b") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("split1:1", node.input(0));
      found++;
    }
    if (node.name() == "id_b") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("exp_b", node.input(0));
      found++;
    }
    if (node.name() == "ArithmeticOptimizer/_exp_a2_split2") {
      EXPECT_EQ("Exp", node.op());
      EXPECT_EQ(4, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("^ctrl1", node.input(1));
      EXPECT_EQ("^ctrl2", node.input(2));
      EXPECT_EQ("^ctrl3", node.input(3));
      found++;
    }
    if (node.name() == "ArithmeticOptimizer/_cos_exp_a2_split2") {
      EXPECT_EQ("Cos", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ArithmeticOptimizer/_exp_a2_split2", node.input(0));
      found++;
    }
    if (node.name() == "split2") {
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("ArithmeticOptimizer/_cos_exp_a2_split2", node.input(0));
      EXPECT_EQ("size_splits2", node.input(1));
      EXPECT_EQ("axis", node.input(2));
      found++;
    }
    if (node.name() == "id_a2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("split2", node.input(0));
      found++;
    }
    if (node.name() == "id_b2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("split2:1", node.input(0));
      found++;
    }
  }
  EXPECT_EQ(10, found);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
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
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("sn1", node.input(0));
      found++;
    } else if (node.name() == "out2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("id1", node.input(0));
      found++;
    } else if (node.name() == "sn1") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("a", node.input(0));
      found++;
    }
  }
  EXPECT_EQ(3, found);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
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
      EXPECT_EQ("eq", node.input(0));
      ++found;
    }
    if (node.name() == "id_not_neq") {
      EXPECT_EQ("neq", node.input(0));
      ++found;
    }
    if (node.name() == "id_not_lt") {
      EXPECT_EQ("lt", node.input(0));
      ++found;
    }
    if (node.name() == "id_not_le") {
      EXPECT_EQ("le", node.input(0));
      ++found;
    }
    if (node.name() == "id_not_gt") {
      EXPECT_EQ("gt", node.input(0));
      ++found;
    }
    if (node.name() == "id_not_ge") {
      EXPECT_EQ("ge", node.input(0));
      ++found;
    }

    if (node.name() == "eq") {
      EXPECT_EQ("NotEqual", node.op());
      ++found;
    }
    if (node.name() == "neq") {
      EXPECT_EQ("Equal", node.op());
      ++found;
    }
    if (node.name() == "lt") {
      EXPECT_EQ("GreaterEqual", node.op());
      ++found;
    }
    if (node.name() == "le") {
      EXPECT_EQ("Greater", node.op());
      ++found;
    }
    if (node.name() == "gt") {
      EXPECT_EQ("LessEqual", node.op());
      ++found;
    }
    if (node.name() == "ge") {
      EXPECT_EQ("Less", node.op());
      ++found;
    }
  }
  EXPECT_EQ(12, found);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(tensors.size(), tensors_expected.size());
  EXPECT_EQ(tensors.size(), item.fetch.size());
  for (int i = 0; i < item.fetch.size(); ++i) {
    test::ExpectTensorEqual<bool>(tensors_expected[i], tensors[i]);
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyOptimizeMaxOrMinOfMonotonic(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);
  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());

  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  EXPECT_EQ(item.graph.node_size(), output.node_size());
  // Check if the inputs are switched
  int required_node_count = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "sqrt") {
      EXPECT_EQ("Sqrt", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("reduce_max", node.input(0));
      ++required_node_count;
    } else if (node.name() == "reduce_max") {
      EXPECT_EQ("Max", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      ++required_node_count;
    }
  }
  EXPECT_EQ(2, required_node_count);
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
  EXPECT_EQ(1, tensors_expected.size());

  GraphDef output;
  ArithmeticOptimizer optimizer;
  EnableOnlyUnaryOpsComposition(&optimizer);
  OptimizeAndPrune(&optimizer, &item, &output);

  EXPECT_EQ(3, output.node_size());

  // Check that Sqrt/Log/Relu were replaced with a single op.
  int required_node_count = 0;
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "final_out") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("relu/unary_ops_composition", node.input(0));
      ++required_node_count;
    } else if (node.name() == "relu/unary_ops_composition") {
      EXPECT_EQ("_UnaryOpsComposition", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("x", node.input(0));

      auto op_names = node.attr().at("op_names").list().s();
      EXPECT_EQ(3, op_names.size());
      EXPECT_EQ("Sqrt", op_names[0]);
      EXPECT_EQ("Log", op_names[1]);
      EXPECT_EQ("Relu", op_names[2]);
      ++required_node_count;
    }
  }
  EXPECT_EQ(2, required_node_count);

  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

}  // namespace grappler
}  // namespace tensorflow
