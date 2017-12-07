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

class ArithmeticOptimizerTest : public ::testing::Test {};

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

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(2, output.node_size());
  const NodeDef& new_c1 = output.node(0);
  EXPECT_EQ("c1", new_c1.name());
  const NodeDef& new_div = output.node(1);
  EXPECT_EQ("div", new_div.name());
  EXPECT_EQ(2, new_div.input_size());
  EXPECT_EQ("c1", new_div.input(0));
  EXPECT_EQ("c1", new_div.input(1));
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

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(5, output.node_size());
  const NodeDef& new_div = output.node(3);
  EXPECT_EQ(4, new_div.input_size());
  EXPECT_EQ("check1", new_div.input(0));
  EXPECT_EQ("check1", new_div.input(1));
  EXPECT_EQ("^assert1", new_div.input(2));
  EXPECT_EQ("^assert1", new_div.input(3));
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
  item.fetch = {"div"};

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(4, output.node_size());
  const NodeDef& new_c1 = output.node(0);
  EXPECT_EQ("c1", new_c1.name());
  const NodeDef& new_c2 = output.node(1);
  EXPECT_EQ("c2", new_c2.name());
  const NodeDef& new_mul1 = output.node(2);
  EXPECT_EQ("mul1", new_mul1.name());
  EXPECT_EQ(2, new_mul1.input_size());
  EXPECT_EQ("c1", new_mul1.input(0));
  EXPECT_EQ("c2", new_mul1.input(1));
  const NodeDef& new_div1 = output.node(3);
  EXPECT_EQ("div1", new_div1.name());
  EXPECT_EQ(2, new_div1.input_size());
  EXPECT_EQ("mul1", new_div1.input(0));
  EXPECT_EQ("mul1", new_div1.input(1));
}

TEST_F(ArithmeticOptimizerTest, MulToSquare) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(s.WithOpName("c"), {1.0f, 2.0f}, {1, 2});
  Output d = ops::Const(s.WithOpName("d"), {3.0f, 4.0f}, {1, 2});
  Output mul = ops::Mul(s.WithControlDependencies(d).WithOpName("mul"), c, c);
  Output id = ops::Identity(s.WithOpName("id"), mul);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

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

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(6, output.node_size());
  EXPECT_EQ("c", output.node(1).input(0));
  EXPECT_EQ("c", output.node(3).input(0));
  EXPECT_EQ("c", output.node(5).input(0));
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
}

TEST_F(ArithmeticOptimizerTest, TrivialSumsSimple) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output add = ops::Add(s.WithOpName("add"), x, x);
  Output id = ops::Identity(s.WithOpName("id"), add);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(5, output.node_size());
  const NodeDef& new_const = output.node(3);
  EXPECT_EQ(OptimizedName("add_const"), new_const.name());
  EXPECT_EQ("^x", new_const.input(0));
  EXPECT_EQ(std::string("\0\0\0@", 4),
            new_const.attr().at("value").tensor().tensor_content());
  const NodeDef& new_mul = output.node(4);
  EXPECT_EQ(OptimizedName("add_mul"), new_mul.name());
  EXPECT_EQ(OptimizedName("add_const"), new_mul.input(0));
  EXPECT_EQ("x", new_mul.input(1));
  const NodeDef& new_id = output.node(2);
  EXPECT_EQ("id", new_id.name());
  EXPECT_EQ(OptimizedName("add_mul"), new_id.input(0));
}

TEST_F(ArithmeticOptimizerTest, TrivialSumsSimpleWithControlDep) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output y = ops::Const(s.WithOpName("y"), {1.0f, 2.0f}, {1, 2});
  Output x = ops::Const(s.WithOpName("x"), {3.0f, 4.0f}, {1, 2});
  Output add = ops::Add(s.WithOpName("add").WithControlDependencies(y), x, x);
  Output id = ops::Identity(s.WithOpName("id"), add);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(6, output.node_size());
  const NodeDef& new_const = output.node(4);
  EXPECT_EQ(OptimizedName("add_const"), new_const.name());
  EXPECT_EQ("^x", new_const.input(0));
  EXPECT_EQ(std::string("\0\0\0@", 4),
            new_const.attr().at("value").tensor().tensor_content());
  const NodeDef& new_mul = output.node(5);
  EXPECT_EQ(OptimizedName("add_mul"), new_mul.name());
  EXPECT_EQ(OptimizedName("add_const"), new_mul.input(0));
  EXPECT_EQ("x", new_mul.input(1));
  EXPECT_EQ("^y", new_mul.input(2));
  const NodeDef& new_id = output.node(3);
  EXPECT_EQ("id", new_id.name());
  EXPECT_EQ(OptimizedName("add_mul"), new_id.input(0));
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
  ArithmeticOptimizer optimizer(RewriterConfig::AGGRESSIVE);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(17, output.node_size());
  // The graph gets optimized to
  // Mul(p,
  //     Add(Add(Const(2), Const(2)),
  //         Add(Const(2), Const(2))))
  EXPECT_EQ(17, output.node_size());
  for (const auto& node : output.node()) {
    if ("id" == node.name()) {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ(OptimizedName("Add_6_hoist_mul"), node.input(0));
    } else if (OptimizedName("Add_6_hoist_mul") == node.name()) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("Placeholder", node.input(0));
      EXPECT_EQ(OptimizedName("Add_6_hoist_add"), node.input(1));
    } else if (OptimizedName("Add_6_hoist_add") == node.name()) {
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ(OptimizedName("Add_4_hoist_add"), node.input(0));
      EXPECT_EQ(OptimizedName("Add_5_hoist_add"), node.input(1));
      EXPECT_EQ("^Placeholder", node.input(2));
    } else if (OptimizedName("Add_4_hoist_add") == node.name()) {
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ(OptimizedName("Add_const"), node.input(0));
      EXPECT_EQ(OptimizedName("Add_1_const"), node.input(1));
      EXPECT_EQ("^Placeholder", node.input(2));
    } else if (OptimizedName("Add_5_hoist_add") == node.name()) {
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ(OptimizedName("Add_const"), node.input(0));
      EXPECT_EQ(OptimizedName("Add_1_const"), node.input(1));
      EXPECT_EQ("^Placeholder", node.input(2));
    } else if (OptimizedName("Add_const") == node.name()) {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^Placeholder", node.input(0));
    } else if (OptimizedName("Add_1_const") == node.name()) {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^Placeholder", node.input(0));
    }
  }
}

TEST_F(ArithmeticOptimizerTest, HoistFactor) {
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
      TF_CHECK_OK(s.ToGraphDef(&item.graph));
      ArithmeticOptimizer optimizer(RewriterConfig::AGGRESSIVE);
      GraphDef output;
      Status status = optimizer.Optimize(nullptr, item, &output);
      TF_EXPECT_OK(status);
      // Run the optimizer twice to make sure the rewrite is idempotent.
      item.graph.Swap(&output);
      status = optimizer.Optimize(nullptr, item, &output);
      TF_EXPECT_OK(status);

      if (use_addn && !matching_shapes) {
        VerifyGraphsMatch(item.graph, output, __LINE__);
      } else {
        EXPECT_EQ(9, output.node_size());
        const NodeDef& new_add = output.node(8);
        EXPECT_EQ(OptimizedName("add_hoist_add"), new_add.name());
        EXPECT_EQ("y1", new_add.input(0));
        EXPECT_EQ("y2", new_add.input(1));
        const NodeDef& new_mul = output.node(7);
        EXPECT_EQ(OptimizedName("add_hoist_mul"), new_mul.name());
        EXPECT_EQ("x", new_mul.input(0));
        EXPECT_EQ(OptimizedName("add_hoist_add"), new_mul.input(1));
        const NodeDef& new_id = output.node(6);
        EXPECT_EQ("id", new_id.name());
        EXPECT_EQ(OptimizedName("add_hoist_mul"), new_id.input(0));
      }
    }
  }
}

TEST_F(ArithmeticOptimizerTest, FuseConjAndTranspose) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output re = ops::Const(s.WithOpName("re"), {1.0, 2.0, 3.0, 4.0}, {2, 2});
  Output im = ops::Const(s.WithOpName("im"), {5.0, 6.0, 7.0, 8.0}, {2, 2});
  Output z = ops::Complex(s.WithOpName("z"), re, im);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
  Output conj = ops::Conj(s.WithOpName("conj"), z);
  Output transp = ops::Transpose(s.WithOpName("trans"), conj, perm);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(7, output.node_size());
  EXPECT_EQ(OptimizedName("trans_fused"), output.node(6).name());
  EXPECT_EQ("ConjugateTranspose", output.node(6).op());
  EXPECT_EQ("z", output.node(6).input(0));
  EXPECT_EQ("perm", output.node(6).input(1));
}

TEST_F(ArithmeticOptimizerTest, FuseConjAndConjugateTranspose) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output re = ops::Const(s.WithOpName("re"), {1.0, 2.0, 3.0, 4.0}, {2, 2});
  Output im = ops::Const(s.WithOpName("im"), {5.0, 6.0, 7.0, 8.0}, {2, 2});
  Output z = ops::Complex(s.WithOpName("z"), re, im);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
  Output conj = ops::Conj(s.WithOpName("conj"), z);
  Output transp =
      ops::ConjugateTranspose(s.WithOpName("conjugate_trans"), conj, perm);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(7, output.node_size());
  EXPECT_EQ(OptimizedName("conjugate_trans_fused"), output.node(6).name());
  EXPECT_EQ("Transpose", output.node(6).op());
  EXPECT_EQ("z", output.node(6).input(0));
  EXPECT_EQ("perm", output.node(6).input(1));
}

TEST_F(ArithmeticOptimizerTest, FuseTransposeAndConj) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output re = ops::Const(s.WithOpName("re"), {1.0, 2.0, 3.0, 4.0}, {2, 2});
  Output im = ops::Const(s.WithOpName("im"), {5.0, 6.0, 7.0, 8.0}, {2, 2});
  Output z = ops::Complex(s.WithOpName("z"), re, im);
  Output perm = ops::Const(s.WithOpName("perm"), {1, 0}, {2});
  Output trans = ops::Transpose(s.WithOpName("trans"), z, perm);
  Output conj = ops::Conj(s.WithOpName("conj"), trans);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(7, output.node_size());
  EXPECT_EQ(OptimizedName("conj_fused"), output.node(6).name());
  EXPECT_EQ("ConjugateTranspose", output.node(6).op());
  EXPECT_EQ("z", output.node(6).input(0));
  EXPECT_EQ("perm", output.node(6).input(1));
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

    ArithmeticOptimizer optimizer;
    GraphDef output;
    Status status = optimizer.Optimize(nullptr, item, &output);
    TF_EXPECT_OK(status);
    // Run the optimizer twice to make sure the rewrite is idempotent.
    item.graph.Swap(&output);
    status = optimizer.Optimize(nullptr, item, &output);
    TF_EXPECT_OK(status);

    EXPECT_EQ(7, output.node_size());
    EXPECT_EQ(OptimizedName("matmul_fused"), output.node(6).name());
    EXPECT_EQ("a", output.node(6).input(0));
    EXPECT_EQ("b", output.node(6).input(1));
    if (matmul_type == "BatchMatMul") {
      EXPECT_TRUE(output.node(6).attr().at("adj_x").b());
      EXPECT_TRUE(output.node(6).attr().at("adj_y").b());
    } else {
      EXPECT_TRUE(output.node(6).attr().at("transpose_a").b());
      EXPECT_TRUE(output.node(6).attr().at("transpose_b").b());
    }
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

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer(RewriterConfig::AGGRESSIVE)
                   .Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(0, std::count_if(
                   output.node().begin(), output.node().end(),
                   [](const NodeDef& node) { return node.op() == "Reshape"; }));
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

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer(RewriterConfig::AGGRESSIVE)
                   .Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(1, std::count_if(
                   output.node().begin(), output.node().end(),
                   [](const NodeDef& node) { return node.op() == "Reshape"; }));
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
  TF_EXPECT_OK(ArithmeticOptimizer(RewriterConfig::AGGRESSIVE)
                   .Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(1, std::count_if(
                   output.node().begin(), output.node().end(),
                   [](const NodeDef& node) { return node.op() == "Reshape"; }));
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

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(1, std::count_if(
                   output.node().begin(), output.node().end(),
                   [](const NodeDef& node) { return node.op() == "Reshape"; }));
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

  item.graph = output;
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

  item.graph = output;
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

TEST_F(ArithmeticOptimizerTest, RemoveInverseTransposes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs_shape =
      ops::Const(s.WithOpName("inputs_shape"), {8, 3, 28, 28}, {4});
  Output inputs =
      ops::RandomUniform(s.WithOpName("inputs"), inputs_shape, DT_FLOAT);
  Output perm1 = ops::Const(s.WithOpName("perm1"), {0, 2, 3, 1}, {4});
  Output perm2 = ops::Const(s.WithOpName("perm2"), {0, 3, 1, 2}, {4});
  Output transpose1 = ops::Transpose(s.WithOpName("transpose1"), inputs, perm1);
  Output transpose2 =
      ops::Transpose(s.WithOpName("transpose2"), transpose1, perm2);
  Output outputs = ops::Identity(s.WithOpName("outputs"), transpose2);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  std::set<string> nodes_after_optimization;
  for (const NodeDef& node : output.node()) {
    nodes_after_optimization.insert(node.name());
  }
  EXPECT_EQ(nodes_after_optimization,
            std::set<string>({"inputs_shape", "inputs", "outputs"}));
}

TEST_F(ArithmeticOptimizerTest, RemoveInverseTransposesMultipleOutputs) {
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
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

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
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));
  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

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
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

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

  item.graph = output;
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

  item.graph = output;
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

  item.graph = output;
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

  item.graph = output;
  TF_EXPECT_OK(
      ConstantFolding(/*cpu_device=*/nullptr).Optimize(nullptr, item, &output));

  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  NodeMap node_map(&output);
  const NodeDef* inputs_node = CHECK_NOTNULL(node_map.GetNode("Placeholder"));
  const NodeDef* transpose_node =
      CHECK_NOTNULL(node_map.GetNode(OptimizedName("Transpose_uint8")));
  const NodeDef* cast_node =
      CHECK_NOTNULL(node_map.GetNode(OptimizedName("Cast_new")));
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

  item.graph = output;
  TF_EXPECT_OK(
      ConstantFolding(/*cpu_device=*/nullptr).Optimize(nullptr, item, &output));

  item.graph = output;
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
  Output inputs =
      ops::Placeholder(s, DT_UINT8, ops::Placeholder::Shape({2, 3}));
  Output bc1 = ops::Bitcast(s, inputs, DT_QINT8);
  Output bc2 = ops::Bitcast(s, bc1, DT_INT8);
  Output outputs = ops::Identity(s.WithOpName("outputs"), bc2);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));
  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(1, std::count_if(
                   output.node().begin(), output.node().end(),
                   [](const NodeDef& node) { return node.op() == "Bitcast"; }));
}

TEST_F(ArithmeticOptimizerTest, CombineAndRemoveBitcasts) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs = ops::Placeholder(s, DT_INT8, ops::Placeholder::Shape({2, 3}));
  Output bc1 = ops::Bitcast(s, inputs, DT_QINT8);
  Output bc2 = ops::Bitcast(s, bc1, DT_INT8);
  Output outputs = ops::Identity(s.WithOpName("outputs"), bc2);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));
  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(0, std::count_if(
                   output.node().begin(), output.node().end(),
                   [](const NodeDef& node) { return node.op() == "Bitcast"; }));
}

TEST_F(ArithmeticOptimizerTest, RemoveRedundantCast) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output inputs = ops::Placeholder(s, DT_INT8, ops::Placeholder::Shape({2, 3}));
  Output cast = ops::Cast(s, inputs, DT_INT8);
  Output outputs = ops::Identity(s.WithOpName("outputs"), cast);

  GrapplerItem item;
  item.fetch = {"outputs"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphDef output;
  TF_EXPECT_OK(ArithmeticOptimizer().Optimize(nullptr, item, &output));
  item.graph = output;
  TF_EXPECT_OK(ModelPruner().Optimize(nullptr, item, &output));

  EXPECT_EQ(0, std::count_if(
                   output.node().begin(), output.node().end(),
                   [](const NodeDef& node) { return node.op() == "Cast"; }));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
