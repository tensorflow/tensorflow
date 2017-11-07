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

class ArithmeticOptimizerTest : public ::testing::Test {};

TEST_F(ArithmeticOptimizerTest, NoOp) {
  // This trivial graph is so basic there's nothing to optimize.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status s = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(s);

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

TEST_F(ArithmeticOptimizerTest, OpDedupping) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c1 = ops::Const(s.WithOpName("c1"), {3.14, 2.7}, {1, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), {3.14, 2.7}, {1, 2});
  Output mul = ops::Mul(s.WithOpName("mul"), c1, c2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(2, output.node_size());
  const NodeDef& new_c1 = output.node(0);
  EXPECT_EQ("c1", new_c1.name());
  const NodeDef& new_mul = output.node(1);
  EXPECT_EQ("mul", new_mul.name());
  EXPECT_EQ(2, new_mul.input_size());
  EXPECT_EQ("c1", new_mul.input(0));
  EXPECT_EQ("c1", new_mul.input(1));
}

TEST_F(ArithmeticOptimizerTest, OpDedupCommutative) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c1 = ops::Const(s.WithOpName("c1"), {1.0f, 2.0f}, {1, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), {3.0f, 4.0f}, {1, 2});
  Output mul1 = ops::Mul(s.WithOpName("mul1"), c1, c2);
  Output mul2 = ops::Mul(s.WithOpName("mul2"), c2, c1);
  Output mul3 = ops::Mul(s.WithOpName("mul3"), mul1, mul2);
  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  const NodeDef& new_mul3 = output.node(3);
  EXPECT_EQ("mul3", new_mul3.name());
  EXPECT_EQ(2, new_mul3.input_size());
  EXPECT_EQ("mul1", new_mul3.input(0));
  EXPECT_EQ("mul1", new_mul3.input(1));
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

TEST_F(ArithmeticOptimizerTest, SimplifyReplaceTrivialSums) {
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

  //  VLOG(2) << output.DebugString();
  EXPECT_EQ(5, output.node_size());
  const NodeDef& new_const = output.node(3);
  EXPECT_EQ("add_const", new_const.name());
  const NodeDef& new_mul = output.node(4);
  EXPECT_EQ("add_mul", new_mul.name());
  EXPECT_EQ("add_const", new_mul.input(0));
  EXPECT_EQ("x", new_mul.input(1));
  const NodeDef& new_id = output.node(2);
  EXPECT_EQ("id", new_id.name());
  EXPECT_EQ("add_mul", new_id.input(0));
}

TEST_F(ArithmeticOptimizerTest, SimplifyHoistFactor) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(s.WithOpName("x"), {1.0f, 2.0f}, {1, 2});
  Output y1 = ops::Const(s.WithOpName("y1"), {3.0f, 4.0f}, {1, 2});
  Output y2 = ops::Const(s.WithOpName("y2"), {5.0f, 6.0f}, {1, 2});
  Output mul1 = ops::Mul(s.WithOpName("mul1"), x, y1);
  Output mul2 = ops::Mul(s.WithOpName("mul2"), y2, x);
  Output add = ops::Add(s.WithOpName("add"), mul1, mul2);
  Output id = ops::Identity(s.WithOpName("id"), add);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ArithmeticOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  LOG(INFO) << output.DebugString();
  EXPECT_EQ(9, output.node_size());
  const NodeDef& new_add = output.node(8);
  EXPECT_EQ("add_hoist", new_add.name());
  EXPECT_EQ("y1", new_add.input(0));
  EXPECT_EQ("y2", new_add.input(1));
  const NodeDef& new_mul = output.node(7);
  EXPECT_EQ("mul1_hoist", new_mul.name());
  EXPECT_EQ("x", new_mul.input(0));
  EXPECT_EQ("add_hoist", new_mul.input(1));
  const NodeDef& new_id = output.node(6);
  EXPECT_EQ("id", new_id.name());
  EXPECT_EQ("mul1_hoist", new_id.input(0));
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
  LOG(INFO) << output.DebugString();

  EXPECT_EQ(7, output.node_size());
  EXPECT_EQ("trans_fused", output.node(6).name());
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
  LOG(INFO) << output.DebugString();

  EXPECT_EQ(7, output.node_size());
  EXPECT_EQ("conjugate_trans_fused", output.node(6).name());
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
  LOG(INFO) << output.DebugString();

  EXPECT_EQ(7, output.node_size());
  EXPECT_EQ("conj_fused", output.node(6).name());
  EXPECT_EQ("ConjugateTranspose", output.node(6).op());
  EXPECT_EQ("z", output.node(6).input(0));
  EXPECT_EQ("perm", output.node(6).input(1));
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

  for (const auto& node : output.node()) {
    LOG(INFO) << node.DebugString();
  }

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

  for (const auto& node : output.node()) {
    LOG(INFO) << node.DebugString();
  }

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
      CHECK_NOTNULL(node_map.GetNode("Transpose_uint8"));
  const NodeDef* cast_node = CHECK_NOTNULL(node_map.GetNode("Cast_new"));
  const NodeDef* weights_node =
      CHECK_NOTNULL(node_map.GetNode("weights_scaled"));
  const NodeDef* conv_node = CHECK_NOTNULL(node_map.GetNode("Conv2D"));

  EXPECT_EQ(output.node_size(), 7);
  EXPECT_EQ(transpose_node->input(0), inputs_node->name());
  EXPECT_EQ(cast_node->input(0), transpose_node->name());
  EXPECT_EQ(conv_node->input(0), cast_node->name());
  EXPECT_EQ(conv_node->input(1), weights_node->name());
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
