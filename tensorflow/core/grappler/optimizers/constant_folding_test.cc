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

#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {
namespace grappler {
namespace {

class ConstantFoldingTest : public ::testing::Test {
 protected:
  std::vector<Tensor> EvaluateNodes(const GraphDef& graph,
                                    const std::vector<string>& fetch) {
    SessionOptions options;
    std::unique_ptr<tensorflow::Session> session(NewSession(options));
    TF_CHECK_OK(session->Create(graph));
    RunOptions run_options;
    std::vector<Tensor> output_tensors;
    TF_CHECK_OK(
        session->Run(run_options, {}, fetch, fetch, &output_tensors, nullptr));
    TF_CHECK_OK(session->Close());
    return output_tensors;
  }
};

TEST_F(ConstantFoldingTest, SimpleFolding) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 1.0f, {1});
  Output b = ops::Const(s.WithOpName("b"), 2.0f, {1});
  Output c = ops::AddN(s.WithOpName("c").WithDevice("/CPU:0"), {a, b});
  Output d = ops::AddN(s.WithOpName("d"), {b, c});

  GrapplerItem item;
  item.fetch.push_back("d");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(1, output.node_size());

  const NodeDef& node_d = output.node(0);
  EXPECT_EQ("d", node_d.name());
  EXPECT_EQ("Const", node_d.op());

  std::vector<string> fetch = {"d"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(ConstantFoldingTest, AddTree) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output c1 = ops::Const(s.WithOpName("c1"), 2.0f, {1});
  Output c2 = ops::Const(s.WithOpName("c2"), 2.0f, {2});
  Output c4 = ops::Const(s.WithOpName("c4"), 4.0f, {2});
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output add_child = ops::Add(s.WithOpName("add_child"), c2, x);
  Output add_parent = ops::Add(s.WithOpName("add_parent"), c1, add_child);
  Output mul_child = ops::Mul(s.WithOpName("mul_child"), c2, x);
  Output mul_parent = ops::Mul(s.WithOpName("mul_parent"), c1, mul_child);
  Output addmul_child = ops::Add(s.WithOpName("addmul_child"), c2, x);
  Output addmul_parent =
      ops::Mul(s.WithOpName("addmul_parent"), c1, addmul_child);

  GrapplerItem item;
  item.fetch = {"add_parent", "mul_parent", "addmul_parent"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(9, output.node_size());

  // We expect the following rewrite(s) to occur (for both Add and Mul):
  //    +                +             +
  //   / \              / \           / \
  // 2.0   +     -->   x   +    -->  x  4.0
  //      / \             / \
  //    2.0  x          2.0 2.0

  for (const auto& node : output.node()) {
    if (node.name() == "add_child") {
      EXPECT_EQ("Const", node.op());
      TensorProto t = node.attr().at("value").tensor();
      EXPECT_EQ(1, t.tensor_shape().dim_size());
      EXPECT_EQ(2, t.tensor_shape().dim(0).size());
    } else if (node.name() == "add_parent") {
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("add_child", node.input(1));
    } else if (node.name() == "mul_child") {
      EXPECT_EQ("Const", node.op());
      TensorProto t = node.attr().at("value").tensor();
      EXPECT_EQ(1, t.tensor_shape().dim_size());
      EXPECT_EQ(2, t.tensor_shape().dim(0).size());
    } else if (node.name() == "mul_parent") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("mul_child", node.input(1));
    } else if (node.name() == "addmul_child") {
      // Unchanged.
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("c2", node.input(0));
      EXPECT_EQ("x", node.input(1));
    }
  }

  // Check that the reciprocals have the expected value.
  std::vector<string> fetch = {"c4"};
  auto tensor_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(fetch.size(), tensor_expected.size());
  fetch = {"add_child", "mul_child"};
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<float>(tensor_expected[0], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, NeutralElement) {
  for (bool use_const : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                                ops::Placeholder::Shape(TensorShape({2, 2})));
    Output y = ops::Placeholder(s.WithOpName("y"), DT_FLOAT,
                                ops::Placeholder::Shape(TensorShape({2, 2})));
    Output a = ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                                ops::Placeholder::Shape(TensorShape({3, 2})));
    Output b = ops::Placeholder(s.WithOpName("b"), DT_FLOAT,
                                ops::Placeholder::Shape(TensorShape({2, 3})));
    Output bias = ops::Placeholder(s.WithOpName("bias"), DT_FLOAT,
                                   ops::Placeholder::Shape(TensorShape({2})));
    Output zeros = !use_const ? ops::ZerosLike(s.WithOpName("zeros"), x)
                              : ops::Const(s.WithOpName("zeros"), 0.0f, {2, 2});
    Output zeros_1d = ops::Const(s.WithOpName("zeros_1d"), 0.0f, {2});
    Output ones = !use_const ? ops::OnesLike(s.WithOpName("ones"), x)
                             : ops::Const(s.WithOpName("ones"), 1.0f, {2, 2});
    Output mul1 = ops::Mul(s.WithOpName("mul1"), x, zeros);
    Output mul2 = ops::Mul(s.WithOpName("mul2"), zeros, y);
    Output mul3 = ops::Mul(s.WithOpName("mul3"), x, ones);
    Output mul4 = ops::Mul(s.WithOpName("mul4"), ones, y);
    Output mul5 = ops::Mul(s.WithOpName("mul5"), x, zeros_1d);
    Output mul6 = ops::Mul(s.WithOpName("mul6"), zeros_1d, y);
    Output div1 = ops::Div(s.WithOpName("div1"), x, ones);
    Output div2 = ops::Div(s.WithOpName("div2"), ones, y);
    Output matmul1 = ops::MatMul(s.WithOpName("matmul1"), x, zeros);
    Output matmul2 = ops::MatMul(s.WithOpName("matmul2"), zeros, y);
    Output matmul3 = ops::MatMul(s.WithOpName("matmul3"), a, zeros);
    Output matmul4 = ops::MatMul(s.WithOpName("matmul4"), zeros, b);
    Output add1 = ops::Add(s.WithOpName("add1"), x, zeros);
    Output add2 = ops::Add(s.WithOpName("add2"), zeros, y);
    Output bias_add1 = ops::BiasAdd(s.WithOpName("bias_add1"), x, zeros_1d);
    Output bias_add2 = ops::BiasAdd(s.WithOpName("bias_add2"), zeros, bias);
    Output sub1 = ops::Sub(s.WithOpName("sub1"), x, zeros);
    Output sub2 = ops::Sub(s.WithOpName("sub2"), zeros, y);
    Output addn =
        ops::AddN(s.WithOpName("addn"),
                  {mul1, mul2, mul3, mul4, mul5, mul6, div1, div2, matmul1,
                   matmul2, add1, add2, bias_add1, bias_add2, sub1, sub2});
    GrapplerItem item;
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    item.fetch = {"addn", "matmul3", "matmul4"};

    ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                              nullptr /* cpu_device */);
    GraphDef output;
    Status status = optimizer.Optimize(nullptr, item, &output);
    TF_EXPECT_OK(status);

    EXPECT_EQ(27, output.node_size());
    for (int i = 0; i < output.node_size(); ++i) {
      const NodeDef& node = output.node(i);
      const string& name = node.name();
      if (name == "mul1") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^x", node.input(0));
        EXPECT_EQ("^zeros", node.input(1));
      } else if (name == "mul2") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^zeros", node.input(0));
        EXPECT_EQ("^y", node.input(1));
      } else if (name == "mul3") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^ones", node.input(1));
      } else if (name == "mul4") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ("^ones", node.input(1));
      } else if (name == "mul5") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^x", node.input(0));
        EXPECT_EQ("^zeros_1d", node.input(1));
      } else if (name == "mul6") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^zeros_1d", node.input(0));
        EXPECT_EQ("^y", node.input(1));
      } else if (name == "div1") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^ones", node.input(1));
      } else if (name == "div2") {
        EXPECT_EQ("Reciprocal", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ("^ones", node.input(1));
      } else if (name == "matmul1") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^x", node.input(0));
        EXPECT_EQ("^zeros", node.input(1));
      } else if (name == "matmul2") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^zeros", node.input(0));
        EXPECT_EQ("^y", node.input(1));
      } else if (name == "matmul3") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^a", node.input(0));
        EXPECT_EQ("^zeros", node.input(1));
        TensorProto t = node.attr().at("value").tensor();
        EXPECT_EQ(1, t.float_val_size());
        EXPECT_EQ(0, t.float_val(0));
        EXPECT_EQ(2, t.tensor_shape().dim_size());
        EXPECT_EQ(3, t.tensor_shape().dim(0).size());
        EXPECT_EQ(2, t.tensor_shape().dim(1).size());
      } else if (name == "matmul4") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^zeros", node.input(0));
        EXPECT_EQ("^b", node.input(1));
        TensorProto t = node.attr().at("value").tensor();
        EXPECT_EQ(1, t.float_val_size());
        EXPECT_EQ(0, t.float_val(0));
        EXPECT_EQ(2, t.tensor_shape().dim_size());
        EXPECT_EQ(2, t.tensor_shape().dim(0).size());
        EXPECT_EQ(3, t.tensor_shape().dim(1).size());
      } else if (name == "add1") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^zeros", node.input(1));
      } else if (name == "add2") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ("^zeros", node.input(1));
      } else if (name == "bias_add1") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^zeros_1d", node.input(1));
      } else if (name == "bias_add2") {
        // We don't eliminate this one, because it requires broadcasting.
        EXPECT_EQ("BiasAdd", node.op());
        EXPECT_EQ("zeros", node.input(0));
        EXPECT_EQ("bias", node.input(1));
      } else if (name == "sub1") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^zeros", node.input(1));
      } else if (name == "sub2") {
        // We don't handle this case yet.
        EXPECT_EQ("Sub", node.op());
        EXPECT_EQ("zeros", node.input(0));
        EXPECT_EQ("y", node.input(1));
      }
      const std::set<string> square_zero_const{"mul1", "mul2",    "mul5",
                                               "mul6", "matmul1", "matmul2"};
      if (square_zero_const.count(name) > 0) {
        TensorProto t = node.attr().at("value").tensor();
        EXPECT_EQ(1, t.float_val_size());
        EXPECT_EQ(0, t.float_val(0));
        EXPECT_EQ(2, t.tensor_shape().dim_size());
        EXPECT_EQ(2, t.tensor_shape().dim(0).size());
        EXPECT_EQ(2, t.tensor_shape().dim(1).size());
      }
    }
  }
}

TEST_F(ConstantFoldingTest, StrengthReduce_Reciprocal) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output cf_half = ops::Const(s.WithOpName("cf_half"), 0.5f, {1});
  Output xf = ops::Placeholder(s.WithOpName("xf"), DT_FLOAT,
                               ops::Placeholder::Shape(TensorShape({2, 2})));
  Output xi = ops::Placeholder(s.WithOpName("xi"), DT_INT32,
                               ops::Placeholder::Shape(TensorShape({2, 2})));
  Output ci = ops::Const(s.WithOpName("ci"), 2, {1});
  Output cf = ops::Const(s.WithOpName("cf"), 2.0f, {1});
  Output div_i = ops::Div(s.WithOpName("div_i"), xi, ci);
  Output div_f = ops::Div(s.WithOpName("div_f"), xf, cf);
  Output realdiv = ops::RealDiv(s.WithOpName("realdiv"), xf, cf);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"div_f", "div_i", "realdiv"};
  ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                            nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(8, output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    const string& name = node.name();
    if (name == "div_i") {
      // Integer division is unchanged.
      EXPECT_EQ("Div", node.op());
      EXPECT_EQ("xi", node.input(0));
      EXPECT_EQ("ci", node.input(1));
    } else if (name == "div_f") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ("xf", node.input(0));
      EXPECT_EQ("ConstantFolding/div_f_recip", node.input(1));
    } else if (name == "realdiv") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ("xf", node.input(0));
      EXPECT_EQ("ConstantFolding/realdiv_recip", node.input(1));
    } else if (name == "ConstantFolding/div_f_recip") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("dtype").type());
      TensorProto t = node.attr().at("value").tensor();
      EXPECT_EQ(DT_FLOAT, t.dtype());
      EXPECT_EQ(1, t.tensor_shape().dim_size());
      EXPECT_EQ(1, t.tensor_shape().dim(0).size());
    } else if (name == "ConstantFolding/realdiv_recip") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("dtype").type());
      TensorProto t = node.attr().at("value").tensor();
      EXPECT_EQ(DT_FLOAT, t.dtype());
      EXPECT_EQ(1, t.tensor_shape().dim_size());
      EXPECT_EQ(1, t.tensor_shape().dim(0).size());
    }
  }

  // Check that the reciprocals have the expected value.
  std::vector<string> fetch = {"cf_half"};
  auto tensor_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(fetch.size(), tensor_expected.size());
  fetch = {"ConstantFolding/div_f_recip", "ConstantFolding/realdiv_recip"};
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<float>(tensor_expected[0], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, NeutralElement_PartialShape_UnknownOutputShape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x_known =
      ops::Placeholder(s.WithOpName("x_known"), DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({2, 2})));
  Output x_partially_known =
      ops::Placeholder(s.WithOpName("x_partially_unknown"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output x_unknown = ops::Placeholder(s.WithOpName("x_unknown"), DT_FLOAT);
  Output zeros_known = ops::ZerosLike(s.WithOpName("zeros_known"), x_known);
  Output zeros_partially_known =
      ops::ZerosLike(s.WithOpName("zeros_partially_known"), x_partially_known);
  Output zeros_unknown =
      ops::ZerosLike(s.WithOpName("zeros_unknown"), x_unknown);

  // Multiplies without any additional ops to supply the output shape.
  int count = 0;
  std::vector<Output> muls;
  std::unordered_set<string> not_converted;
  std::unordered_set<string> to_const;
  std::unordered_set<string> to_identity;
  for (const auto* x : {&x_known, &x_partially_known, &x_unknown}) {
    for (const auto* zeros :
         {&zeros_known, &zeros_partially_known, &zeros_unknown}) {
      const string name = strings::StrCat("mul_", count++);
      muls.push_back(ops::Mul(s.WithOpName(name), *x, *zeros));
      if (x == &x_partially_known && zeros == &zeros_partially_known) {
        to_identity.insert(name);
      } else if (x == &x_unknown || zeros == &zeros_unknown) {
        not_converted.insert(name);
      } else {
        to_const.insert(name);
      }
    }
  }

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                            nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  LOG(INFO) << output.DebugString();

  EXPECT_EQ(15, output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    const string& name = node.name();
    if (to_const.count(name) > 0) {
      EXPECT_EQ("Const", node.op()) << node.name();
    } else if (to_identity.count(name) > 0) {
      EXPECT_EQ("Identity", node.op()) << node.name();
    } else if (not_converted.count(name) > 0) {
      EXPECT_EQ("Mul", node.op()) << node.name();
    }
  }
}

TEST_F(ConstantFoldingTest, NeutralElement_PartialShape_KnownOutputShape) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output known_shape = ops::Const(s.WithOpName("known_shape"), 0.0f, {2, 2});
  Output x_partially_known =
      ops::Placeholder(s.WithOpName("x_partially_unknown"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output x_unknown = ops::Placeholder(s.WithOpName("x_unknown"), DT_FLOAT);
  Output zeros_partially_known =
      ops::ZerosLike(s.WithOpName("zeros_partially_known"), x_partially_known);
  Output zeros_unknown =
      ops::ZerosLike(s.WithOpName("zeros_unknown"), x_unknown);

  // If at least one of the inputs to AddN has a known shape, shape inference
  // will propagate the shape back to the inputs of AddN, making the
  // output shapes of all its inputs known
  std::vector<Output> muls_deduced_output_shape;
  std::unordered_set<string> to_const;
  int count = 0;
  for (const auto& x : {x_partially_known, x_unknown}) {
    for (const auto& zeros : {zeros_partially_known, zeros_unknown}) {
      const string name = strings::StrCat("mul_", count++);
      muls_deduced_output_shape.push_back(
          ops::Mul(s.WithOpName(name), x, zeros));
      to_const.insert(name);
    }
  }
  // We add a known shape as input to AddN to propagate it back to the
  // multiplies above, which means they can all be turned into Const nodes.
  muls_deduced_output_shape.push_back(known_shape);
  Output addn1 = ops::AddN(s.WithOpName("addn1"), muls_deduced_output_shape);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                            nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  LOG(INFO) << output.DebugString();

  EXPECT_EQ(10, output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    const string& name = node.name();
    if (to_const.count(name) > 0) {
      EXPECT_EQ("Const", node.op()) << node.name();
      EXPECT_EQ(2, node.input_size());
      EXPECT_TRUE(IsControlInput(node.input(0)));
      EXPECT_TRUE(IsControlInput(node.input(1)));
    }
  }
}

TEST_F(ConstantFoldingTest, CreateConstNodes) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

#define MAKE_TEST_GRAPH(TYPE)                                               \
  Output TYPE##_const =                                                     \
      ops::Const(s.WithOpName(#TYPE "_const"), static_cast<TYPE>(10), {5}); \
  Output TYPE##_mul =                                                       \
      ops::Mul(s.WithOpName(#TYPE "_mul"), TYPE##_const, TYPE##_const);     \
  Output TYPE##_id = ops::Identity(s.WithOpName(#TYPE "_id"), TYPE##_mul)

  MAKE_TEST_GRAPH(float);
  MAKE_TEST_GRAPH(double);
  MAKE_TEST_GRAPH(int64);
  MAKE_TEST_GRAPH(int32);
  MAKE_TEST_GRAPH(int16);
  MAKE_TEST_GRAPH(int8);
  MAKE_TEST_GRAPH(uint8);
#undef MAKE_TEST_GRAPH

  Output bool_const = ops::Const(s.WithOpName("bool_const"), true, {5});
  Output bool_and =
      ops::LogicalAnd(s.WithOpName("bool_and"), bool_const, bool_const);
  Output bool_id = ops::Identity(s.WithOpName("bool_id"), bool_and);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(24, output.node_size());
  for (const NodeDef& node : output.node()) {
#define CHECK_RESULT(TYPE, FIELD)                                             \
  if (node.name() == #TYPE "_mul") {                                          \
    EXPECT_EQ(5,                                                              \
              node.attr().at("value").tensor().tensor_shape().dim(0).size()); \
    EXPECT_EQ(1, node.attr().at("value").tensor().FIELD##_val_size());        \
    EXPECT_EQ(10 * 10, node.attr().at("value").tensor().FIELD##_val(0));      \
  }

    CHECK_RESULT(float, float);
    CHECK_RESULT(double, double);
    CHECK_RESULT(int64, int64);
    CHECK_RESULT(int32, int);
    CHECK_RESULT(int16, int);
    CHECK_RESULT(int8, int);
    CHECK_RESULT(uint8, int);
#undef CHECK_RESULT

    if (node.name() == "bool_and") {
      EXPECT_EQ(5,
                node.attr().at("value").tensor().tensor_shape().dim(0).size());
      EXPECT_EQ(1, node.attr().at("value").tensor().bool_val_size());
      EXPECT_EQ(true && true, node.attr().at("value").tensor().bool_val(0));
    }
  }
}

TEST_F(ConstantFoldingTest, FoldingNodeWithTwoOutputs) {
  // Build a simple graph with a few trivially prunable ops.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 10, {5});
  auto b = ops::Unique(s.WithOpName("b"), {a});
  Output c = ops::Identity(s.WithOpName("c"), {b.y});
  Output d = ops::Identity(s.WithOpName("d"), {b.idx});
  Output e = ops::Identity(s.WithOpName("e"), {c});
  Output f = ops::Identity(s.WithOpName("f"), {d});

  GrapplerItem item;
  item.fetch.push_back("e");
  item.fetch.push_back("f");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(2, output.node_size());

  const NodeDef& new_c = output.node(0);
  EXPECT_EQ("e", new_c.name());
  EXPECT_EQ("Const", new_c.op());

  const NodeDef& new_d = output.node(1);
  EXPECT_EQ("f", new_d.name());
  EXPECT_EQ("Const", new_d.op());

  std::vector<string> fetch = {"e", "f"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(fetch.size(), tensors_expected.size());
  EXPECT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, ControlDependencies) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(scope.WithOpName("dflt"), 3.14f, {1});
  Output p1 = ops::PlaceholderWithDefault(scope.WithOpName("p1"), dflt, {1});
  Output p2 = ops::PlaceholderWithDefault(scope.WithOpName("p2"), dflt, {1});
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(p1), 10, {3});
  Output i1 = ops::Identity(scope.WithOpName("i1"), {c});
  Output i2 =
      ops::Identity(scope.WithOpName("i2").WithControlDependencies(p2), {i1});
  Output i3 = ops::Identity(scope.WithOpName("e"), {i2});

  GrapplerItem item;
  item.fetch.push_back("e");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> expected_nodes = {"dflt", "p1", "p2", "e"};
  EXPECT_EQ(output.node_size(), expected_nodes.size());
  int i = 0;
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_EQ(expected_nodes[i], output.node(i).name());
    i++;
    if (node.name() == "e") {
      EXPECT_EQ("Const", node.op());
      ++found;
      auto folded = EvaluateNodes(output, {"e"});
      auto expected = EvaluateNodes(item.graph, {"e"});
      EXPECT_EQ(1, expected.size());
      EXPECT_EQ(1, folded.size());
      test::ExpectTensorEqual<int>(folded[0], expected[0]);
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
      EXPECT_EQ("^p2", node.input(1));
    }
  }
  EXPECT_EQ(1, found);
}

TEST_F(ConstantFoldingTest, ControlDependenciesEmptyFetch) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(scope.WithOpName("dflt"), 3.14f, {1});
  Output p1 = ops::PlaceholderWithDefault(scope.WithOpName("p1"), dflt, {1});
  Output p2 = ops::PlaceholderWithDefault(scope.WithOpName("p2"), dflt, {1});
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(p1), 10, {3});
  Output i1 = ops::Identity(scope.WithOpName("i1"), {c});
  Output i2 =
      ops::Identity(scope.WithOpName("i2").WithControlDependencies(p2), {i1});
  Output i3 = ops::Identity(scope.WithOpName("e"), {i2});

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> expected_nodes = {"dflt", "p1", "p2", "c",
                                        "i1",   "i2", "e"};
  EXPECT_EQ(output.node_size(), expected_nodes.size());
  int i = 0;
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_EQ(expected_nodes[i], output.node(i).name());
    i++;
    if (node.name() == "i1") {
      EXPECT_EQ("Const", node.op());
      ++found;
      auto folded = EvaluateNodes(output, {"i1"});
      auto expected = EvaluateNodes(item.graph, {"i1"});
      EXPECT_EQ(1, expected.size());
      EXPECT_EQ(1, folded.size());
      test::ExpectTensorEqual<int>(folded[0], expected[0]);
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
    }
    if (node.name() == "i2") {
      EXPECT_EQ("Const", node.op());
      ++found;
      auto folded = EvaluateNodes(output, {"i2"});
      auto expected = EvaluateNodes(item.graph, {"i2"});
      EXPECT_EQ(1, expected.size());
      EXPECT_EQ(1, folded.size());
      test::ExpectTensorEqual<int>(folded[0], expected[0]);
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
      EXPECT_EQ("^p2", node.input(1));
    }
  }
  EXPECT_EQ(2, found);
}

TEST_F(ConstantFoldingTest, ControlDependenciesDeduplicate) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output dflt = ops::Const(scope.WithOpName("dflt"), 3.14f, {1});
  Output p1 = ops::PlaceholderWithDefault(scope.WithOpName("p1"), dflt, {1});
  Output p2 = ops::PlaceholderWithDefault(scope.WithOpName("p2"), dflt, {1});
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(p1), 10, {3});
  Output i1 = ops::Identity(scope.WithOpName("i1")
                                .WithControlDependencies(p2)
                                .WithControlDependencies(p1),
                            {c});
  Output i2 = ops::Identity(scope.WithOpName("i2"), {i1});

  GrapplerItem item;
  item.fetch.push_back("i2");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> expected_nodes = {"dflt", "p1", "p2", "i2"};
  EXPECT_EQ(output.node_size(), expected_nodes.size());
  int i = 0;
  for (const auto& node : output.node()) {
    EXPECT_EQ(expected_nodes[i], output.node(i).name());
    i++;
    if (node.name() == "i2") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^p1", node.input(0));
      EXPECT_EQ("^p2", node.input(1));
    }
  }
}

TEST_F(ConstantFoldingTest, VariableNumberOfOutputs) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  // Add a DynamicPartition node to the graph
  Output input = ops::Const(scope.WithOpName("in0"), 314, {3, 4, 5});
  Output indices = ops::Const(scope.WithOpName("indices"), 1, {3, 4});
  int num_partitions = 4;
  ops::DynamicPartition part(scope.WithOpName("partition"), input, indices,
                             num_partitions);

  std::vector<string> outputs;
  for (int i = 0; i < num_partitions; ++i) {
    string part_out_name = strings::StrCat("part_out", i);
    ops::Identity partition_out(scope.WithOpName(part_out_name),
                                {part.outputs[i]});
    outputs.push_back(part_out_name);
  }

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  // Add a ConcatOffset node to the graph
  Tensor initial_val(DT_INT32, TensorShape({3}));
  test::FillIota<int>(&initial_val, 7);
  for (int i = 1; i < 5; ++i) {
    TF_CHECK_OK(NodeDefBuilder(strings::StrCat("in", i), "Const")
                    .Attr("dtype", DT_INT32)
                    .Attr("value", initial_val)
                    .Finalize(item.graph.add_node()));
  }
  Tensor concat_dim(DT_INT32, TensorShape({}));
  test::FillIota<int>(&concat_dim, 0);
  TF_CHECK_OK(NodeDefBuilder("concat_dim", "Const")
                  .Attr("dtype", DT_INT32)
                  .Attr("value", concat_dim)
                  .Finalize(item.graph.add_node()));

  TF_CHECK_OK(NodeDefBuilder("concat_offsets", "ConcatOffset")
                  .Input("concat_dim", 0, DT_INT32)
                  .Input({NodeDefBuilder::NodeOut("in1", 0, DT_INT32),
                          NodeDefBuilder::NodeOut("in2", 0, DT_INT32),
                          NodeDefBuilder::NodeOut("in3", 0, DT_INT32),
                          NodeDefBuilder::NodeOut("in4", 0, DT_INT32)})
                  .Finalize(item.graph.add_node()));

  for (int i = 0; i < 4; ++i) {
    string concat_offset_out_name = strings::StrCat("concat_offset_out", i);
    TF_CHECK_OK(NodeDefBuilder(concat_offset_out_name, "Identity")
                    .Attr("T", DT_INT32)
                    .Input("concat_offsets", i, DT_INT32)
                    .Finalize(item.graph.add_node()));
    outputs.push_back(concat_offset_out_name);
  }

  item.fetch = outputs;
  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int constant_folded = 0;
  for (const auto& node : output.node()) {
    if (node.name().find("part_out") != string::npos ||
        node.name().find("concat_offset_out") != string::npos) {
      ++constant_folded;
      EXPECT_EQ("Const", node.op());
    }
  }
  EXPECT_EQ(8, constant_folded);

  auto expected = EvaluateNodes(item.graph, outputs);
  auto optimized = EvaluateNodes(output, outputs);
  ASSERT_EQ(expected.size(), optimized.size());
  for (int i = 0; i < expected.size(); ++i) {
    test::ExpectTensorEqual<int>(expected[i], optimized[i]);
  }
}

TEST_F(ConstantFoldingTest, ShapeMaterialization) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output v1 = ops::Variable(scope.WithOpName("v1"), {3}, DT_FLOAT);
  Output v2 = ops::Variable(scope.WithOpName("v2"), {5, 7}, DT_FLOAT);
  Output v3 = ops::Variable(scope.WithOpName("v3"), {11, 13}, DT_FLOAT);
  Output rank = ops::Rank(scope.WithOpName("rank"), v1);
  Output shape = ops::Shape(scope.WithOpName("shape"), v2);
  Output size = ops::Size(scope.WithOpName("size"), v3);
  Output p1 = ops::Multiply(scope.WithOpName("p1"), size, rank);
  Output p2 = ops::Multiply(scope.WithOpName("p2"), p1, shape);

  GrapplerItem item;
  item.fetch.push_back("p2");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "p2") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("^v3", node.input(0));
      EXPECT_EQ("^v1", node.input(1));
      EXPECT_EQ("^v2", node.input(2));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      // rank = 1, shape = (5, 7), size = 143 = 11*13
      // p2 = (715, 1001) = (5*143, 7*143)
      EXPECT_EQ(715, value.flat<int>()(0));
      EXPECT_EQ(1001, value.flat<int>()(1));
    }
  }
  EXPECT_EQ(1, found);
}

TEST_F(ConstantFoldingTest, ShapeMaterializationEmptyFetch) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output v1 = ops::Variable(scope.WithOpName("v1"), {3}, DT_FLOAT);
  Output v2 = ops::Variable(scope.WithOpName("v2"), {5, 7}, DT_FLOAT);
  Output v3 = ops::Variable(scope.WithOpName("v3"), {11, 13}, DT_FLOAT);
  Output rank = ops::Rank(scope.WithOpName("rank"), v1);
  Output shape = ops::Shape(scope.WithOpName("shape"), v2);
  Output size = ops::Size(scope.WithOpName("size"), v3);
  Output p1 = ops::Multiply(scope.WithOpName("p1"), size, rank);
  Output p2 = ops::Multiply(scope.WithOpName("p2"), p1, shape);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "size") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^v3", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(11 * 13, value.flat<int>()(0));
    } else if (node.name() == "rank") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^v1", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(1, value.flat<int>()(0));
    } else if (node.name() == "shape") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^v2", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(5, value.flat<int>()(0));
      EXPECT_EQ(7, value.flat<int>()(1));
    }
  }
  EXPECT_EQ(3, found);
}

TEST_F(ConstantFoldingTest, ShapeMaterializationShapeN) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output v1 = ops::Variable(scope.WithOpName("v1"), {3, -1}, DT_FLOAT);
  Output v2 = ops::Variable(scope.WithOpName("v2"), {}, DT_FLOAT);
  Output v3 = ops::Variable(scope.WithOpName("v3"), {4, 6}, DT_FLOAT);
  auto s = ops::ShapeN(scope.WithOpName("s"), {v1, v2, v3});
  Output i1a = ops::Identity(scope.WithOpName("i1a"), s[0]);
  Output i1b = ops::Identity(scope.WithOpName("i1b"), s[0]);
  Output i2a = ops::Identity(scope.WithOpName("i2a"), s[1]);
  Output i2b = ops::Identity(scope.WithOpName("i2b"), s[1]);
  Output i2c = ops::Identity(scope.WithOpName("i2c"), s[1]);
  Output i3a = ops::Identity(scope.WithOpName("i3a"), s[2]);
  Output i3b = ops::Identity(scope.WithOpName("i3b"), s[2]);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_NE(AddPrefixToNodeName("s-0", kConstantFoldingConst), node.name());
    EXPECT_NE(AddPrefixToNodeName("s-1", kConstantFoldingConst), node.name());
    if (node.name() == "i1a" || node.name() == "i1b") {
      ++found;
      EXPECT_EQ("s", node.input(0));
    }
    if (node.name() == "i2a" || node.name() == "i2b" || node.name() == "i2c") {
      ++found;
      EXPECT_EQ("s:1", node.input(0));
    }
    if (node.name() == "i3a" || node.name() == "i3b") {
      ++found;
      EXPECT_EQ(AddPrefixToNodeName("s-2", kConstantFoldingConst),
                node.input(0));
    }
    if (node.name() == "s") {
      ++found;
      EXPECT_EQ("ShapeN", node.op());
      EXPECT_EQ("v1", node.input(0));
      EXPECT_EQ("v2", node.input(1));
      EXPECT_EQ("v3", node.input(2));
    }
    if (node.name() == AddPrefixToNodeName("s-2", kConstantFoldingConst)) {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ("^s", node.input(0));
      Tensor value;
      CHECK(value.FromProto(node.attr().at("value").tensor()));
      EXPECT_EQ(4, value.flat<int>()(0));
      EXPECT_EQ(6, value.flat<int>()(1));
    }
  }
  EXPECT_EQ(9, found);
}

TEST_F(ConstantFoldingTest, SwitchNodesEmptyFetch) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  ops::Variable v_in(scope.WithOpName("v_in"), {3}, DT_FLOAT);
  ops::Variable v_ctrl(scope.WithOpName("v_ctrl"), {}, DT_BOOL);
  ops::Switch s1(scope.WithOpName("switch"), v_in, v_ctrl);
  ops::Rank rank(scope.WithOpName("rank"), s1.output_false);
  ops::Identity i(scope.WithOpName("i"), s1.output_true);
  ops::Size size(scope.WithOpName("size"), i);
  ops::Square p1(scope.WithOpName("p1"), rank);
  ops::Square p2(scope.WithOpName("p2"), size);
  ops::Merge m(scope.WithOpName("m"), {p1.y, p2.y});

  Output predicate =
      ops::Const(scope.WithOpName("false"), false, TensorShape({}));
  Output constant =
      ops::Const(scope.WithOpName("constant"), 1.0f, TensorShape({1}));
  ops::Switch s2(scope.WithOpName("switch2"), constant, predicate);
  ops::Identity statically_known(scope.WithOpName("i2"), s2.output_false);
  ops::Identity never_generated(scope.WithOpName("i3"), s2.output_true);
  ops::Merge m2(scope.WithOpName("m2"),
                {statically_known.output, never_generated.output});

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::set<string> present_nodes = {"v_in",     "v_ctrl",
                                    "switch",   "i",
                                    "p1",       "p2",
                                    "m",        "false",
                                    "constant", "switch2",
                                    "i2",       "i3",
                                    "m2",       "ConstantFoldingCtrl/switch_0",
                                    "rank",     "size"};
  std::set<string> not_present_nodes = {"ConstantFolding/switch2-0"};
  EXPECT_EQ(present_nodes.size(), output.node_size());
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_TRUE(present_nodes.find(node.name()) != present_nodes.end());
    EXPECT_TRUE(not_present_nodes.find(node.name()) == not_present_nodes.end());
    present_nodes.erase(node.name());
    not_present_nodes.erase(node.name());
    if (node.name() == "rank") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFoldingCtrl/switch_0", node.input(0));
    }
    if (node.name() == "size") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^i", node.input(0));
    }
    if (node.name() == "i2") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    }
    if (node.name() == "i3") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("switch2:1", node.input(0));
    }
  }
  EXPECT_EQ(4, found);
}

TEST_F(ConstantFoldingTest, SwitchNodes) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  ops::Variable v_in(scope.WithOpName("v_in"), {3}, DT_FLOAT);
  ops::Variable v_ctrl(scope.WithOpName("v_ctrl"), {}, DT_BOOL);
  ops::Switch s1(scope.WithOpName("switch"), v_in, v_ctrl);
  ops::Rank rank(scope.WithOpName("rank"), s1.output_false);
  ops::Identity i(scope.WithOpName("i"), s1.output_true);
  ops::Size size(scope.WithOpName("size"), i);
  ops::Square p1(scope.WithOpName("p1"), rank);
  ops::Square p2(scope.WithOpName("p2"), size);
  ops::Merge m(scope.WithOpName("m"), {p1.y, p2.y});

  Output predicate =
      ops::Const(scope.WithOpName("false"), false, TensorShape({}));
  Output constant =
      ops::Const(scope.WithOpName("constant"), 1.0f, TensorShape({1}));
  ops::Switch s2(scope.WithOpName("switch2"), constant, predicate);
  ops::Identity statically_known(scope.WithOpName("i2"), s2.output_false);
  ops::Identity never_generated(scope.WithOpName("i3"), s2.output_true);
  ops::Merge m2(scope.WithOpName("m2"),
                {statically_known.output, never_generated.output});

  GrapplerItem item;
  item.fetch.push_back("m");
  item.fetch.push_back("m2");

  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  std::set<string> present_nodes = {"v_in",     "v_ctrl",
                                    "switch",   "i",
                                    "p1",       "p2",
                                    "m",        "false",
                                    "constant", "switch2",
                                    "i2",       "i3",
                                    "m2",       "ConstantFoldingCtrl/switch_0"};
  std::set<string> not_present_nodes = {"rank", "size",
                                        "ConstantFolding/switch2-0"};
  EXPECT_EQ(present_nodes.size(), output.node_size());

  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_TRUE(present_nodes.find(node.name()) != present_nodes.end());
    EXPECT_TRUE(not_present_nodes.find(node.name()) == not_present_nodes.end());
    present_nodes.erase(node.name());
    not_present_nodes.erase(node.name());
    if (node.name() == "i2") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    }
    if (node.name() == "i3") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("switch2:1", node.input(0));
    }
  }
  EXPECT_EQ(2, found);
}

TEST_F(ConstantFoldingTest, MergeNodes) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output x =
      ops::RandomNormal(scope.WithOpName("x"), {3, 5}, DataType::DT_FLOAT);
  Output y =
      ops::RandomNormal(scope.WithOpName("y"), {3, 5}, DataType::DT_FLOAT);
  Output const1 =
      ops::Const(scope.WithOpName("const1").WithControlDependencies(x), 2.7f,
                 TensorShape({3, 5}));
  Output const2 =
      ops::Const(scope.WithOpName("const2"), 3.14f, TensorShape({3, 5}));
  Output const3 =
      ops::Const(scope.WithOpName("const3").WithControlDependencies(x), 3.14f,
                 TensorShape({3, 5}));

  // Create 3 merge nodes: m1 is foldable, m2 and m3 aren't.
  ops::Merge m1(scope.WithOpName("m1"), {x, const1, const2});
  ops::Merge m2(scope.WithOpName("m2"), {const1, const3});
  ops::Merge m3(scope.WithOpName("m3"), {x, y});

  ops::Identity out1(scope.WithOpName("out1"), m1.output);
  ops::Identity idx1(scope.WithOpName("idx1"), m1.value_index);
  ops::Identity out2(scope.WithOpName("out2"), m2.output);
  ops::Identity idx2(scope.WithOpName("idx2"), m2.value_index);
  ops::Identity out3(scope.WithOpName("out3"), m3.output);
  ops::Identity idx3(scope.WithOpName("idx3"), m3.value_index);

  GrapplerItem item;
  item.fetch = {"out1", "idx1", "out2", "idx2", "out3", "idx3"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found_nodes = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "out1") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "idx1") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "ConstantFolding/m1") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "ConstantFolding/m1_index") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^m1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "out2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m2", node.input(0));
      ++found_nodes;
    } else if (node.name() == "idx2") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m2:1", node.input(0));
      ++found_nodes;
    } else if (node.name() == "out3") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m3", node.input(0));
      ++found_nodes;
    } else if (node.name() == "idx3") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m3:1", node.input(0));
      ++found_nodes;
    }
  }
  // Make sure the graph contains all the nodes we're expecting.
  EXPECT_EQ(6, found_nodes);

  std::vector<string> fetch = {"out1", "idx1"};
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(2, tensors.size());
  const Tensor& out_value = tensors[0];
  EXPECT_EQ(3 * 5, out_value.NumElements());
  for (int i = 0; i < 3 * 5; ++i) {
    EXPECT_EQ(3.14f, out_value.flat<float>()(i));
  }
  const Tensor& out_idx = tensors[1];
  EXPECT_EQ(1, out_idx.NumElements());
  EXPECT_EQ(2, out_idx.flat<int32>()(0));
}

TEST_F(ConstantFoldingTest, NoOpReduction) {
  // Build a simple graph with a reduction that can be reduced to the identity.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output v = ops::Variable(scope.WithOpName("v"), {3, 5, 7}, DT_FLOAT);
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(v), 0, {0});
  Output i = ops::Identity(scope.WithOpName("i"), c);
  Output p = ops::Prod(scope.WithOpName("p"), v, i);
  Output s = ops::Square(scope.WithOpName("s"), p);

  GrapplerItem item;
  item.fetch.push_back("s");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  bool found = false;
  for (const auto& node : output.node()) {
    if (node.name() == "p") {
      found = true;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("v", node.input(0));
      EXPECT_EQ("^i", node.input(1));
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(ConstantFoldingTest, NoOpReshape) {
  // Build a simple graph with a reshape that can be reduced to the identity.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  // A reshape than can be optimized
  Output d1 = ops::Const(scope.WithOpName("d1"), 3.14f, {17});
  Output v1 = ops::Variable(scope.WithOpName("v1"), {17}, DT_FLOAT);
  Output c1 =
      ops::Const(scope.WithOpName("c1").WithControlDependencies(v1), 17, {1});
  Output i1 = ops::Identity(scope.WithOpName("i1"), c1);
  Output r1 =
      ops::Reshape(scope.WithOpName("r1").WithControlDependencies(d1), v1, i1);
  Output s1 = ops::Square(scope.WithOpName("s1"), r1);

  // A multi dimensional reshape than can be optimized
  Output v3 = ops::Variable(scope.WithOpName("v3"), {5, 5, 5}, DT_FLOAT);
  Output c3 =
      ops::Const(scope.WithOpName("c3").WithControlDependencies(v3), 5, {3});
  Output i3 = ops::Identity(scope.WithOpName("i3"), c3);
  Output r3 = ops::Reshape(scope.WithOpName("r3"), v3, i3);
  Output s3 = ops::Square(scope.WithOpName("s3"), r3);

  // A multi dimensional partially defined reshape than can be optimized
  Output v4 = ops::Variable(scope.WithOpName("v4"), {5, 5, 5}, DT_FLOAT);
  Output c4 = ops::Const(scope.WithOpName("c4").WithControlDependencies(v4),
                         {5, -1, 5}, {3});
  Output i4 = ops::Identity(scope.WithOpName("i4"), c4);
  Output r4 = ops::Reshape(scope.WithOpName("r4"), v4, i4);
  Output s4 = ops::Square(scope.WithOpName("s4"), r4);

  // A reshape that can't be optimized
  Output v2 = ops::Variable(scope.WithOpName("v2"), {17, 1}, DT_FLOAT);
  Output c2 =
      ops::Const(scope.WithOpName("c2").WithControlDependencies(v2), 17, {1});
  Output r2 = ops::Reshape(scope.WithOpName("r2"), v2, c2);
  Output s2 = ops::Square(scope.WithOpName("s2"), r2);

  GrapplerItem item;
  item.fetch = {"s1", "s2", "s3", "s4"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "r1") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      ASSERT_EQ(3, node.input_size());
      EXPECT_EQ("v1", node.input(0));
      EXPECT_EQ("^i1", node.input(1));
      EXPECT_EQ("^d1", node.input(2));
    } else if (node.name() == "r3") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("v3", node.input(0));
      EXPECT_EQ("^i3", node.input(1));
    } else if (node.name() == "r4") {
      ++found;
      EXPECT_EQ("Identity", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("v4", node.input(0));
      EXPECT_EQ("^i4", node.input(1));
    } else if (node.name() == "r2") {
      ++found;
      EXPECT_EQ("Reshape", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("v2", node.input(0));
      EXPECT_EQ("c2", node.input(1));
    }
  }
  EXPECT_EQ(4, found);
}

TEST_F(ConstantFoldingTest, Packing) {
  // Build a simple graph with a large constant that can be folded.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(scope.WithOpName("c"), 3.14f, {1000});
  Output i1 = ops::Identity(scope.WithOpName("i1"), c);
  Output i2 = ops::Identity(scope.WithOpName("i2"), c);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Make sure that the representation of the folded constant is space
  // efficient: in particular, the whole message should be smaller than 8k (the
  // size needed to naively encode 1000 floats folded twice).
  EXPECT_GT(8000, output.ByteSizeLong());
}

TEST_F(ConstantFoldingTest, MaterializeBroadcastGradientArgs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a =
      ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output b = ops::Square(s.WithOpName("b"), a);
  Output c = ops::Mul(s.WithOpName("c"), a, b);
  Output d = ops::Shape(s.WithOpName("d"), a);
  Output e = ops::Shape(s.WithOpName("e"), b);

  auto f = ops::internal::BroadcastGradientArgs(s.WithOpName("f"), d, e);
  Output o1 = ops::Identity(s.WithOpName("o1"), f.r0);
  Output o2 = ops::Identity(s.WithOpName("o2"), f.r1);

  Output g = ops::Placeholder(s.WithOpName("g"), DT_FLOAT,
                              ops::Placeholder::Shape(PartialTensorShape({1})));
  Output h = ops::Shape(s.WithOpName("h"), g);
  auto i = ops::internal::BroadcastGradientArgs(s.WithOpName("i"), d, h);
  Output p1 = ops::Identity(s.WithOpName("p1"), i.r0);
  Output p2 = ops::Identity(s.WithOpName("p2"), i.r1);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(RewriterConfig::AGGRESSIVE, nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "o1") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-0", node.input(0));
    } else if (node.name() == "o2") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-1", node.input(0));
    } else if (node.name() == "ConstantFolding/f-0") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^f", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "ConstantFolding/f-1") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^f", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "p1") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/i-0", node.input(0));
    } else if (node.name() == "p2") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("i:1", node.input(0));
    } else if (node.name() == "ConstantFolding/i-0") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^i", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    }
  }
  EXPECT_EQ(7, found);
}

TEST_F(ConstantFoldingTest, MaterializeReductionIndices) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output input =
      ops::Placeholder(s.WithOpName("input"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
  Output indices = ops::Placeholder(s.WithOpName("indices"), DT_INT32);
  Output sum = ops::Sum(s.WithOpName("sum"), input, indices);
  Output size = ops::Const(s.WithOpName("size"), 1, {1});
  Output reshape = ops::Reshape(s.WithOpName("reshape"), sum, size);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch.push_back("reshape");

  ConstantFolding fold(RewriterConfig::AGGRESSIVE, nullptr /* cpu_device */);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "ConstantFolding/sum-reduction_indices") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ("^indices", node.input(0));
      EXPECT_EQ(2, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "sum") {
      ++found;
      EXPECT_EQ("ConstantFolding/sum-reduction_indices", node.input(1));
    } else if (node.name() == "indices") {
      ++found;
    }
  }
  EXPECT_EQ(3, found);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

//  LocalWords:  NewRootScope
