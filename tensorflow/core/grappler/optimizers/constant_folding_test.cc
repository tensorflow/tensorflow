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
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace grappler {
namespace {

class ConstantFoldingTest : public GrapplerTest {
 protected:
  template <DataType DTYPE>
  void SimpleNeutralElementTest() {
    for (bool use_snapshot : {false, true}) {
      typedef typename EnumToDataType<DTYPE>::Type T;
      tensorflow::Scope s = tensorflow::Scope::NewRootScope();
      Output x = ops::Placeholder(s.WithOpName("x"), DTYPE,
                                  ops::Placeholder::Shape(TensorShape({2, 2})));
      Output v = ops::Variable(s.WithOpName("v"), {2, 2}, DTYPE);
      Tensor zeros_t(DTYPE, TensorShape({2, 2}));
      Tensor ones_t(DTYPE, TensorShape({2, 2}));
      Tensor x_t(DTYPE, TensorShape({2, 2}));
      for (int i = 0; i < 4; ++i) {
        zeros_t.flat<T>()(i) = T(0);
        ones_t.flat<T>()(i) = T(1);
        x_t.flat<T>()(i) = T(i + 1);
      }
      Output zeros = ops::Const(s.WithOpName("zeros"), zeros_t);
      Output ones = ops::Const(s.WithOpName("ones"), ones_t);
      Output mul1;
      Output mul2;
      Output add1;
      Output add2;
      if (DTYPE == DT_BOOL) {
        mul1 = ops::LogicalAnd(s.WithOpName("mul1"), x, zeros);
        mul2 = ops::LogicalAnd(s.WithOpName("mul2"), x, ones);
        add1 = ops::LogicalOr(s.WithOpName("add1"), x, zeros);
        add2 = ops::LogicalOr(s.WithOpName("add2"), x, ones);
      } else {
        mul1 = ops::Mul(s.WithOpName("mul1"), x, zeros);
        mul2 = ops::Mul(s.WithOpName("mul2"), x, ones);
        add1 = ops::Add(s.WithOpName("add1"), x, zeros);
        add1 = ops::Add(s.WithOpName("add2"), x, ones);
      }
      if (use_snapshot) {
        // Add an op with ref input to prevent Snapshot from being
        // turned into Identity.
        ops::Assign(s.WithOpName("assign"), v, ones);
      }
      GrapplerItem item;
      TF_CHECK_OK(s.ToGraphDef(&item.graph));
      item.fetch = {"mul1", "mul2", "add1", "add2"};
      ConstantFolding optimizer(nullptr /* cpu_device */);
      GraphDef output;
      Status status = optimizer.Optimize(nullptr, item, &output);
      TF_EXPECT_OK(status);

      EXPECT_EQ(7, output.node_size());
      const string snapshot_or_identity =
          use_snapshot ? "Snapshot" : "Identity";
      for (int i = 0; i < output.node_size(); ++i) {
        const NodeDef& node = output.node(i);
        const string& name = node.name();
        if (name == "mul1") {
          EXPECT_EQ("Const", node.op());
          EXPECT_EQ("^x", node.input(0));
          EXPECT_EQ("^zeros", node.input(1));
        } else if (name == "mul2") {
          EXPECT_EQ(snapshot_or_identity, node.op());
          EXPECT_EQ("x", node.input(0));
          EXPECT_EQ("^ones", node.input(1));
        } else if (name == "add1") {
          EXPECT_EQ(snapshot_or_identity, node.op());
          EXPECT_EQ("x", node.input(0));
          EXPECT_EQ("^zeros", node.input(1));
        } else if (name == "add2") {
          if (DTYPE == DT_BOOL) {
            EXPECT_EQ("Const", node.op());
            EXPECT_EQ("^x", node.input(0));
            EXPECT_EQ("^ones", node.input(1));
          } else {
            EXPECT_EQ("Add", node.op());
            EXPECT_EQ("x", node.input(0));
            EXPECT_EQ("ones", node.input(1));
          }
        }
      }
      auto tensors_expected =
          EvaluateNodes(item.graph, item.fetch, {{"x", x_t}});
      auto tensors = EvaluateNodes(output, item.fetch, {{"x", x_t}});
      EXPECT_EQ(4, tensors_expected.size());
      EXPECT_EQ(4, tensors.size());
      for (int i = 0; i < item.fetch.size(); ++i) {
        test::ExpectTensorEqual<T>(tensors_expected[i], tensors[i]);
      }
    }
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  Output c2 = ops::Const(s.WithOpName("c2"), 2.0f, {2});
  Output c3 = ops::Const(s.WithOpName("c3"), 3.0f, {2});
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output add_child = ops::Add(s.WithOpName("add_child"), c2, x);
  Output c1 = ops::Const(s.WithOpName("c1").WithControlDependencies(add_child),
                         1.0f, {1});
  Output add_parent = ops::Add(s.WithOpName("add_parent"), c1, add_child);

  Output y = ops::Placeholder(s.WithOpName("y"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output c4 = ops::Const(s.WithOpName("c4"), 4.0f, {2});
  Output c5 = ops::Const(s.WithOpName("c5"), 5.0f, {2});
  Output c20 = ops::Const(s.WithOpName("c20"), 20.0f, {2});
  Output mul_child = ops::Mul(s.WithOpName("mul_child"), c4, y);
  Output mul_parent = ops::Mul(s.WithOpName("mul_parent"), c5, mul_child);
  Output addmul_child = ops::Add(s.WithOpName("addmul_child"), c4, x);
  Output addmul_parent =
      ops::Mul(s.WithOpName("addmul_parent"), c5, addmul_child);

  GrapplerItem item;
  item.fetch = {"add_parent", "mul_parent", "addmul_parent"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // We expect the following rewrite(s) to occur:
  //
  //    +                +             +
  //   / \              / \           / \
  // 1.0  +     -->    x   +    -->  x  3.0
  //     / \              / \
  //   2.0  x           1.0 2.0
  //
  //    *                *             *
  //   / \              / \           / \
  // 4.0  *     -->    y   *    -->  y  20.0
  //     / \              / \
  //   5.0  y           4.0 5.0

  EXPECT_EQ(11, output.node_size());
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
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("mul_child", node.input(1));
    } else if (node.name() == "addmul_child") {
      // Unchanged.
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("c4", node.input(0));
      EXPECT_EQ("x", node.input(1));
    }
  }

  // Check that the result nodes have the expected value.
  std::vector<string> fetch = {"c3", "c20"};
  auto tensor_expected = EvaluateNodes(item.graph, fetch);
  EXPECT_EQ(fetch.size(), tensor_expected.size());
  fetch = {"add_child", "mul_child"};
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<float>(tensor_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, ConvPushDownTest) {
  // Tests if the following rewrite is performed:
  //
  //         *                       Conv2D
  //        / \                       / \
  //       c  Conv2D        -->      x  (c * filter)
  //           / \
  //          x  filter
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  int input_depth = 3;
  int filter_count = 5;
  int filter_size = 2;
  TensorShape filter_shape(
      {filter_size, filter_size, input_depth, filter_count});
  Tensor filter_values(DT_FLOAT, filter_shape);
  for (int i = 0; i < filter_values.NumElements(); ++i) {
    filter_values.flat<float>()(i) = std::sqrt(static_cast<float>(i));
  }
  Output filter =
      ops::Const(s.WithOpName("filter"), Input::Initializer(filter_values));

  int batch_size = 4;
  int input_dim = 10;
  TensorShape input_shape({batch_size, input_dim, input_dim, input_depth});
  Output input = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                                  ops::Placeholder::Shape(input_shape));

  Output conv =
      ops::Conv2D(s.WithOpName("conv"), input, filter, {1, 1, 1, 1}, "VALID");
  Output c = ops::Const(s.WithOpName("c"), 3.0f, {1});
  Output mul = ops::Mul(s.WithOpName("mul"), c, conv);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding fold(nullptr);
  GraphDef output;
  Status status = fold.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::cout << output.DebugString() << std::endl;

  EXPECT_EQ(5, output.node_size());
  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "mul") {
      found++;
      EXPECT_EQ("Conv2D", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("conv/merged_input", node.input(1));
    } else if (node.name() == "conv/merged_input") {
      found++;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    }
  }
  EXPECT_EQ(2, found);

  // Check that const folded multiplication node has the expected value.
  std::vector<string> fetch = {"mul"};
  Tensor value(DT_FLOAT, input_shape);
  for (int i = 0; i < value.NumElements(); ++i) {
    value.flat<float>()(i) = i;
  }
  auto actual = EvaluateNodes(output, fetch, {{"x", value}});
  auto expected = EvaluateNodes(item.graph, fetch, {{"x", value}});
  test::ExpectTensorEqual<float>(expected[0], actual[0]);
}

TEST_F(ConstantFoldingTest, NeutralElement) {
  int kConst = 0;
  int kLike = 1;
  int kFill = 2;
  for (int const_type : {kConst, kLike, kFill}) {
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
    Output zeros_1d = ops::Const(s.WithOpName("zeros_1d"), 0.0f, {2});
    Output zeros_const = ops::Const(s.WithOpName("zeros_const"), 0.0f, {2, 2});
    Output zeros_like = ops::ZerosLike(s.WithOpName("zeros_like"), x);
    Output zeros_fill = ops::Fill(s.WithOpName("zeros_fill"), {2, 2}, 0.0f);
    Output zeros = const_type == kConst
                       ? zeros_const
                       : (const_type == kLike ? zeros_like : zeros_fill);
    Output ones_const = ops::Const(s.WithOpName("ones_const"), 1.0f, {2, 2});
    Output ones_like = ops::OnesLike(s.WithOpName("ones_like"), x);
    Output ones_fill = ops::Fill(s.WithOpName("ones_fill"), {2, 2}, 1.0f);
    Output ones = const_type == kConst
                      ? ones_const
                      : (const_type == kLike ? ones_like : ones_fill);
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
    Output concat =
        ops::Stack(s.WithOpName("stack"),
                   {mul1, mul2, mul3, mul4, mul5, mul6, div1, div2, matmul1,
                    matmul2, add1, add2, bias_add1, bias_add2, sub1, sub2});
    GrapplerItem item;
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    item.fetch = {"stack", "matmul3", "matmul4"};

    ConstantFolding optimizer(nullptr /* cpu_device */);
    GraphDef output;
    Status status = optimizer.Optimize(nullptr, item, &output);
    TF_EXPECT_OK(status);

    const string suffix =
        (const_type == kConst ? "_const"
                              : (const_type == kLike ? "_like" : "_fill"));
    const string zeros_name = strings::StrCat("zeros", suffix);
    const string ones_name = strings::StrCat("ones", suffix);
    const string ctrl_zeros_name = strings::StrCat("^zeros", suffix);
    const string ctrl_ones_name = strings::StrCat("^ones", suffix);
    EXPECT_EQ(27, output.node_size());
    for (int i = 0; i < output.node_size(); ++i) {
      const NodeDef& node = output.node(i);
      const string& name = node.name();
      if (name == "mul1") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^x", node.input(0));
        EXPECT_EQ(ctrl_zeros_name, node.input(1));
      } else if (name == "mul2") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ(ctrl_zeros_name, node.input(0));
        EXPECT_EQ("^y", node.input(1));
      } else if (name == "mul3") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ(ctrl_ones_name, node.input(1));
      } else if (name == "mul4") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ(ctrl_ones_name, node.input(1));
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
        EXPECT_EQ(ctrl_ones_name, node.input(1));
      } else if (name == "div2") {
        EXPECT_EQ("Reciprocal", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ(ctrl_ones_name, node.input(1));
      } else if (name == "matmul1") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^x", node.input(0));
        EXPECT_EQ(ctrl_zeros_name, node.input(1));
      } else if (name == "matmul2") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ(ctrl_zeros_name, node.input(0));
        EXPECT_EQ("^y", node.input(1));
      } else if (name == "matmul3") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^a", node.input(0));
        EXPECT_EQ(ctrl_zeros_name, node.input(1));
        TensorProto t = node.attr().at("value").tensor();
        EXPECT_EQ(1, t.float_val_size());
        EXPECT_EQ(0, t.float_val(0));
        EXPECT_EQ(2, t.tensor_shape().dim_size());
        EXPECT_EQ(3, t.tensor_shape().dim(0).size());
        EXPECT_EQ(2, t.tensor_shape().dim(1).size());
      } else if (name == "matmul4") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ(ctrl_zeros_name, node.input(0));
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
        EXPECT_EQ(ctrl_zeros_name, node.input(1));
      } else if (name == "add2") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ(ctrl_zeros_name, node.input(1));
      } else if (name == "bias_add1") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^zeros_1d", node.input(1));
      } else if (name == "bias_add2") {
        // We don't eliminate this one, because it requires broadcasting.
        EXPECT_EQ("BiasAdd", node.op());
        EXPECT_EQ(zeros_name, node.input(0));
        EXPECT_EQ("bias", node.input(1));
      } else if (name == "sub1") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ(ctrl_zeros_name, node.input(1));
      } else if (name == "sub2") {
        EXPECT_EQ("Neg", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ(ctrl_zeros_name, node.input(1));
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
    auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 2}));
    auto b_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 3}));
    auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
    auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
    auto bias_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2}));

    auto tensors_expected = EvaluateNodes(
        item.graph, item.fetch,
        {{"x", x_t}, {"y", y_t}, {"a", a_t}, {"b", b_t}, {"bias", bias_t}});
    EXPECT_EQ(item.fetch.size(), tensors_expected.size());
    auto tensors = EvaluateNodes(
        output, item.fetch,
        {{"x", x_t}, {"y", y_t}, {"a", a_t}, {"b", b_t}, {"bias", bias_t}});
    EXPECT_EQ(item.fetch.size(), tensors.size());
    for (int i = 0; i < item.fetch.size(); ++i) {
      test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-6);
    }
  }
}

TEST_F(ConstantFoldingTest, NeutralElement_ShortFloats) {
  SimpleNeutralElementTest<DT_BOOL>();
  SimpleNeutralElementTest<DT_HALF>();
  SimpleNeutralElementTest<DT_BFLOAT16>();
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
  ConstantFolding optimizer(nullptr /* cpu_device */);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

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

  const std::vector<string> fetch = {"mul_0", "mul_4", "mul_8"};
  auto x_known_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto x_partially_unknown_t =
      GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 4}));
  auto x_unknown_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({5, 7}));
  auto expected_tensors =
      EvaluateNodes(item.graph, fetch,
                    {{"x_known", x_known_t},
                     {"x_partially_unknown", x_partially_unknown_t},
                     {"x_unknown", x_unknown_t}});
  EXPECT_EQ(fetch.size(), expected_tensors.size());
  auto tensors = EvaluateNodes(output, fetch,
                               {{"x_known", x_known_t},
                                {"x_partially_unknown", x_partially_unknown_t},
                                {"x_unknown", x_unknown_t}});
  EXPECT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < tensors.size(); i++)
    test::ExpectTensorNear<float>(expected_tensors[i], tensors[i], 1e-5);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

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
  const std::vector<string> fetch = {"addn1"};
  auto x_partially_unknown_t =
      GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto x_unknown_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto expected_tensors =
      EvaluateNodes(item.graph, fetch,
                    {{"x_partially_unknown", x_partially_unknown_t},
                     {"x_unknown", x_unknown_t}});
  EXPECT_EQ(1, expected_tensors.size());
  auto tensors = EvaluateNodes(output, fetch,
                               {{"x_partially_unknown", x_partially_unknown_t},
                                {"x_unknown", x_unknown_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(expected_tensors[0], tensors[0], 1e-5);
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
  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());
  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
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
  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
  auto v1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3}));
  auto v2_t = GenerateRandomTensor<DT_FLOAT>({5, 7});
  auto v3_t = GenerateRandomTensor<DT_FLOAT>({11, 13});

  auto tensors_expected = EvaluateNodes(
      item.graph, item.fetch, {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}});
  EXPECT_EQ(1, item.fetch.size());
  auto tensors = EvaluateNodes(output, item.fetch,
                               {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}});
  EXPECT_EQ(1, item.fetch.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  auto v1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3}));
  auto v2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({5, 7}));
  auto v3_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({11, 13}));
  std::vector<string> fetch_nodes = {"p2"};
  auto tensors_expected = EvaluateNodes(
      item.graph, fetch_nodes, {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}});
  EXPECT_EQ(1, tensors_expected.size());
  auto tensors = EvaluateNodes(output, fetch_nodes,
                               {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_NE(AddPrefixToNodeName("s-matshapes-0", kConstantFoldingConst),
              node.name());
    EXPECT_NE(AddPrefixToNodeName("s-matshapes-1", kConstantFoldingConst),
              node.name());
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
      EXPECT_EQ(AddPrefixToNodeName("s-matshapes-2", kConstantFoldingConst),
                node.input(0));
    }
    if (node.name() == "s") {
      ++found;
      EXPECT_EQ("ShapeN", node.op());
      EXPECT_EQ("v1", node.input(0));
      EXPECT_EQ("v2", node.input(1));
      EXPECT_EQ("v3", node.input(2));
    }
    if (node.name() ==
        AddPrefixToNodeName("s-matshapes-2", kConstantFoldingConst)) {
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

  auto v1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 4}));
  auto v2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({5, 6}));
  auto v3_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4, 6}));
  const std::vector<string> fetch_nodes = {"i1a", "i1b", "i2a", "i2b",
                                           "i2c", "i3a", "i3b"};
  auto tensors_expected = EvaluateNodes(
      item.graph, fetch_nodes, {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}});
  EXPECT_EQ(fetch_nodes.size(), tensors_expected.size());
  auto tensors = EvaluateNodes(output, fetch_nodes,
                               {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}});
  EXPECT_EQ(fetch_nodes.size(), tensors.size());
  for (int i = 0; i < fetch_nodes.size(); i++)
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
}

TEST_F(ConstantFoldingTest, ShapeMaterializationShapeN_MultipleOutputs) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output v1 = ops::Variable(scope.WithOpName("v1"), {3, -1}, DT_FLOAT);
  Output v2 = ops::Variable(scope.WithOpName("v2"), {4, 6}, DT_FLOAT);
  auto s = ops::ShapeN(scope.WithOpName("s"), {v1, v2});
  auto id_n = ops::IdentityN(scope.WithOpName("id_n"), {s[0], s[1]});
  Output ia = ops::Identity(scope.WithOpName("ia"), id_n[0]);
  Output ib = ops::Identity(scope.WithOpName("ib"), id_n[1]);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("ia");
  item.fetch.push_back("ib");

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_NE(AddPrefixToNodeName("s-matshapes-0", kConstantFoldingConst),
              node.name());
    if (node.name() == "s") {
      ++found;
      EXPECT_EQ("ShapeN", node.op());
      EXPECT_EQ("v1", node.input(0));
      EXPECT_EQ("v2", node.input(1));
    }
    if (node.name() == "id_n") {
      ++found;
      EXPECT_EQ("IdentityN", node.op());
      EXPECT_EQ("s", node.input(0));
      EXPECT_EQ(AddPrefixToNodeName("s-matshapes-1", kConstantFoldingConst),
                node.input(1));
    }
    if (node.name() == "ia") {
      ++found;
      EXPECT_EQ("id_n", node.input(0));
    }
    if (node.name() == "ib") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ("^s", node.input(0));
      EXPECT_EQ("^id_n", node.input(1));
    }
  }
  EXPECT_EQ(4, found);

  auto v1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 4}));
  auto v2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4, 6}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"v1", v1_t}, {"v2", v2_t}});
  EXPECT_EQ(2, tensors_expected.size());
  auto tensors =
      EvaluateNodes(output, item.fetch, {{"v1", v1_t}, {"v2", v2_t}});
  EXPECT_EQ(2, tensors.size());
  for (int i = 0; i < tensors.size(); i++)
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
    EXPECT_TRUE(present_nodes.find(node.name()) != present_nodes.end())
        << node.name();
    EXPECT_TRUE(not_present_nodes.find(node.name()) == not_present_nodes.end())
        << node.name();
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

  auto v_in_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3}));
  Tensor v_ctrl_t(DT_BOOL, TensorShape({}));

  v_ctrl_t.flat<bool>()(0) = true;
  std::vector<string> fetch_nodes = {"m", "m2"};
  auto tensors_expected = EvaluateNodes(
      item.graph, fetch_nodes, {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors_expected.size());
  auto tensors = EvaluateNodes(output, fetch_nodes,
                               {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
  test::ExpectTensorNear<float>(tensors_expected[1], tensors[1], 1e-5);

  v_ctrl_t.flat<bool>()(0) = false;
  tensors_expected = EvaluateNodes(item.graph, fetch_nodes,
                                   {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors_expected.size());
  tensors = EvaluateNodes(output, fetch_nodes,
                          {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
  test::ExpectTensorNear<float>(tensors_expected[1], tensors[1], 1e-5);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  auto v_in_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3}));
  Tensor v_ctrl_t(DT_BOOL, TensorShape({}));
  v_ctrl_t.flat<bool>()(0) = true;
  auto tensors_expected = EvaluateNodes(
      item.graph, item.fetch, {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors_expected.size());
  auto tensors = EvaluateNodes(output, item.fetch,
                               {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
  test::ExpectTensorNear<float>(tensors_expected[1], tensors[1], 1e-5);

  v_ctrl_t.flat<bool>()(0) = false;
  tensors_expected = EvaluateNodes(item.graph, item.fetch,
                                   {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors_expected.size());
  tensors = EvaluateNodes(output, item.fetch,
                          {{"v_in", v_in_t}, {"v_ctrl", v_ctrl_t}});
  EXPECT_EQ(2, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
  test::ExpectTensorNear<float>(tensors_expected[1], tensors[1], 1e-5);
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
  // m4 is not foldable because the only constant input
  // has a control input, so we cannot know if it will be
  // triggered.
  ops::Merge m4(scope.WithOpName("m4"), {x, const1});

  ops::Identity out1(scope.WithOpName("out1"), m1.output);
  ops::Identity idx1(scope.WithOpName("idx1"), m1.value_index);
  ops::Identity out2(scope.WithOpName("out2"), m2.output);
  ops::Identity idx2(scope.WithOpName("idx2"), m2.value_index);
  ops::Identity out3(scope.WithOpName("out3"), m3.output);
  ops::Identity idx3(scope.WithOpName("idx3"), m3.value_index);
  ops::Identity out4(scope.WithOpName("out4"), m4.output);
  ops::Identity idx4(scope.WithOpName("idx4"), m4.value_index);

  GrapplerItem item;
  item.fetch = {"out1", "idx1", "out2", "idx2", "out3", "idx3", "out4", "idx4"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(19, output.node_size());
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
    } else if (node.name() == "out4") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m4", node.input(0));
      ++found_nodes;
    } else if (node.name() == "idx4") {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("m4:1", node.input(0));
      ++found_nodes;
    }
  }
  // Make sure the graph contains all the nodes we're expecting.
  EXPECT_EQ(8, found_nodes);

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

TEST_F(ConstantFoldingTest, SplitRemoval) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 =
      ops::Variable(scope.WithOpName("in1"), TensorShape({2}), DT_FLOAT);
  Output in2 =
      ops::Variable(scope.WithOpName("in2"), TensorShape({4}), DT_FLOAT);
  auto split_dim = ops::Const(scope.WithOpName("split_dim"), {0}, {});
  ops::Split s1(scope.WithOpName("s1"), split_dim, in1, 1);
  ops::Split s2(scope.WithOpName("s2"), split_dim, in2, 2);

  ops::Add out(scope.WithOpName("out"), s1[0], s2[0]);

  GrapplerItem item;
  item.fetch = {"out"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("split_dim", "Const", {}, {}, &want);
  AddNode("s1", "Identity", {"in1", AsControlDependency("split_dim")}, {},
          &want);
  AddNode("s2", "Split", {"in2", "split_dim"}, {}, &want);
  AddNode("out", "Add", {"s1", "s2"}, {}, &want);

  CompareGraphs(want, got);

  auto in1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2}));
  auto in2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors_expected.size());
  auto tensors =
      EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
}

TEST_F(ConstantFoldingTest, SplitVRemoval) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 =
      ops::Variable(scope.WithOpName("in1"), TensorShape({2}), DT_FLOAT);
  Output in2 =
      ops::Variable(scope.WithOpName("in2"), TensorShape({5}), DT_FLOAT);
  auto split_dim = ops::Const(scope.WithOpName("split_dim"), {0}, {});
  auto size_splits1 = ops::Const(scope.WithOpName("size_splits1"), {2}, {1});
  auto size_splits2 = ops::Const(scope.WithOpName("size_splits2"), {2, 3}, {2});
  ops::SplitV s1(scope.WithOpName("s1"), in1, size_splits1, split_dim, 1);
  ops::SplitV s2(scope.WithOpName("s2"), in2, size_splits2, split_dim, 2);

  ops::Add out(scope.WithOpName("out"), s1[0], s2[0]);

  GrapplerItem item;
  item.fetch = {"out"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("split_dim", "Const", {}, {}, &want);
  AddNode("size_splits1", "Const", {}, {}, &want);
  AddNode("size_splits2", "Const", {}, {}, &want);
  AddNode("s1", "Identity",
          {"in1", AsControlDependency("size_splits1"),
           AsControlDependency("split_dim")},
          {}, &want);
  AddNode("s2", "SplitV", {"in2", "size_splits2", "split_dim"}, {}, &want);
  AddNode("out", "Add", {"s1", "s2"}, {}, &want);

  CompareGraphs(want, got);

  auto in1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2}));
  auto in2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({5}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors_expected.size());
  auto tensors =
      EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
}

TEST_F(ConstantFoldingTest, TransposeOnSize1DimsRemoval) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 = ops::Variable(scope.WithOpName("in1"), TensorShape({1, 2, 4, 1}),
                             DT_FLOAT);
  Output p1 = ops::Const(scope.WithOpName("p1"), {3, 2, 1, 0}, {4});
  Output in2 = ops::Variable(scope.WithOpName("in2"), TensorShape({1, 4, 2, 1}),
                             DT_FLOAT);
  Output p2 = ops::Const(scope.WithOpName("p2"), {3, 1, 2, 0}, {4});
  ops::Transpose t1(scope.WithOpName("t1"), in1, p1);
  ops::Transpose t2(scope.WithOpName("t2").WithControlDependencies({in1}), in2,
                    p2);

  ops::Add out1(scope.WithOpName("out1"), t1, t2);

  GrapplerItem item;
  item.fetch = {"out1"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("p1", "Const", {}, {}, &want);
  AddNode("p2", "Const", {}, {}, &want);
  AddNode("t1", "Transpose", {"in1", "p1"}, {}, &want);
  AddNode("t2", "Identity",
          {"in2", AsControlDependency("in1"), AsControlDependency("p2")}, {},
          &want);
  AddNode("out1", "Add", {"t1", "t2"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, RandomShuffleOnScalarRemoval) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 =
      ops::Variable(scope.WithOpName("in1"), TensorShape({}), DT_FLOAT);
  Output in2 =
      ops::Variable(scope.WithOpName("in2"), TensorShape({}), DT_FLOAT);
  ops::RandomShuffle s1(scope.WithOpName("s1"), in1);
  ops::RandomShuffle s2(scope.WithOpName("s2").WithControlDependencies({in1}),
                        in2);

  ops::Add out1(scope.WithOpName("out1"), s1, s2);
  ops::Identity out2(scope.WithOpName("out2"), s2);

  GrapplerItem item;
  item.fetch = {"out1", "out2"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("s1", "Identity", {"in1"}, {}, &want);
  AddNode("s2", "Identity", {"in2", AsControlDependency("in1")}, {}, &want);
  AddNode("out1", "Add", {"s1", "s2"}, {}, &want);
  AddNode("out2", "Identity", {"s2"}, {}, &want);

  CompareGraphs(want, got);

  auto in1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({}));
  auto in2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(2, tensors_expected.size());
  auto tensors =
      EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(2, tensors.size());
  for (int i = 0; i < tensors.size(); i++)
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-5);
}

TEST_F(ConstantFoldingTest, ReverseOnSize1DimsRemoval) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 = ops::Variable(scope.WithOpName("in1"), TensorShape({1, 2, 4, 1}),
                             DT_FLOAT);
  Output a1 = ops::Const(scope.WithOpName("a1"), {3, 2, 1, 0}, {4});
  Output in2 = ops::Variable(scope.WithOpName("in2"), TensorShape({1, 2, 4, 1}),
                             DT_FLOAT);
  Output a2 = ops::Const(scope.WithOpName("a2"), {0, 3}, {2});
  ops::Reverse r1(scope.WithOpName("r1"), in1, a1);
  ops::Reverse r2(scope.WithOpName("r2").WithControlDependencies({in1}), in2,
                  a2);

  ops::Add out1(scope.WithOpName("out1"), r1, r2);

  GrapplerItem item;
  item.fetch = {"out1"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("a1", "Const", {}, {}, &want);
  AddNode("a2", "Const", {}, {}, &want);
  AddNode("r1", "ReverseV2", {"in1", "a1"}, {}, &want);
  AddNode("r2", "Identity",
          {"in2", AsControlDependency("in1"), AsControlDependency("a2")}, {},
          &want);
  AddNode("out1", "Add", {"r1", "r2"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, SliceWithSameDimensionRemoval) {
  {  // size = {3, 5}
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    auto in1 = ops::Variable(scope.WithOpName("in1"), {3, 5}, DT_FLOAT);
    auto begin = ops::Const(scope.WithOpName("begin"), {0, 0}, {2});
    auto size = ops::Const(scope.WithOpName("size"), {3, 5}, {2});
    Output in2 = ops::Variable(scope.WithOpName("in2"), {4, 6}, DT_FLOAT);
    ops::Slice s1(scope.WithOpName("s1"), in1, begin, size);
    ops::Slice s2(scope.WithOpName("s2"), in2, begin, size);

    ops::Add out(scope.WithOpName("out"), s1, s2);

    GrapplerItem item;
    item.fetch = {"out"};
    TF_CHECK_OK(scope.ToGraphDef(&item.graph));

    ConstantFolding optimizer(nullptr /* cpu_device */);
    GraphDef got;
    Status status = optimizer.Optimize(nullptr, item, &got);
    TF_EXPECT_OK(status);

    GraphDef want;
    AddNode("in1", "VariableV2", {}, {}, &want);
    AddNode("in2", "VariableV2", {}, {}, &want);
    AddNode("begin", "Const", {}, {}, &want);
    AddNode("size", "Const", {}, {}, &want);
    AddNode("s1", "Identity",
            {"in1", AsControlDependency("begin"), AsControlDependency("size")},
            {}, &want);
    AddNode("s2", "Slice", {"in2", "begin", "size"}, {}, &want);
    AddNode("out", "Add", {"s1", "s2"}, {}, &want);

    CompareGraphs(want, got);

    auto in1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5}));
    auto in2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4, 6}));
    auto tensors_expected =
        EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors_expected.size());
    auto tensors =
        EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
  }
  {  // size = {-1, -1}
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    auto in1 =
        ops::Variable(scope.WithOpName("in1"), {3, 5}, DataType::DT_FLOAT);
    auto begin1 = ops::Const(scope.WithOpName("begin1"), {0, 0}, {2});
    auto begin2 = ops::Const(scope.WithOpName("begin2"), {1, 1}, {2});
    auto size = ops::Const(scope.WithOpName("size"), {-1, -1}, {2});
    Output in2 =
        ops::Variable(scope.WithOpName("in2"), {4, 6}, DataType::DT_FLOAT);
    ops::Slice s1(scope.WithOpName("s1"), in1, begin1, size);
    ops::Slice s2(scope.WithOpName("s2"), in2, begin2, size);

    ops::Add out(scope.WithOpName("out"), s1, s2);

    GrapplerItem item;
    item.fetch = {"out"};
    TF_CHECK_OK(scope.ToGraphDef(&item.graph));

    ConstantFolding optimizer(nullptr /* cpu_device */);
    GraphDef got;
    Status status = optimizer.Optimize(nullptr, item, &got);
    TF_EXPECT_OK(status);

    GraphDef want;
    AddNode("in1", "VariableV2", {}, {}, &want);
    AddNode("in2", "VariableV2", {}, {}, &want);
    AddNode("begin1", "Const", {}, {}, &want);
    AddNode("begin2", "Const", {}, {}, &want);
    AddNode("size", "Const", {}, {}, &want);
    AddNode("s1", "Identity",
            {"in1", AsControlDependency("begin1"), AsControlDependency("size")},
            {}, &want);
    AddNode("s2", "Slice", {"in2", "begin2", "size"}, {}, &want);
    AddNode("out", "Add", {"s1", "s2"}, {}, &want);

    CompareGraphs(want, got);

    auto in1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5}));
    auto in2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4, 6}));
    auto tensors_expected =
        EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors_expected.size());
    auto tensors =
        EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
  }
}

TEST_F(ConstantFoldingTest, StridedSliceWithSameDimensionRemoval) {
  {  // no mask
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    auto in1 = ops::Variable(scope.WithOpName("in1"), {3, 5, 2}, DT_FLOAT);
    auto begin = ops::Const(scope.WithOpName("begin"), {0, 0}, {2});
    auto end = ops::Const(scope.WithOpName("end"), {3, 5}, {2});
    auto strides = ops::Const(scope.WithOpName("strides"), {1, 1}, {2});
    Output in2 = ops::Variable(scope.WithOpName("in2"), {4, 6, 2}, DT_FLOAT);
    ops::StridedSlice s1(scope.WithOpName("s1"), in1, begin, end, strides);
    ops::StridedSlice s2(scope.WithOpName("s2"), in2, begin, end, strides);

    ops::Add out(scope.WithOpName("out"), s1, s2);

    GrapplerItem item;
    item.fetch = {"out"};
    TF_CHECK_OK(scope.ToGraphDef(&item.graph));

    ConstantFolding optimizer(nullptr /* cpu_device */);
    GraphDef got;
    Status status = optimizer.Optimize(nullptr, item, &got);
    TF_EXPECT_OK(status);

    GraphDef want;
    AddNode("in1", "VariableV2", {}, {}, &want);
    AddNode("in2", "VariableV2", {}, {}, &want);
    AddNode("begin", "Const", {}, {}, &want);
    AddNode("end", "Const", {}, {}, &want);
    AddNode("strides", "Const", {}, {}, &want);
    AddNode("s1", "Identity",
            {"in1", AsControlDependency("begin"), AsControlDependency("end"),
             AsControlDependency("strides")},
            {}, &want);
    AddNode("s2", "StridedSlice", {"in2", "begin", "end", "strides"}, {},
            &want);
    AddNode("out", "Add", {"s1", "s2"}, {}, &want);

    CompareGraphs(want, got);

    auto in1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5, 2}));
    auto in2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({4, 6, 2}));
    auto tensors_expected =
        EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors_expected.size());
    auto tensors =
        EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
  }
  {  // with begin/end/ellipsis mask
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    // s1 = in1[:, ..., 0:5, 0:6]
    auto in1 =
        ops::Variable(scope.WithOpName("in1"), {2, 3, 4, 5, 6}, DT_FLOAT);
    auto begin1 = ops::Const(scope.WithOpName("begin1"), {0, 0, 0}, {3});
    auto end1 = ops::Const(scope.WithOpName("end1"), {0, 5, 6}, {3});
    auto strides1 = ops::Const(scope.WithOpName("strides1"), {1, 1, 1}, {3});
    ops::StridedSlice s1(
        scope.WithOpName("s1"), in1, begin1, end1, strides1,
        ops::StridedSlice::Attrs().BeginMask(1).EndMask(1).EllipsisMask(2));

    Output in2 =
        ops::Variable(scope.WithOpName("in2"), {5, 8, 5, 6, 9}, DT_FLOAT);
    auto begin2 = ops::Const(scope.WithOpName("begin2"), {0, 0, 0, 0, 0}, {5});
    auto end2 = ops::Const(scope.WithOpName("end2"), {2, 3, 4, 5, 6}, {5});
    auto strides2 =
        ops::Const(scope.WithOpName("strides2"), {1, 1, 1, 1, 1}, {5});
    ops::StridedSlice s2(scope.WithOpName("s2"), in2, begin2, end2, strides2);

    ops::Add out(scope.WithOpName("out"), s1, s2);

    GrapplerItem item;
    item.fetch = {"out"};
    TF_CHECK_OK(scope.ToGraphDef(&item.graph));

    ConstantFolding optimizer(nullptr /* cpu_device */);
    GraphDef got;
    Status status = optimizer.Optimize(nullptr, item, &got);
    TF_EXPECT_OK(status);

    GraphDef want;
    AddNode("in1", "VariableV2", {}, {}, &want);
    AddNode("in2", "VariableV2", {}, {}, &want);
    AddNode("begin1", "Const", {}, {}, &want);
    AddNode("end1", "Const", {}, {}, &want);
    AddNode("strides1", "Const", {}, {}, &want);
    AddNode("s1", "Identity",
            {"in1", AsControlDependency("begin1"), AsControlDependency("end1"),
             AsControlDependency("strides1")},
            {}, &want);
    AddNode("begin2", "Const", {}, {}, &want);
    AddNode("end2", "Const", {}, {}, &want);
    AddNode("strides2", "Const", {}, {}, &want);
    AddNode("s2", "StridedSlice", {"in2", "begin2", "end2", "strides2"}, {},
            &want);
    AddNode("out", "Add", {"s1", "s2"}, {}, &want);

    CompareGraphs(want, got);

    auto in1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 3, 4, 5, 6}));
    auto in2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({5, 8, 5, 6, 9}));
    auto tensors_expected =
        EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors_expected.size());
    auto tensors =
        EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
  }
}

TEST_F(ConstantFoldingTest, TileWithMultipliesBeingOne) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto in1 = ops::Variable(scope.WithOpName("in1"), {4, 6}, DT_FLOAT);
  auto in2 = ops::Variable(scope.WithOpName("in2"), {4, 3}, DT_FLOAT);
  auto multiplies1 = ops::Const(scope.WithOpName("multiplies1"), {1, 1}, {2});
  auto multiplies2 = ops::Const(scope.WithOpName("multiplies2"), {1, 2}, {2});

  ops::Tile t1(scope.WithOpName("t1"), in1, multiplies1);
  ops::Tile t2(scope.WithOpName("t2"), in2, multiplies2);

  ops::Add out(scope.WithOpName("out"), t1, t2);

  GrapplerItem item;
  item.fetch = {"out"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("multiplies1", "Const", {}, {}, &want);
  AddNode("multiplies2", "Const", {}, {}, &want);
  AddNode("t1", "Identity", {"in1", AsControlDependency("multiplies1")}, {},
          &want);
  AddNode("t2", "Tile", {"in2", "multiplies2"}, {}, &want);
  AddNode("out", "Add", {"t1", "t2"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, PaddingWithZeroSize) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto in1 = ops::Variable(scope.WithOpName("in1"), {4, 6}, DT_INT32);
  auto in2 = ops::Variable(scope.WithOpName("in2"), {2, 2}, DT_INT32);
  auto paddings1 =
      ops::Const(scope.WithOpName("paddings1"), {0, 0, 0, 0}, {2, 2});
  auto paddings2 =
      ops::Const(scope.WithOpName("paddings2"), {1, 1, 2, 2}, {2, 2});
  auto c1 = ops::Const(scope.WithOpName("c1"), 1);
  auto c2 = ops::Const(scope.WithOpName("c2"), 1);

  ops::PadV2 p1(scope.WithOpName("p1"), in1, paddings1, c1);
  ops::PadV2 p2(scope.WithOpName("p2"), in2, paddings2, c2);

  ops::Add out(scope.WithOpName("out"), p1, p2);

  GrapplerItem item;
  item.fetch = {"out"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("paddings1", "Const", {}, {}, &want);
  AddNode("paddings2", "Const", {}, {}, &want);
  AddNode("c1", "Const", {}, {}, &want);
  AddNode("c2", "Const", {}, {}, &want);
  AddNode("p1", "Identity",
          {"in1", AsControlDependency("paddings1"), AsControlDependency("c1")},
          {}, &want);
  AddNode("p2", "PadV2", {"in2", "paddings2", "c2"}, {}, &want);
  AddNode("out", "Add", {"p1", "p2"}, {}, &want);

  CompareGraphs(want, got);

  auto in1_t = GenerateRandomTensor<DT_INT32>(TensorShape({4, 6}));
  auto in2_t = GenerateRandomTensor<DT_INT32>(TensorShape({2, 2}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors_expected.size());
  auto tensors =
      EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
}

TEST_F(ConstantFoldingTest, SqueezeWithAllDimesionsGreaterThanOne) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto in1 = ops::Variable(scope.WithOpName("in1"), {2, 3}, DT_INT32);
  auto in2 = ops::Variable(scope.WithOpName("in2"), {1, 2, 3, 1}, DT_INT32);

  ops::Squeeze s1(scope.WithOpName("s1"), in1);
  ops::Squeeze s2(scope.WithOpName("s2"), in2);

  ops::Add out(scope.WithOpName("out"), s1, s2);

  GrapplerItem item;
  item.fetch = {"out"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("s1", "Identity", {"in1"}, {}, &want);
  AddNode("s2", "Squeeze", {"in2"}, {}, &want);
  AddNode("out", "Add", {"s1", "s2"}, {}, &want);

  CompareGraphs(want, got);

  auto in1_t = GenerateRandomTensor<DT_INT32>(TensorShape({2, 3}));
  auto in2_t = GenerateRandomTensor<DT_INT32>(TensorShape({1, 2, 3, 1}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors_expected.size());
  auto tensors =
      EvaluateNodes(got, item.fetch, {{"in1", in1_t}, {"in2", in2_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<int>(tensors_expected[0], tensors[0]);
}

TEST_F(ConstantFoldingTest, NoOpReduction) {
  // Build a simple graph with reductions that can be reduced to the
  // identity.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output v = ops::Variable(scope.WithOpName("v"), {3, 5, 7}, DT_FLOAT);
  Output c =
      ops::Const(scope.WithOpName("c").WithControlDependencies(v), 0, {0});
  Output i = ops::Identity(scope.WithOpName("i"), c);
  Output p = ops::Prod(scope.WithOpName("p"), v, i);
  Output s = ops::Square(scope.WithOpName("s"), p);

  Output v2 = ops::Variable(scope.WithOpName("v2"), {3, 5, 1}, DT_FLOAT);
  Output c2 =
      ops::Const(scope.WithOpName("c2").WithControlDependencies(v), 2, {1});
  ops::Prod::Attrs attr;
  attr = attr.KeepDims(true);
  Output p2 = ops::Prod(scope.WithOpName("p2"), v2, c2, attr);

  GrapplerItem item;
  item.fetch = {"s", "p2"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "p") {
      found++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("v", node.input(0));
      EXPECT_EQ("^i", node.input(1));
    } else if (node.name() == "p2") {
      found++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("v2", node.input(0));
      EXPECT_EQ("^c2", node.input(1));
    }
  }
  EXPECT_EQ(2, found);

  auto v_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5, 7}));
  auto v2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5, 1}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch, {{"v", v_t}, {"v2", v2_t}});
  EXPECT_EQ(2, tensors_expected.size());
  auto tensors = EvaluateNodes(output, item.fetch, {{"v", v_t}, {"v2", v2_t}});
  EXPECT_EQ(2, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
  test::ExpectTensorNear<float>(tensors_expected[1], tensors[1], 1e-5);
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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

  auto v1_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({17}));
  auto v2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({17, 1}));
  auto v3_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({5, 5, 5}));
  auto v4_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({5, 5, 5}));
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch,
                    {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}, {"v4", v4_t}});
  EXPECT_EQ(4, tensors_expected.size());
  auto tensors =
      EvaluateNodes(output, item.fetch,
                    {{"v1", v1_t}, {"v2", v2_t}, {"v3", v3_t}, {"v4", v4_t}});
  EXPECT_EQ(4, tensors.size());
  for (int i = 0; i < tensors.size(); i++)
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-5);
}

TEST_F(ConstantFoldingTest, Packing) {
  // Build a simple graph with a large constant that can be folded.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output c = ops::Const(scope.WithOpName("c"), 3.14f, {1000});
  Output i1 = ops::Identity(scope.WithOpName("i1"), c);
  Output i2 = ops::Identity(scope.WithOpName("i2"), c);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  const std::vector<string> fetch_nodes = {"i1", "i2"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch_nodes);
  EXPECT_EQ(fetch_nodes.size(), tensors_expected.size());
  auto tensors = EvaluateNodes(output, fetch_nodes);
  EXPECT_EQ(fetch_nodes.size(), tensors.size());
  for (int i = 0; i < fetch_nodes.size(); i++)
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-5);

  // Make sure that the representation of the folded constant is space
  // efficient: in particular, the whole message should be smaller than 8k
  // (the size needed to naively encode 1000 floats folded twice).
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

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> fetch_nodes = {"o1", "o2", "p1", "p2"};
  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({1, 5}));
  auto g_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({1}));
  auto tensors_expected =
      EvaluateNodes(item.graph, fetch_nodes, {{"a", a_t}, {"g", g_t}});
  EXPECT_EQ(fetch_nodes.size(), tensors_expected.size());

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "o1") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-bcastargs-0", node.input(0));
    } else if (node.name() == "o2") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-bcastargs-1", node.input(0));
    } else if (node.name() == "ConstantFolding/f-bcastargs-0") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^f", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "ConstantFolding/f-bcastargs-1") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^f", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "p1") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("i", node.input(0));
    } else if (node.name() == "p2") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("i:1", node.input(0));
    }
  }
  EXPECT_EQ(6, found);

  auto tensors = EvaluateNodes(output, fetch_nodes, {{"a", a_t}, {"g", g_t}});
  EXPECT_EQ(fetch_nodes.size(), tensors.size());
  for (int i = 0; i < fetch_nodes.size(); i++)
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
}

TEST_F(ConstantFoldingTest, MaterializeBroadcastGradientArgs_InfiniteLoop) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a =
      ops::Placeholder(s.WithOpName("a"), DT_FLOAT,
                       ops::Placeholder::Shape(PartialTensorShape({2, 2})));
  Output b = ops::Square(s.WithOpName("b"), a);
  Output c = ops::Mul(s.WithOpName("c"), a, b);
  Output d = ops::Shape(s.WithOpName("d"), a);
  Output e = ops::Shape(s.WithOpName("e"), b);

  auto f = ops::internal::BroadcastGradientArgs(s.WithOpName("f"), d, e);
  Output o1 = ops::Identity(s.WithOpName("o1"), f.r0);
  Output o2 = ops::Identity(s.WithOpName("o2"), f.r1);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  std::vector<string> fetch_nodes = {"o1", "o2"};
  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto tensors_expected = EvaluateNodes(item.graph, fetch_nodes, {{"a", a_t}});
  EXPECT_EQ(fetch_nodes.size(), tensors_expected.size());

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(11, output.node_size());
  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "ConstantFolding/f-folded-1") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^a", node.input(0));
      EXPECT_EQ("^b", node.input(1));
    } else if (node.name() == "d") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^a", node.input(0));
    } else if (node.name() == "e") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^b", node.input(0));
    } else if (node.name() == "o1") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-bcastargs-0", node.input(0));
    } else if (node.name() == "o2") {
      ++found;
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("ConstantFolding/f-bcastargs-1", node.input(0));
    } else if (node.name() == "ConstantFolding/f-bcastargs-0") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFolding/f-folded-1", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    } else if (node.name() == "ConstantFolding/f-bcastargs-1") {
      ++found;
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFolding/f-folded-1", node.input(0));
      EXPECT_EQ(0, TensorShape(node.attr().at("value").tensor().tensor_shape())
                       .num_elements());
    }
  }
  EXPECT_EQ(7, found);
  auto tensors = EvaluateNodes(output, fetch_nodes, {{"a", a_t}});
  EXPECT_EQ(fetch_nodes.size(), tensors.size());
  for (int i = 0; i < fetch_nodes.size(); i++)
    test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
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

  auto input_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 4}));
  Tensor indices_t(DT_INT32, TensorShape({2}));
  indices_t.flat<int>()(0) = 0;
  indices_t.flat<int>()(1) = 1;
  auto tensors_expected = EvaluateNodes(
      item.graph, item.fetch, {{"input", input_t}, {"indices", indices_t}});
  EXPECT_EQ(1, tensors_expected.size());

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
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

  auto tensors = EvaluateNodes(output, item.fetch,
                               {{"input", input_t}, {"indices", indices_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
}

TEST_F(ConstantFoldingTest, LargeConstant) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  // Generate a 4k by 4k constant matrix.
  Output mat_diag =
      ops::Const(scope.WithOpName("mat_diag"), 3.14f, TensorShape({1024 * 4}));
  Output mat = ops::Diag(scope.WithOpName("mat"), mat_diag);
  Output out = ops::Identity(scope.WithOpName("out"), mat);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("out");

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Make sure the diag node hasn't been folded, since it would use too much
  // memory to encode the corresponding constant.
  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "out") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("mat", node.input(0));
      ++found;
    } else if (node.name() == "mat") {
      EXPECT_EQ("Diag", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("mat_diag", node.input(0));
      ++found;
    }
  }
  EXPECT_EQ(2, found);

  EXPECT_GT(1024 * 1024, output.ByteSizeLong());

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  EXPECT_EQ(1, tensors_expected.size());
  auto tensors = EvaluateNodes(output, item.fetch);
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(ConstantFoldingTest, SwitchIdenticalInputs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output x = ops::Placeholder(s.WithOpName("x"), DT_BOOL,
                              ops::Placeholder::Shape(TensorShape({})));
  ops::Switch sw = ops::Switch(s.WithOpName("switch"), x, x);
  Output id_false = ops::LogicalNot(s.WithOpName("id_false"), sw.output_false);
  Output id_true = ops::LogicalNot(s.WithOpName("id_true"), sw.output_true);

  GrapplerItem item;
  item.fetch.push_back("id_false");
  item.fetch.push_back("id_true");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(6, output.node_size());
  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "switch" || node.name() == "x") {
      ++found;
    }
    if (node.name() == "id_false") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFoldingCtrl/switch_0", node.input(0));
      ++found;
    }
    if (node.name() == "id_true") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^ConstantFoldingCtrl/switch_1", node.input(0));
      ++found;
    }
    if (node.name() == "ConstantFoldingCtrl/switch_0") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("switch", node.input(0));
      ++found;
    }
    if (node.name() == "ConstantFoldingCtrl/switch_1") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("switch:1", node.input(0));
      ++found;
    }
  }
  EXPECT_EQ(6, found);

  // Evaluate id_true when input tensor x is true.
  Tensor x_t(DT_BOOL, TensorShape({}));
  x_t.flat<bool>()(0) = true;
  auto tensors_expected = EvaluateNodes(item.graph, {"id_true"}, {{"x", x_t}});
  EXPECT_EQ(1, tensors_expected.size());
  auto tensors = EvaluateNodes(output, {"id_true"}, {{"x", x_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<bool>(tensors_expected[0], tensors[0]);

  // Evalute id_false when input tensor is false.
  x_t.flat<bool>()(0) = false;
  tensors_expected = EvaluateNodes(item.graph, {"id_false"}, {{"x", x_t}});
  EXPECT_EQ(1, tensors_expected.size());
  tensors = EvaluateNodes(output, {"id_false"}, {{"x", x_t}});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<bool>(tensors_expected[0], tensors[0]);
}

TEST_F(ConstantFoldingTest, PartialFolding_AssociativeAndCommutative) {
  std::function<Output(const Scope&, InputList)> addn_fun =
      [](const Scope& scope, InputList inputs) {
        return ops::AddN(scope, inputs);
      };
  std::function<Output(const Scope&, InputList)> accumulate_fun =
      [](const Scope& scope, InputList inputs) {
        return ops::AccumulateNV2(scope, inputs, TensorShape({2, 2}));
      };
  for (bool use_add_n : {true, false}) {
    auto fun = use_add_n ? addn_fun : accumulate_fun;
    const string op_name = use_add_n ? "AddN" : "AccumulateNV2";
    Scope s = Scope::NewRootScope();
    Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                                ops::Placeholder::Shape(TensorShape({2, 2})));
    Output y = ops::Placeholder(s.WithOpName("y"), DT_FLOAT,
                                ops::Placeholder::Shape(TensorShape({2, 2})));
    Output z = ops::Placeholder(s.WithOpName("z"), DT_FLOAT,
                                ops::Placeholder::Shape(TensorShape({2, 2})));
    Output c1 = ops::Const(s.WithOpName("c1"), 1.0f, {2, 2});
    Output c2 = ops::Const(s.WithOpName("c2"), 2.0f, {2, 2});
    Output c3 = ops::Const(s.WithOpName("c3"), 3.0f, {2, 2});
    Output acc0 = fun(s.WithOpName("acc0"), {c1, c2, c3});
    Output acc1 = fun(s.WithOpName("acc1"), {x, y, z});
    Output acc2 = fun(s.WithOpName("acc2"), {c1, x, y});
    Output acc3 = fun(s.WithOpName("acc3"), {c1, c2, z});
    Output acc4 = fun(s.WithOpName("acc4"), {c1, y, c2});
    Output acc5 = fun(s.WithOpName("acc5"), {x, c1, c2});
    Output acc6 = fun(s.WithOpName("acc6"), {x, c1, y, c2});
    Output stack = ops::Stack(s.WithOpName("stack"),
                              {acc0, acc1, acc2, acc3, acc4, acc5, acc6});

    GrapplerItem item;
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    item.fetch = {"stack"};

    ConstantFolding optimizer(nullptr /* cpu_device */);
    GraphDef output;
    Status status = optimizer.Optimize(nullptr, item, &output);
    TF_EXPECT_OK(status);

    EXPECT_EQ(16, output.node_size());
    for (const NodeDef& node : output.node()) {
      if (node.name() == "acc0") {
        EXPECT_EQ("Const", node.op());
      }
      if (node.name() == "acc1") {
        EXPECT_EQ(op_name, node.op());
        EXPECT_EQ(3, node.input_size());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("y", node.input(1));
        EXPECT_EQ("z", node.input(2));
      }
      if (node.name() == "acc2") {
        EXPECT_EQ(op_name, node.op());
        EXPECT_EQ(3, node.input_size());
        EXPECT_EQ("c1", node.input(0));
        EXPECT_EQ("x", node.input(1));
        EXPECT_EQ("y", node.input(2));
      }
      if (node.name() == "acc3") {
        EXPECT_EQ(op_name, node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("ConstantFolding/acc3_partial_split_2", node.input(0));
        EXPECT_EQ("z", node.input(1));
      }
      if (node.name() == "acc4") {
        EXPECT_EQ(op_name, node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("ConstantFolding/acc4_partial_split_2", node.input(0));
        EXPECT_EQ("y", node.input(1));
      }
      if (node.name() == "acc5") {
        EXPECT_EQ(op_name, node.op());
        EXPECT_EQ(2, node.input_size());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("ConstantFolding/acc5_partial_split_2", node.input(1));
      }
      if (node.name() == "acc6") {
        EXPECT_EQ(op_name, node.op());
        EXPECT_EQ(3, node.input_size());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("ConstantFolding/acc6_partial_split_2", node.input(1));
        EXPECT_EQ("y", node.input(2));
      }
      if (str_util::StartsWith(node.name(), "ConstantFolding/")) {
        EXPECT_EQ("Const", node.op());
      }
    }

    std::vector<string> fetch = {"acc0"};
    auto tensors_expected = EvaluateNodes(item.graph, fetch);
    auto tensors = EvaluateNodes(output, fetch);
    EXPECT_EQ(1, tensors_expected.size());
    EXPECT_EQ(1, tensors.size());
    test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
  }
}

TEST_F(ConstantFoldingTest, PartialFolding_Concat) {
  Scope s = Scope::NewRootScope();
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output y = ops::Placeholder(s.WithOpName("y"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output z = ops::Placeholder(s.WithOpName("z"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output axis = ops::Const(s.WithOpName("axis"), 0, {});
  Output c1 = ops::Const(s.WithOpName("c1"), 1.0f, {2, 2});
  Output c2 = ops::Const(s.WithOpName("c2"), 2.0f, {2, 2});
  Output concat0 = ops::Concat(s.WithOpName("concat0"), {c1, c2, c1}, axis);
  Output concat1 = ops::Concat(s.WithOpName("concat1"), {x, y, z}, axis);
  Output concat2 = ops::Concat(s.WithOpName("concat2"), {c1, x, y}, axis);
  Output concat3 = ops::Concat(s.WithOpName("concat3"), {c1, c2, z}, axis);
  Output concat4 = ops::Concat(s.WithOpName("concat4"), {c1, y, c2}, axis);
  Output concat5 = ops::Concat(s.WithOpName("concat5"), {x, c1, c2}, axis);
  Output concat6 = ops::Concat(s.WithOpName("concat6"), {x, c1, y, c2}, axis);
  Output concat7 = ops::Concat(s.WithOpName("concat7"), {x, y, c1, c2}, axis);
  Output concat8 = ops::Concat(s.WithOpName("concat8"), {x, c1, c2, y}, axis);
  Output concat9 = ops::Concat(s.WithOpName("concat9"), {c1, c2, x, y}, axis);

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.fetch = {"concat0", "concat1", "concat2", "concat3", "concat4",
                "concat5", "concat6", "concat7", "concat8", "concat9"};

  auto tensors_expected = EvaluateNodes(item.graph, {"concat0"});
  EXPECT_EQ(1, tensors_expected.size());
  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(21, output.node_size());
  for (int i = 0; i < output.node_size(); ++i) {
    const NodeDef& node = output.node(i);
    if (node.name() == "concat0") {
      EXPECT_EQ("Const", node.op());
    } else if (node.name() == "concat3") {
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("ConstantFolding/concat3_partial_split_0", node.input(0));
      EXPECT_EQ("z", node.input(1));
      EXPECT_EQ("axis", node.input(2));
    } else if (node.name() == "concat5") {
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("ConstantFolding/concat5_partial_split_1", node.input(1));
      EXPECT_EQ("axis", node.input(2));
    } else if (node.name() == "concat7") {
      EXPECT_EQ(4, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("ConstantFolding/concat7_partial_split_2", node.input(2));
      EXPECT_EQ("axis", node.input(3));
    } else if (node.name() == "concat8") {
      EXPECT_EQ(4, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("ConstantFolding/concat8_partial_split_1", node.input(1));
      EXPECT_EQ("y", node.input(2));
      EXPECT_EQ("axis", node.input(3));
    } else if (node.name() == "concat9") {
      EXPECT_EQ(4, node.input_size());
      EXPECT_EQ("ConstantFolding/concat9_partial_split_0", node.input(0));
      EXPECT_EQ("x", node.input(1));
      EXPECT_EQ("y", node.input(2));
      EXPECT_EQ("axis", node.input(3));
    } else if (str_util::StartsWith(node.name(), "ConstantFolding/")) {
      EXPECT_EQ("Const", node.op());
    } else {
      EXPECT_EQ(item.graph.node(i).DebugString(), node.DebugString());
    }
  }

  auto tensors = EvaluateNodes(output, {"concat0"});
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-6);
}

TEST_F(ConstantFoldingTest, PartialFolding_IdentityN) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output x = ops::Placeholder(scope.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({})));
  Output c1 = ops::Const(scope.WithOpName("c1"), 1.0f, {2, 2});
  Output c2 = ops::Const(scope.WithOpName("c2"), 2.0f, {2, 2});
  auto id_n = ops::IdentityN(scope.WithOpName("id_n"), {c1, x, c2});
  auto id0 = ops::Identity(scope.WithOpName("id0"), id_n[0]);
  auto id1 = ops::Identity(scope.WithOpName("id1"), id_n[1]);
  auto add0 = ops::Add(scope.WithOpName("add0"), id_n[0], id_n[1]);
  auto add1 = ops::Add(scope.WithOpName("add1"), id_n[0], id_n[2]);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("id0");
  item.fetch.push_back("id1");
  item.fetch.push_back("add0");
  item.fetch.push_back("add1");

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(8, output.node_size());
  for (const auto& node : output.node()) {
    // id_n should remain unchanged.
    if (node.name() == "id_n") {
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("c1", node.input(0));
      EXPECT_EQ("x", node.input(1));
      EXPECT_EQ("c2", node.input(2));
    }
    // id0 should be constant folded, and a control dependency from id_n.
    if (node.name() == "id0") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^id_n", node.input(0));
    }
    // id1 is unchanged.
    if ("id1" == node.name()) {
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("id_n:1", node.input(0));
    }

    if ("add0" == node.name()) {
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("c1", node.input(0));
      EXPECT_EQ("id_n:1", node.input(1));
    }
    // add1 should bo constant folded and have a control dependency from id_n.
    if ("add1" == node.name()) {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^id_n", node.input(0));
    }
  }

  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({}));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, {{"x", x_t}});
  EXPECT_EQ(4, tensors_expected.size());
  auto tensors = EvaluateNodes(output, item.fetch, {{"x", x_t}});
  EXPECT_EQ(4, tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-5);
  }
}

TEST_F(ConstantFoldingTest, TrivialPack) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output x =
      ops::RandomNormal(scope.WithOpName("x"), {2, 2}, DataType::DT_FLOAT);
  Output y = ops::Const(scope.WithOpName("y"), {2.0f}, {});
  auto stack =
      ops::Stack(scope.WithOpName("stack").WithControlDependencies({y}), {x},
                 ops::Stack::Axis(1));

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("stack");

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(5, output.node_size());
  for (const auto& node : output.node()) {
    if (node.name() == "stack") {
      EXPECT_EQ("stack", node.name());
      EXPECT_EQ("ExpandDims", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("ConstantFolding/stack_const_axis", node.input(1));
      EXPECT_EQ("^y", node.input(2));
    } else if (node.name() == "ConstantFolding/stack_const_axis") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^x", node.input(0));
    }
  }

  std::vector<string> fetch = {"stack"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  EXPECT_EQ(tensors_expected[0].shape(), tensors[0].shape());
}

// The test does not evalute the optimized and original graphs to check if their
// outputs are the same. See b/78233179.
TEST_F(ConstantFoldingTest, Enter) {
  GrapplerItem item;
  AttrValue frame_name;
  frame_name.set_s("foo");
  AttrValue is_constant_true;
  is_constant_true.set_b(true);
  AttrValue is_constant_false;
  is_constant_false.set_b(false);
  AttrValue type;
  type.set_type(DT_FLOAT);
  AttrValue value;
  Tensor value_tensor(DT_FLOAT, TensorShape({}));
  value_tensor.flat<float>()(0) = 1;
  value_tensor.AsProtoTensorContent(value.mutable_tensor());

  GraphDef& graph = item.graph;
  AddNode("x", "Placeholder", {}, {{"dtype", type}}, &graph);
  AddNode("c1", "Const", {"^x"}, {{"value", value}, {"dtype", type}}, &graph);
  AddNode("enter1", "Enter", {"x"},
          {{"T", type},
           {"frame_name", frame_name},
           {"is_constant", is_constant_true}},
          &graph);
  AddNode("enter2", "Enter", {"c1"},
          {{"T", type},
           {"frame_name", frame_name},
           {"is_constant", is_constant_true}},
          &graph);
  AddNode("enter3", "Enter", {"c1"},
          {{"T", type},
           {"frame_name", frame_name},
           {"is_constant", is_constant_false}},
          &graph);
  AddNode("id1", "Identity", {"enter1"}, {{"T", type}}, &graph);
  AddNode("id2", "Identity", {"enter2"}, {{"T", type}}, &graph);
  AddNode("id3", "Identity", {"enter2"}, {{"T", type}}, &graph);
  AddNode("id4", "Identity", {"enter3"}, {{"T", type}}, &graph);
  item.fetch.push_back("id1");
  item.fetch.push_back("id2");
  item.fetch.push_back("id3");
  item.fetch.push_back("id4");

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(9, output.node_size());
  for (const NodeDef& node : output.node()) {
    if (node.name() == "id1") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("enter1", node.input(0));
    }
    if (node.name() == "id2" || node.name() == "id3") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^enter2", node.input(0));
    }
    if (node.name() == "id4") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("enter3", node.input(0));
    }
  }
}

TEST_F(ConstantFoldingTest, TensorArraySize) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output size = ops::Const(scope.WithOpName("size"), 5, TensorShape({}));
  Output placeholder =
      ops::Placeholder(scope.WithOpName("placeholder"), DT_RESOURCE,
                       ops::Placeholder::Shape(TensorShape({2})));
  Output foo = ops::Const(scope.WithOpName("foo"), 5.0f, TensorShape({}));
  auto dynamic_array =
      ops::TensorArray(scope.WithOpName("dynamic"), size, DT_FLOAT,
                       ops::TensorArray::DynamicSize(true));
  auto static_array =
      ops::TensorArray(scope.WithOpName("static"), size, DT_FLOAT,
                       ops::TensorArray::DynamicSize(false));
  auto dynamic_sz = ops::TensorArraySize(
      scope.WithOpName("dynamic_sz"), dynamic_array.handle, dynamic_array.flow);
  auto static_sz = ops::TensorArraySize(scope.WithOpName("static_sz"),
                                        static_array.handle, static_array.flow);
  auto placeholder_sz = ops::TensorArraySize(scope.WithOpName("placeholder_sz"),
                                             placeholder, foo);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  auto tensors_expected =
      EvaluateNodes(item.graph, {"dynamic_sz", "static_sz"});

  ConstantFolding optimizer(nullptr /* cpu_device */);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(8, output.node_size());
  EXPECT_EQ("dynamic_sz", output.node(5).name());
  EXPECT_EQ("TensorArraySizeV3", output.node(5).op());
  EXPECT_EQ("static_sz", output.node(6).name());
  EXPECT_EQ("Const", output.node(6).op());
  EXPECT_EQ("placeholder_sz", output.node(7).name());
  EXPECT_EQ("TensorArraySizeV3", output.node(7).op());

  auto tensors_actual = EvaluateNodes(output, {"dynamic_sz", "static_sz"});
  EXPECT_EQ(2, tensors_expected.size());
  EXPECT_EQ(2, tensors_actual.size());
  test::ExpectTensorEqual<int32>(tensors_expected[0], tensors_actual[0]);
  test::ExpectTensorEqual<int32>(tensors_expected[1], tensors_actual[1]);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
