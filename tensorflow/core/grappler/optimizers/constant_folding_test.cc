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

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"

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
        if (DTYPE == DT_FLOAT) {
          mul1 = ops::MulNoNan(s.WithOpName("mul1"), x, zeros);
        } else {
          mul1 = ops::Mul(s.WithOpName("mul1"), x, zeros);
        }
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
      ConstantFolding optimizer(/*cpu_device=*/nullptr);
      GraphDef output;
      Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  void MulConvPushDownTest(const TensorShape& input_shape,
                           const TensorShape& filter_shape,
                           const TensorShape& mul_const_input_shape,
                           const bool use_3d_conv, const char* padding,
                           const char* data_format, const bool expect_folded) {
    // Tests if the following rewrite is performed:
    //
    //         *                       Conv2D
    //        / \                       / \
    //       c  Conv2D        -->      x  (c * filter)
    //           / \
    //          x  filter
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();

    Tensor filter_values(DT_FLOAT, filter_shape);
    for (int i = 0; i < filter_values.NumElements(); ++i) {
      filter_values.flat<float>()(i) = std::sqrt(static_cast<float>(i));
    }
    Output filter =
        ops::Const(s.WithOpName("filter"), Input::Initializer(filter_values));

    Output input = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                                    ops::Placeholder::Shape(input_shape));

    Output conv;
    if (use_3d_conv) {
      conv = ops::Conv3D(s.WithOpName("conv"), input, filter, {1, 1, 1, 1, 1},
                         padding, ops::Conv3D::DataFormat(data_format));
    } else {
      conv = ops::Conv2D(s.WithOpName("conv"), input, filter, {1, 1, 1, 1},
                         padding, ops::Conv2D::DataFormat(data_format));
    }
    Tensor mul_const_input(DT_FLOAT, mul_const_input_shape);
    for (int i = 0; i < mul_const_input.NumElements(); ++i) {
      mul_const_input.flat<float>()(i) = static_cast<float>(i + 3);
    }
    Output c =
        ops::Const(s.WithOpName("c"), Input::Initializer(mul_const_input));
    Output mul = ops::Mul(s.WithOpName("mul"), c, conv);

    GrapplerItem item;
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    TF_EXPECT_OK(status);

    EXPECT_EQ(5, output.node_size());
    int found = 0;
    if (expect_folded) {
      for (const auto& node : output.node()) {
        if (node.name() == "mul") {
          found++;
          EXPECT_EQ(use_3d_conv ? "Conv3D" : "Conv2D", node.op());
          EXPECT_EQ(2, node.input_size());
          EXPECT_EQ("x", node.input(0));
          EXPECT_EQ("conv/merged_input", node.input(1));
        } else if (node.name() == "conv/merged_input") {
          found++;
          EXPECT_EQ("Const", node.op());
          EXPECT_EQ(0, node.input_size());
        }
      }
    } else {
      for (const auto& node : output.node()) {
        if (node.name() == "mul") {
          found++;
          EXPECT_EQ("Mul", node.op());
          EXPECT_EQ(2, node.input_size());
          EXPECT_EQ("c", node.input(0));
          EXPECT_EQ("conv", node.input(1));
        } else if (node.name() == "conv") {
          found++;
          EXPECT_EQ(use_3d_conv ? "Conv3D" : "Conv2D", node.op());
          EXPECT_EQ(2, node.input_size());
          EXPECT_EQ("x", node.input(0));
          EXPECT_EQ("filter", node.input(1));
        }
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

  template <typename T>
  void PaddingWithZeroSize() {
    tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

    auto in1 = ops::Variable(scope.WithOpName("in1"), {4, 6}, DT_INT32);
    auto in2 = ops::Variable(scope.WithOpName("in2"), {2, 2}, DT_INT32);
    auto paddings1 =
        ops::Const<T>(scope.WithOpName("paddings1"), {0, 0, 0, 0}, {2, 2});
    auto paddings2 =
        ops::Const<T>(scope.WithOpName("paddings2"), {1, 1, 2, 2}, {2, 2});
    auto c1 = ops::Const(scope.WithOpName("c1"), 1);
    auto c2 = ops::Const(scope.WithOpName("c2"), 1);

    ops::PadV2 p1(scope.WithOpName("p1"), in1, paddings1, c1);
    ops::PadV2 p2(scope.WithOpName("p2"), in2, paddings2, c2);

    ops::Add out(scope.WithOpName("out"), p1, p2);

    GrapplerItem item;
    item.fetch = {"out"};
    TF_CHECK_OK(scope.ToGraphDef(&item.graph));

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef got;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
    TF_EXPECT_OK(status);

    GraphDef want;
    AddNode("in1", "VariableV2", {}, {}, &want);
    AddNode("in2", "VariableV2", {}, {}, &want);
    AddNode("paddings1", "Const", {}, {}, &want);
    AddNode("paddings2", "Const", {}, {}, &want);
    AddNode("c1", "Const", {}, {}, &want);
    AddNode("c2", "Const", {}, {}, &want);
    AddNode(
        "p1", "Identity",
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  Output c1 = ops::Const(s.WithOpName("c1"), 1.0f, {1});
  Output c2 = ops::Const(s.WithOpName("c2"), 2.0f, {2});
  Output c3 = ops::Const(s.WithOpName("c3"), 3.0f, {2});
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output add_child = ops::Add(s.WithOpName("add_child"), c2, x);
  Output add_parent = ops::Add(s.WithOpName("add_parent"), c1, add_child);

  Output c4 = ops::Const(s.WithOpName("c4"), 4.0f, {2});
  Output c5 = ops::Const(s.WithOpName("c5"), 5.0f, {2});
  Output c20 = ops::Const(s.WithOpName("c20"), 20.0f, {2});
  Output y = ops::Placeholder(s.WithOpName("y"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output mul_child = ops::Mul(s.WithOpName("mul_child"), c4, y);
  Output mul_parent = ops::Mul(s.WithOpName("mul_parent"), c5, mul_child);
  Output addmul_child = ops::Add(s.WithOpName("addmul_child"), c4, x);
  Output addmul_parent =
      ops::Mul(s.WithOpName("addmul_parent"), c5, addmul_child);

  GrapplerItem item;
  item.fetch = {"add_parent", "mul_parent", "addmul_parent"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  EXPECT_EQ(10, output.node_size());
  for (const auto& node : output.node()) {
    if (node.name() == "add_child") {
      EXPECT_EQ("Const", node.op());
      TensorProto t = node.attr().at("value").tensor();
      ASSERT_EQ(1, t.tensor_shape().dim_size());
      EXPECT_EQ(2, t.tensor_shape().dim(0).size());
    } else if (node.name() == "add_parent") {
      EXPECT_EQ("Add", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("add_child", node.input(1));
    } else if (node.name() == "mul_child") {
      EXPECT_EQ("Const", node.op());
      TensorProto t = node.attr().at("value").tensor();
      EXPECT_EQ(1, t.tensor_shape().dim_size());
      EXPECT_EQ(2, t.tensor_shape().dim(0).size());
    } else if (node.name() == "mul_parent") {
      EXPECT_EQ("Mul", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("mul_child", node.input(1));
    } else if (node.name() == "addmul_child") {
      // Unchanged.
      EXPECT_EQ("Add", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("c4", node.input(0));
      EXPECT_EQ("x", node.input(1));
    }
  }

  // Check that the result nodes have the expected value.
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto y_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));

  std::vector<string> fetch = {"add_parent", "mul_parent"};
  auto tensor_expected =
      EvaluateNodes(item.graph, fetch, {{"x", x_t}, {"y", y_t}});
  ASSERT_EQ(fetch.size(), tensor_expected.size());
  fetch = {"add_parent", "mul_parent"};
  auto tensors = EvaluateNodes(output, fetch, {{"x", x_t}, {"y", y_t}});
  ASSERT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<float>(tensor_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, AddSubtactTree) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output c1 = ops::Const(s.WithOpName("c1"), 1.0f, {1});
  Output x = ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                              ops::Placeholder::Shape(TensorShape({2, 2})));
  Output sub_child = ops::Sub(s.WithOpName("sub_child"), x, x);
  Output add_parent = ops::Add(s.WithOpName("add_parent"), sub_child, c1);

  GrapplerItem item;
  item.fetch = {"add_parent"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  // We expect the following rewrite(s) to occur:
  //
  //     +                +
  //    / \              / \
  //   -   1     -->    -   x
  //  / \              / \
  // x   x            1   x

  EXPECT_EQ(4, output.node_size());
  for (const auto& node : output.node()) {
    if (node.name() == "sub_child") {
      EXPECT_EQ("Sub", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("c1", node.input(0));
      EXPECT_EQ("x", node.input(1));
    } else if (node.name() == "add_parent") {
      EXPECT_EQ("Add", node.op());
      ASSERT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("sub_child", node.input(1));
    }
  }

  // Check that the result nodes have the expected value.
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));

  std::vector<string> fetch = {"add_parent"};
  auto tensor_expected = EvaluateNodes(item.graph, fetch, {{"x", x_t}});
  ASSERT_EQ(fetch.size(), tensor_expected.size());
  fetch = {"add_parent"};
  auto tensors = EvaluateNodes(output, fetch, {{"x", x_t}});
  ASSERT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<float>(tensor_expected[i], tensors[i]);
  }
}

TEST_F(ConstantFoldingTest, ConstantPushDown) {
  for (int is_add : {true, false}) {
    for (int is_parent_commutative : {true, false}) {
      for (int is_child_commutative : {true, false}) {
        for (int is_left_child_const : {true, false}) {
          for (int is_left_leaf_const : {true, false}) {
            tensorflow::Scope s = tensorflow::Scope::NewRootScope();
            Output c2 = ops::Const(s.WithOpName("c2"), 2.0f, {2});
            Output c3 = ops::Const(s.WithOpName("c3"), 3.0f, {2});
            Output x =
                ops::Placeholder(s.WithOpName("x"), DT_FLOAT,
                                 ops::Placeholder::Shape(TensorShape({2, 2})));

            auto get_op = [&](bool is_commutative, bool is_left_arg_const,
                              const string& name, const Output& const_arg,
                              const Output non_const_arg) -> Output {
              if (is_add) {
                if (is_commutative) {
                  return ops::Add(
                      s.WithOpName(name),
                      is_left_arg_const ? const_arg : non_const_arg,
                      is_left_arg_const ? non_const_arg : const_arg);
                } else {
                  return ops::Sub(
                      s.WithOpName(name),
                      is_left_arg_const ? const_arg : non_const_arg,
                      is_left_arg_const ? non_const_arg : const_arg);
                }
              } else {
                if (is_commutative) {
                  return ops::Mul(
                      s.WithOpName(name),
                      is_left_arg_const ? const_arg : non_const_arg,
                      is_left_arg_const ? non_const_arg : const_arg);
                } else {
                  return ops::Div(
                      s.WithOpName(name),
                      is_left_arg_const ? const_arg : non_const_arg,
                      is_left_arg_const ? non_const_arg : const_arg);
                }
              }
            };

            Output child = get_op(is_child_commutative, is_left_leaf_const,
                                  "child", c2, x);
            Output parent = get_op(is_parent_commutative, is_left_child_const,
                                   "parent", c3, child);
            GrapplerItem item;
            item.fetch = {"parent"};
            TF_CHECK_OK(s.ToGraphDef(&item.graph));

            ConstantFolding optimizer(/*cpu_device=*/nullptr);
            GraphDef output;
            Status status =
                optimizer.Optimize(/*cluster=*/nullptr, item, &output);
            TF_EXPECT_OK(status);

            // Check that the result nodes have the expected value.
            auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
            std::vector<string> fetch = {"parent"};
            auto tensor_expected =
                EvaluateNodes(item.graph, fetch, {{"x", x_t}});
            ASSERT_EQ(fetch.size(), tensor_expected.size());
            fetch = {"parent"};
            auto tensors = EvaluateNodes(output, fetch, {{"x", x_t}});
            ASSERT_EQ(fetch.size(), tensors.size());
            for (int i = 0; i < fetch.size(); i++) {
              test::ExpectTensorEqual<float>(tensor_expected[i], tensors[i]);
            }
          }
        }
      }
    }
  }
}

TEST_F(ConstantFoldingTest, ConstantPushDownBiasAdd) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c_mat = ops::Const(s.WithOpName("c_mat"), 2.0f, {2, 2});
  Output c_vec = ops::Const(s.WithOpName("c_vec"), 3.0f, {2});
  Output x_mat = ops::Placeholder(s.WithOpName("x_mat"), DT_FLOAT,
                                  ops::Placeholder::Shape(TensorShape({2, 2})));
  Output x_vec = ops::Placeholder(s.WithOpName("x_vec"), DT_FLOAT,
                                  ops::Placeholder::Shape(TensorShape({2})));
  // Rewrite expected for cases 1 through 3 and their symmetric equivalents,
  // and case 4.
  Output child1 = ops::BiasAdd(s.WithOpName("child1"), c_mat, x_vec);
  Output parent1 = ops::Add(s.WithOpName("parent1"), child1, c_vec);
  Output child1a = ops::BiasAdd(s.WithOpName("child1a"), c_mat, x_vec);
  Output parent1a = ops::Add(s.WithOpName("parent1a"), c_vec, child1a);

  Output child2 = ops::BiasAdd(s.WithOpName("child2"), x_mat, c_vec);
  Output parent2 = ops::Add(s.WithOpName("parent2"), child2, c_mat);
  Output child2a = ops::BiasAdd(s.WithOpName("child2a"), x_mat, c_vec);
  Output parent2a = ops::Add(s.WithOpName("parent2a"), c_mat, child2a);

  Output child3 = ops::Add(s.WithOpName("child3"), c_mat, x_vec);
  Output parent3 = ops::BiasAdd(s.WithOpName("parent3"), child3, c_vec);
  Output child3a = ops::Add(s.WithOpName("child3a"), x_vec, c_mat);
  Output parent3a = ops::BiasAdd(s.WithOpName("parent3a"), child3a, c_vec);

  Output child4 = ops::BiasAdd(s.WithOpName("child4"), c_mat, x_vec);
  Output parent4 = ops::BiasAdd(s.WithOpName("parent4"), child4, c_vec);

  // No rewrite expected.
  Output child5 = ops::Add(s.WithOpName("child5"), x_vec, x_vec);
  Output parent5 = ops::BiasAdd(s.WithOpName("parent5"), c_mat, child5);
  Output child6 = ops::Add(s.WithOpName("child6"), x_vec, c_vec);
  Output parent6 = ops::BiasAdd(s.WithOpName("parent6"), c_mat, child6);
  Output child7 = ops::Add(s.WithOpName("child7"), x_mat, c_vec);
  Output parent7 = ops::BiasAdd(s.WithOpName("parent7"), child7, c_vec);

  GrapplerItem item;
  item.fetch = {"parent1",  "parent2", "parent3", "parent1a", "parent2a",
                "parent3a", "parent4", "parent5", "parent6",  "parent7"};
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(24, output.node_size());
  for (const auto& node : output.node()) {
    if (node.name() == "child1" || node.name() == "child1a" ||
        node.name() == "child2" || node.name() == "child2a" ||
        node.name() == "child3" || node.name() == "child3a" ||
        node.name() == "child4") {
      EXPECT_EQ(node.op(), "Const") << " node: " << node.name();
    } else if (node.name() != "c_mat" && node.name() != "c_vec") {
      EXPECT_NE(node.op(), "Const") << " node: " << node.name();
    }
  }
  // Check that the result nodes have the expected value.
  auto x_mat_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 2}));
  auto x_vec_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({2}));
  std::vector<string> fetch = item.fetch;
  auto tensor_expected = EvaluateNodes(
      item.graph, fetch, {{"x_vec", x_vec_t}, {"x_mat", x_mat_t}});
  ASSERT_EQ(fetch.size(), tensor_expected.size());
  auto tensors =
      EvaluateNodes(output, fetch, {{"x_vec", x_vec_t}, {"x_mat", x_mat_t}});
  ASSERT_EQ(fetch.size(), tensors.size());
  for (int i = 0; i < fetch.size(); i++) {
    test::ExpectTensorEqual<float>(tensor_expected[i], tensors[i]);
  }
}

// This test fails on ROCm platform (see commit message for details)
#ifndef TENSORFLOW_USE_ROCM
TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv2D_ScalarConst) {
  for (string data_format : {
         "NHWC",
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
             "NCHW"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
       }) {
    MulConvPushDownTest(
        /*input_shape=*/data_format == "NHWC" ? TensorShape{4, 10, 10, 3}
                                              : TensorShape{4, 3, 10, 10},
        /*filter_shape=*/{2, 2, 3, 5},
        /*mul_const_input_shape=*/{},
        /*use_3d_conv=*/false,
        /*padding=*/"VALID", data_format.c_str(),
        /*expect_folded=*/true);
  }
}
#endif

// This test fails on ROCm platform (see commit message for details)
#ifndef TENSORFLOW_USE_ROCM
TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv2D_SingletonConst) {
  for (string data_format : {
         "NHWC",
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
             "NCHW"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
       }) {
    for (auto mul_const_input_shape :
         {TensorShape{1}, TensorShape{1, 1, 1, 1}}) {
      MulConvPushDownTest(
          /*input_shape=*/data_format == "NHWC" ? TensorShape{4, 10, 10, 3}
                                                : TensorShape{4, 3, 10, 10},
          /*filter_shape=*/{2, 2, 3, 5}, mul_const_input_shape,
          /*use_3d_conv=*/false,
          /*padding=*/"VALID", data_format.c_str(),
          /*expect_folded=*/true);
    }
  }
}
#endif

TEST_F(ConstantFoldingTest,
       MulConvPushDownTest_Conv2D_SingletonConst_ShapeMismatch) {
  for (string data_format : {
         "NHWC",
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
             "NCHW"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
       }) {
    MulConvPushDownTest(
        /*input_shape=*/data_format == "NHWC" ? TensorShape{4, 10, 10, 3}
                                              : TensorShape{4, 3, 10, 10},
        /*filter_shape=*/{2, 2, 3, 5},
        /*mul_const_input_shape=*/{1, 1, 1, 1, 1},
        /*use_3d_conv=*/false,
        /*padding=*/"VALID", data_format.c_str(),
        /*expect_folded=*/false);
  }
}

TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv2D_3x1x3Const) {
  for (auto data_format : {
         "NHWC",
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
             "NCHW"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
       }) {
    MulConvPushDownTest(
        /*input_shape=*/{3, 3, 3, 3},
        /*filter_shape=*/{3, 3, 3, 3},
        /*mul_const_input_shape=*/{3, 1, 3},
        /*use_3d_conv=*/false,
        /*padding=*/"SAME", data_format,
        /*expect_folded=*/false);
  }
}

TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv2D_NHWC_VectorLikeConst) {
  for (auto mul_const_input_shape :
       {TensorShape{3}, TensorShape{1, 3}, TensorShape{1, 1, 1, 3}}) {
    MulConvPushDownTest(
        /*input_shape=*/{3, 3, 3, 3},
        /*filter_shape=*/{3, 3, 3, 3}, mul_const_input_shape,
        /*use_3d_conv=*/false,
        /*padding=*/"SAME",
        /*data_format=*/"NHWC",
        /*expect_folded=*/true);
  }
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv2D_NCHW_VectorLikeConst) {
  for (auto mul_const_input_shape :
       {TensorShape{3}, TensorShape{3, 1, 1}, TensorShape{1, 3, 1, 1}}) {
    MulConvPushDownTest(
        /*input_shape=*/{3, 3, 3, 3},
        /*filter_shape=*/{3, 3, 3, 3}, mul_const_input_shape,
        /*use_3d_conv=*/false,
        /*padding=*/"SAME",
        /*data_format=*/"NCHW",
        // TODO(laigd): optimization should happen in this case.
        /*expect_folded=*/false);
  }
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv2D_3x1Const) {
  for (auto data_format : {
         "NHWC",
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
             "NCHW"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
       }) {
    MulConvPushDownTest(
        /*input_shape=*/{3, 3, 3, 3},
        /*filter_shape=*/{3, 3, 3, 3},
        /*mul_const_input_shape=*/{3, 1},
        /*use_3d_conv=*/false,
        /*padding=*/"SAME", data_format,
        /*expect_folded=*/false);
  }
}

// This test fails on ROCm platform (see commit message for details)
#ifndef TENSORFLOW_USE_ROCM
TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv3D_NDHWC_1x1x3Const) {
  MulConvPushDownTest(
      /*input_shape=*/{3, 3, 3, 3, 3},
      /*filter_shape=*/{3, 3, 3, 3, 3},
      /*mul_const_input_shape=*/{1, 1, 3},
      /*use_3d_conv=*/true,
      /*padding=*/"SAME",
      /*data_format=*/"NDHWC",
      /*expect_folded=*/true);
}
#endif

TEST_F(ConstantFoldingTest, MulConvPushDownTest_Conv3D_NCDHW_3x1x1x1Const) {
  MulConvPushDownTest(
      /*input_shape=*/{3, 3, 3, 3, 3},
      /*filter_shape=*/{3, 3, 3, 3, 3},
      /*mul_const_input_shape=*/{3, 1, 1, 1},
      /*use_3d_conv=*/true,
      /*padding=*/"SAME",
      /*data_format=*/"NDHWC",
      // TODO(laigd): optimization should happen in this case.
      /*expect_folded=*/false);
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
    Output zeros_const_bcast =
        ops::Const(s.WithOpName("zeros_const_bcast"), 0.0f, {2, 2, 2});
    Output zeros_like = ops::ZerosLike(s.WithOpName("zeros_like"), x);
    Output zeros_fill = ops::Fill(s.WithOpName("zeros_fill"), {2, 2}, 0.0f);
    Output zeros = const_type == kConst
                       ? zeros_const
                       : (const_type == kLike ? zeros_like : zeros_fill);
    Output ones_const = ops::Const(s.WithOpName("ones_const"), 1.0f, {2, 2});
    Output ones_const_bcast =
        ops::Const(s.WithOpName("ones_const_bcast"), 1.0f, {2, 2, 2});
    Output ones_like = ops::OnesLike(s.WithOpName("ones_like"), x);
    Output ones_fill = ops::Fill(s.WithOpName("ones_fill"), {2, 2}, 1.0f);
    Output ones = const_type == kConst
                      ? ones_const
                      : (const_type == kLike ? ones_like : ones_fill);
    Output mul1 = ops::Mul(s.WithOpName("mul1"), x, zeros);
    Output mul2 = ops::Mul(s.WithOpName("mul2"), zeros, y);
    Output mul1_bcast =
        ops::Mul(s.WithOpName("mul1_bcast"), x, ones_const_bcast);
    Output mul2_bcast =
        ops::Mul(s.WithOpName("mul2_bcast"), ones_const_bcast, y);
    Output mul3 = ops::Mul(s.WithOpName("mul3"), x, ones);
    Output mul4 = ops::Mul(s.WithOpName("mul4"), ones, y);
    Output mul5 = ops::MulNoNan(s.WithOpName("mul5"), x, zeros_1d);
    Output mul6 = ops::MulNoNan(s.WithOpName("mul6"), zeros_1d, y);
    Output div1 = ops::Div(s.WithOpName("div1"), x, ones);
    Output div2 = ops::Div(s.WithOpName("div2"), ones, y);
    Output matmul1 = ops::MatMul(s.WithOpName("matmul1"), x, zeros);
    Output matmul2 = ops::MatMul(s.WithOpName("matmul2"), zeros, y);
    Output matmul3 = ops::MatMul(s.WithOpName("matmul3"), a, zeros);
    Output matmul4 = ops::MatMul(s.WithOpName("matmul4"), zeros, b);
    Output add1 = ops::Add(s.WithOpName("add1"), x, zeros);
    Output add2 = ops::Add(s.WithOpName("add2"), zeros, y);
    Output add1_bcast =
        ops::Add(s.WithOpName("add1_bcast"), x, zeros_const_bcast);
    Output add2_bcast =
        ops::Add(s.WithOpName("add2_bcast"), zeros_const_bcast, y);
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
    item.fetch = {"stack",      "matmul3",    "matmul4",   "mul1_bcast",
                  "mul2_bcast", "add1_bcast", "add2_bcast"};

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    TF_EXPECT_OK(status);

    const string suffix =
        (const_type == kConst ? "_const"
                              : (const_type == kLike ? "_like" : "_fill"));
    const string zeros_name = strings::StrCat("zeros", suffix);
    const string ones_name = strings::StrCat("ones", suffix);
    const string ctrl_zeros_name = strings::StrCat("^zeros", suffix);
    const string ctrl_ones_name = strings::StrCat("^ones", suffix);

    EXPECT_EQ(const_type == kFill ? 42 : 38, output.node_size());
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
      } else if (name == "mul1_bcast") {
        EXPECT_EQ("BroadcastTo", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^ones_const_bcast", node.input(2));
      } else if (name == "mul2_bcast") {
        EXPECT_EQ("BroadcastTo", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ("^ones_const_bcast", node.input(2));
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
      } else if (name == "add1_bcast") {
        EXPECT_EQ("BroadcastTo", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^zeros_const_bcast", node.input(2));
      } else if (name == "add2_bcast") {
        EXPECT_EQ("BroadcastTo", node.op());
        EXPECT_EQ("y", node.input(0));
        EXPECT_EQ("^zeros_const_bcast", node.input(2));
      } else if (name == "bias_add1") {
        EXPECT_EQ("Identity", node.op());
        EXPECT_EQ("x", node.input(0));
        EXPECT_EQ("^zeros_1d", node.input(1));
      } else if (name == "bias_add2") {
        EXPECT_EQ("BroadcastTo", node.op());
        EXPECT_EQ("bias", node.input(0));
        EXPECT_EQ("ConstantFolding/bias_add2-broadcastto_shape-1",
                  node.input(1));
        EXPECT_EQ(ctrl_zeros_name, node.input(2));
      } else if (name == "ConstantFolding/bias_add2-broadcastto_shape-1") {
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ(ctrl_zeros_name, node.input(0));
        EXPECT_EQ(node.attr().at("dtype").type(), DT_INT32);
        TensorProto t = node.attr().at("value").tensor();
        EXPECT_EQ(DT_INT32, t.dtype());
        EXPECT_EQ(1, t.tensor_shape().dim_size());
        EXPECT_EQ(2, t.tensor_shape().dim(0).size());
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
  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
  Output i3 = ops::Identity(scope.WithOpName("i3"), {i2});

  GrapplerItem item;
  item.fetch.push_back("i3");
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> expected_nodes = {"dflt", "p1", "p2", "i3"};
  EXPECT_EQ(output.node_size(), expected_nodes.size());
  int i = 0;
  int found = 0;
  for (const auto& node : output.node()) {
    EXPECT_EQ(expected_nodes[i], output.node(i).name());
    i++;
    if (node.name() == "i3") {
      EXPECT_EQ("Const", node.op());
      ++found;
      auto folded = EvaluateNodes(output, {"i3"});
      auto expected = EvaluateNodes(item.graph, {"i3"});
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("split_dim", "Const", {}, {}, &want);
  AddNode("s1", "Identity", {"in1", AsControlDependency("split_dim")}, {},
          &want);
  AddNode("s2", "Split", {"split_dim", "in2"}, {}, &want);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef got;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef got;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef got;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef got;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

TEST_F(ConstantFoldingTest, MergeConcat) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 = ops::Variable(scope.WithOpName("in1"), {4, 6}, DT_FLOAT);
  Output in2 = ops::Variable(scope.WithOpName("in2"), {4, 6}, DT_FLOAT);
  Output in3 = ops::Variable(scope.WithOpName("in3"), {4, 6}, DT_FLOAT);
  Output axis = ops::Const(scope.WithOpName("axis"), 0, {});

  ops::Concat c1(scope.WithOpName("c1"), {in1, in2}, axis);
  ops::Concat c2(scope.WithOpName("c2"), {Output(c1), in3}, axis);

  GrapplerItem item;
  item.fetch = {"c2"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("in3", "VariableV2", {}, {}, &want);
  AddNode("axis", "Const", {}, {}, &want);
  AddNode("c2", "ConcatV2", {"in1", "in2", "in3", "axis"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, MergeConcat_SameInput) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 = ops::Variable(scope.WithOpName("in1"), {4, 6}, DT_FLOAT);
  Output in2 = ops::Variable(scope.WithOpName("in2"), {4, 6}, DT_FLOAT);
  Output in3 = ops::Variable(scope.WithOpName("in3"), {4, 6}, DT_FLOAT);
  Output axis = ops::Const(scope.WithOpName("axis"), 0, {});

  ops::Concat c1(scope.WithOpName("c1"), {in1, in2}, axis);
  ops::Concat c2(scope.WithOpName("c2"), {Output(c1), in3, Output(c1)}, axis);

  GrapplerItem item;
  item.fetch = {"c2"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("in3", "VariableV2", {}, {}, &want);
  AddNode("axis", "Const", {}, {}, &want);
  AddNode("c2", "ConcatV2", {"in1", "in2", "in3", "in1", "in2", "axis"}, {},
          &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, MergeConcat_ConcatWithConst) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 = ops::Variable(scope.WithOpName("in1"), {2, 6}, DT_FLOAT);
  Output in2 = ops::Variable(scope.WithOpName("in2"), {}, DT_FLOAT);
  Output in3 = ops::Variable(scope.WithOpName("in3"), {4, 6}, DT_FLOAT);
  Output axis = ops::Const(scope.WithOpName("axis"), 0, {});

  ops::Concat c1(scope.WithOpName("c1"), {in1, in2}, axis);
  ops::Concat c2(scope.WithOpName("c2"), {Output(c1), in3}, axis);

  GrapplerItem item;
  item.fetch = {"c2"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("in3", "VariableV2", {}, {}, &want);
  AddNode("axis", "Const", {}, {}, &want);
  AddNode("c2", "ConcatV2", {"in1", "in2", "in3", "axis"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, MergeConcat_AxisMismatch) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output in1 = ops::Variable(scope.WithOpName("in1"), {2, 5}, DT_FLOAT);
  Output in2 = ops::Variable(scope.WithOpName("in2"), {}, DT_FLOAT);
  Output in3 = ops::Variable(scope.WithOpName("in3"), {4, 6}, DT_FLOAT);
  Output axis1 = ops::Const(scope.WithOpName("axis1"), 0, {});
  Output axis2 = ops::Const(scope.WithOpName("axis2"), 1, {});

  ops::Concat c1(scope.WithOpName("c1"), {in1, in2}, axis2);
  ops::Concat c2(scope.WithOpName("c2"), {Output(c1), in3}, axis1);

  GrapplerItem item;
  item.fetch = {"c2"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("in1", "VariableV2", {}, {}, &want);
  AddNode("in2", "VariableV2", {}, {}, &want);
  AddNode("in3", "VariableV2", {}, {}, &want);
  AddNode("axis1", "Const", {}, {}, &want);
  AddNode("axis2", "Const", {}, {}, &want);
  AddNode("c1", "ConcatV2", {"in1", "in2", "axis2"}, {}, &want);
  AddNode("c2", "ConcatV2", {"c1", "in3", "axis1"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, MergeConcat_PartialFolding) {
  Scope scope = Scope::NewRootScope();
  Output c1 = ops::Const(scope.WithOpName("c1"), 1.0f, {2, 2});
  Output c2 = ops::Const(scope.WithOpName("c2"), 2.0f, {2, 2});
  Output c3 = ops::Const(scope.WithOpName("c3"), 3.0f, {2, 2});
  Output c4 = ops::Const(scope.WithOpName("c4"), 4.0f, {2, 2});
  Output ph = ops::Placeholder(scope.WithOpName("ph"), DT_FLOAT,
                               ops::Placeholder::Shape(TensorShape({2, 2})));
  Output axis = ops::Const(scope.WithOpName("axis"), 0, {});

  ops::Concat concat1(scope.WithOpName("concat1"), {c1, c2, ph}, axis);
  ops::Concat concat2(scope.WithOpName("concat2"), {c3, c4, Output(concat1)},
                      axis);

  GrapplerItem item;
  item.fetch = {"concat2"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(nullptr, item, &got);
  TF_EXPECT_OK(status);

  GraphDef want;
  AddNode("ConstantFolding/concat2_partial_split_0", "Const", {}, {}, &want);
  AddNode("axis", "Const", {}, {}, &want);
  AddNode("ph", "Placeholder", {}, {}, &want);
  AddNode("concat2", "ConcatV2",
          {"ConstantFolding/concat2_partial_split_0", "ph", "axis"}, {}, &want);

  CompareGraphs(want, got);
}

TEST_F(ConstantFoldingTest, PaddingWithZeroSize) {
  PaddingWithZeroSize<int32>();
  PaddingWithZeroSize<int64>();
}

TEST_F(ConstantFoldingTest, SqueezeWithAllDimensionsGreaterThanOne) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto in1 = ops::Variable(scope.WithOpName("in1"), {2, 3}, DT_INT32);
  auto in2 = ops::Variable(scope.WithOpName("in2"), {1, 2, 3, 1}, DT_INT32);

  ops::Squeeze s1(scope.WithOpName("s1"), in1);
  ops::Squeeze s2(scope.WithOpName("s2"), in2);

  ops::Add out(scope.WithOpName("out"), s1, s2);

  GrapplerItem item;
  item.fetch = {"out"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef got;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &got);
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

  // Test with unknown input shape.
  Output a = ops::Placeholder(scope.WithOpName("a"), DT_FLOAT);
  Output p3 = ops::Prod(scope.WithOpName("p3"), a, i, attr);

  GrapplerItem item;
  item.fetch = {"s", "p2", "p3"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
    } else if (node.name() == "p3") {
      found++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("a", node.input(0));
      EXPECT_EQ("^i", node.input(1));
    }
  }
  EXPECT_EQ(3, found);

  auto v_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5, 7}));
  auto v2_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5, 1}));
  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 5, 7}));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch,
                                        {{"v", v_t}, {"v2", v2_t}, {"a", a_t}});
  EXPECT_EQ(3, tensors_expected.size());
  auto tensors =
      EvaluateNodes(output, item.fetch, {{"v", v_t}, {"v2", v2_t}, {"a", a_t}});
  EXPECT_EQ(3, tensors.size());
  test::ExpectTensorNear<float>(tensors_expected[0], tensors[0], 1e-5);
  test::ExpectTensorNear<float>(tensors_expected[1], tensors[1], 1e-5);
  test::ExpectTensorNear<float>(tensors_expected[2], tensors[2], 1e-5);
}

TEST_F(ConstantFoldingTest, SingleElementEmptyAxisReduction) {
  // Build a simple graph with reductions that involve single-element input and
  // no axes to reduce along.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output input_var_three_dim = ops::Variable(
      scope.WithOpName("input_var_three_dim"), {1, 1, 1}, DT_FLOAT);
  Output input_var_one_dim =
      ops::Variable(scope.WithOpName("input_var_one_dim"), {1}, DT_FLOAT);
  Output one_axis = ops::Const(scope.WithOpName("one_axis"), {0}, {1});
  Output multiple_axes =
      ops::Const(scope.WithOpName("multiple_axes"), {1, 0}, {2});
  Output variable_axis =
      ops::Variable(scope.WithOpName("input_var_axis"), {1}, DT_INT32);
  ops::Mean::Attrs attr;
  attr = attr.KeepDims(false);
  // Should be optimized to Reshape.
  Output mean_1 = ops::Mean(scope.WithOpName("mean_1"), input_var_three_dim,
                            one_axis, attr.KeepDims(false));
  Output mean_2 = ops::Mean(scope.WithOpName("mean_2"), input_var_three_dim,
                            multiple_axes, attr.KeepDims(false));
  // Should remain as-is, since OutputProperties will not be known this node.
  Output mean_3 = ops::Mean(scope.WithOpName("mean_3"), input_var_one_dim,
                            one_axis, attr.KeepDims(false));
  // Should remain as-is.
  Output mean_4 = ops::Mean(scope.WithOpName("mean_4"), input_var_three_dim,
                            variable_axis, attr.KeepDims(false));
  // Should be optimized to Identity, since KeepDims=true.
  Output mean_5 = ops::Mean(scope.WithOpName("mean_5"), input_var_three_dim,
                            multiple_axes, attr.KeepDims(true));

  GrapplerItem item;
  item.fetch = {"mean_1", "mean_2", "mean_3", "mean_4", "mean_5"};
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Ensure Mean node is optimized to Reshape.
  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "mean_1" || node.name() == "mean_2") {
      found++;
      EXPECT_EQ("Reshape", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("input_var_three_dim", node.input(0));
    } else if (node.name() == "mean_3") {
      found++;
      EXPECT_EQ("Mean", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("input_var_one_dim", node.input(0));
    } else if (node.name() == "mean_4") {
      found++;
      EXPECT_EQ("Mean", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("input_var_three_dim", node.input(0));
    } else if (node.name() == "mean_5") {
      found++;
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("^multiple_axes", node.input(1));
    }
  }
  EXPECT_EQ(5, found);

  // Ensure resultant values from Mean and Reshape are the same.
  auto input_var_three_dim_t =
      GenerateRandomTensor<DT_FLOAT>(TensorShape({1, 1, 1}));
  auto input_var_one_dim_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({1}));
  Tensor input_var_axis_t(DT_INT32, TensorShape({1}));
  input_var_axis_t.flat<int32>()(0) = 0;
  auto tensors_expected =
      EvaluateNodes(item.graph, item.fetch,
                    {{"input_var_three_dim", input_var_three_dim_t},
                     {"input_var_one_dim", input_var_one_dim_t},
                     {"input_var_axis", input_var_axis_t}});
  EXPECT_EQ(5, tensors_expected.size());
  auto tensors = EvaluateNodes(output, item.fetch,
                               {{"input_var_three_dim", input_var_three_dim_t},
                                {"input_var_one_dim", input_var_one_dim_t},
                                {"input_var_axis", input_var_axis_t}});
  EXPECT_EQ(5, tensors.size());
  for (int i = 0; i < 5; ++i) {
    test::ExpectTensorNear<float>(tensors_expected[i], tensors[i], 1e-5);
  }
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

TEST_F(ConstantFoldingTest, LargeConstantNoSizeIncrease) {
  // Build a simple graph with a large constant with size greater than
  // kMaxConstantSize that can be folded because the resulting size does not
  // increase.
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  const int64 large_constant_size = kMaxConstantSize + 1;
  Output a = ops::Variable(scope.WithOpName("a"), {1, 1}, DT_FLOAT);
  Output b_const =
      ops::Const(scope.WithOpName("b_const"), 3.14f, {1, large_constant_size});
  Output b = ops::Identity(scope.WithOpName("b"), b_const);
  Output matmul = ops::MatMul(scope.WithOpName("matmul"), a, b);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  item.graph.Swap(&output);
  status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  for (const auto& node : output.node()) {
    if (node.name() == "b") {
      EXPECT_EQ("Const", node.op());
    }
  }
  EXPECT_EQ(4, output.node_size());
  EXPECT_LT(output.ByteSizeLong(), sizeof(float) * large_constant_size + 500);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> fetch_nodes = {"o1", "o2", "p1", "p2"};
  auto a_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({1, 5}));
  auto g_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({1}));
  auto tensors_expected =
      EvaluateNodes(item.graph, fetch_nodes, {{"a", a_t}, {"g", g_t}});
  EXPECT_EQ(fetch_nodes.size(), tensors_expected.size());

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Run a second time to make sure the optimization is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
  for (bool use_reshape : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output input =
        ops::Placeholder(s.WithOpName("input"), DT_FLOAT,
                         ops::Placeholder::Shape(PartialTensorShape({-1, -1})));
    // If use_reshape is false, we need to now the number of indices to apply
    // the rewrite.
    Output indices = ops::Placeholder(
        s.WithOpName("indices"), DT_INT32,
        ops::Placeholder::Shape(PartialTensorShape({use_reshape ? -1 : 2})));
    Output sum = ops::Sum(s.WithOpName("sum"), input, indices);
    if (use_reshape) {
      Output size = ops::Const(s.WithOpName("size"), 1, {1});
      Output reshape = ops::Reshape(s.WithOpName("reshape"), sum, size);
    }

    GrapplerItem item;
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    item.fetch.push_back(use_reshape ? "reshape" : "sum");

    auto input_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({3, 4}));
    Tensor indices_t(DT_INT32, TensorShape({2}));
    indices_t.flat<int>()(0) = 0;
    indices_t.flat<int>()(1) = 1;
    auto tensors_expected = EvaluateNodes(
        item.graph, item.fetch, {{"input", input_t}, {"indices", indices_t}});
    EXPECT_EQ(1, tensors_expected.size());

    // Use aggressive mode to force the shape inference to propagate placeholder
    // shapes.
    ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                              /*cpu_device=*/nullptr);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    TF_EXPECT_OK(status);

    // Run a second time to make sure the optimization is idempotent.
    item.graph.Swap(&output);
    status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    TF_EXPECT_OK(status);

    int found = 0;
    for (const auto& node : output.node()) {
      if (node.name() == "ConstantFolding/sum-reduction_indices") {
        ++found;
        EXPECT_EQ("Const", node.op());
        EXPECT_EQ("^indices", node.input(0));
        EXPECT_EQ(2,
                  TensorShape(node.attr().at("value").tensor().tensor_shape())
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
}

TEST_F(ConstantFoldingTest, MaterializeReductionIndices_NotFullReduction) {
  for (bool input_rank_known : {true, false}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output input =
        (input_rank_known ? ops::Placeholder(s.WithOpName("input"), DT_FLOAT,
                                             ops::Placeholder::Shape(
                                                 PartialTensorShape({-1, -1})))
                          : ops::Placeholder(s.WithOpName("input"), DT_FLOAT));
    Output indices =
        ops::Placeholder(s.WithOpName("indices"), DT_INT32,
                         ops::Placeholder::Shape(
                             PartialTensorShape({input_rank_known ? 1 : 2})));
    Output sum = ops::Sum(s.WithOpName("sum"), input, indices);

    GrapplerItem item;
    TF_CHECK_OK(s.ToGraphDef(&item.graph));
    item.fetch.push_back("sum");

    // Use aggressive mode to force the shape inference to propagate placeholder
    // shapes.
    ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                              /*cpu_device=*/nullptr);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    TF_EXPECT_OK(status);

    CompareGraphs(item.graph, output);
  }
}

TEST_F(ConstantFoldingTest, LargeConstant) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  // Generate a 4k by 4k constant, non-compressible matrix.
  Output mat_diag =
      ops::Const(scope.WithOpName("mat_diag"), 3.14f, TensorShape({1024 * 4}));
  Output mat = ops::Diag(scope.WithOpName("mat"), mat_diag);
  Output out = ops::Identity(scope.WithOpName("out"), mat);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch.push_back("out");

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  // Make sure the diag node hasn't been folded, since it would use too much
  // memory to encode the corresponding constant.
  int found = 0;
  for (const NodeDef& node : output.node()) {
    if (node.name() == "out") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "mat");
      ++found;
    } else if (node.name() == "mat") {
      EXPECT_EQ(node.op(), "Diag");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "mat_diag");
      ++found;
    }
  }
  EXPECT_EQ(found, 2);
  // output should be no longer than the size of the constant "mat_diag"
  // plus a small constant amount for the remaining nodes.
  EXPECT_LT(output.ByteSizeLong(), sizeof(int) * 4 * 1024 + 500);

  auto tensors_expected = EvaluateNodes(item.graph, item.fetch);
  ASSERT_EQ(tensors_expected.size(), 1);
  auto tensors = EvaluateNodes(output, item.fetch);
  ASSERT_EQ(tensors.size(), 1);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  // Evaluate id_false when input tensor is false.
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

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
      if (absl::StartsWith(node.name(), "ConstantFolding/")) {
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
  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
    } else if (absl::StartsWith(node.name(), "ConstantFolding/")) {
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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
  auto stack_no_axis = ops::Stack(scope.WithOpName("stack_no_axis"), {x});

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch = {"stack", "stack_no_axis"};

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);
  EXPECT_EQ(7, output.node_size());
  int found = 0;
  for (const auto& node : output.node()) {
    if (node.name() == "stack") {
      EXPECT_EQ("ExpandDims", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("ConstantFolding/stack_const_axis", node.input(1));
      EXPECT_EQ("^y", node.input(2));
      ++found;
    } else if (node.name() == "stack_no_axis") {
      EXPECT_EQ("ExpandDims", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("ConstantFolding/stack_no_axis_const_axis", node.input(1));
      ++found;
    } else if (node.name() == "ConstantFolding/stack_const_axis") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("^x", node.input(0));
      ++found;
    }
  }
  EXPECT_EQ(found, 3);

  std::vector<string> fetch = {"stack", "stack_no_axis"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(2, tensors_expected.size());
  EXPECT_EQ(2, tensors.size());
  EXPECT_EQ(tensors_expected[0].shape(), tensors[0].shape());
  EXPECT_EQ(tensors_expected[1].shape(), tensors[1].shape());
}

// The test does not evalaute the optimized and original graphs to check if
// their outputs are the same. See b/78233179.
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);
  // Run the optimizer twice to make sure the rewrite is idempotent.
  item.graph.Swap(&output);
  status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
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

TEST_F(ConstantFoldingTest, FoldingPreservesDenormalFlushing) {
  // Multiplying min() with 0.1 gives a denormal without FTZ and zero with FTZ.
  // Make sure constant folding behaves the same way as TensorFlow.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a =
      ops::Const(s.WithOpName("a"), std::numeric_limits<float>::min(), {1});
  Output b = ops::Const(s.WithOpName("b"), 0.1f, {1});
  Output c = ops::Mul(s.WithOpName("c"), a, b);

  GrapplerItem item;
  item.fetch.push_back("c");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(1, output.node_size());

  const NodeDef& node_d = output.node(0);
  EXPECT_EQ("c", node_d.name());
  EXPECT_EQ("Const", node_d.op());

  std::vector<string> fetch = {"c"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  test::ExpectTensorEqual<float>(tensors_expected[0], tensors[0]);
}

TEST_F(ConstantFoldingTest, EvaluatingLargeConstantNoFoldingMergingLoop) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  int size = 10 * 1024 * 1024 / 4 / 2;
  Output nonconst =
      ops::RandomUniform(s.WithOpName("nonconst"), {size, 1}, DT_FLOAT);
  Output const1 = ops::Const(s.WithOpName("const1"), 0.0f, {size, 1});
  Output const2 = ops::Const(s.WithOpName("const2"), 1.0f, {size, 1});
  Output axis = ops::Const(s.WithOpName("axis"), -1, {});
  Output concat1 =
      ops::Concat(s.WithOpName("concat1"), {nonconst, const1}, axis);
  Output result = ops::Concat(s.WithOpName("result"), {concat1, const2}, axis);

  GrapplerItem item;
  item.fetch.push_back("result");
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  std::vector<string> fetch = {"result"};
  auto tensors_expected = EvaluateNodes(item.graph, fetch);
  auto tensors = EvaluateNodes(output, fetch);
  EXPECT_EQ(1, tensors_expected.size());
  EXPECT_EQ(1, tensors.size());
  EXPECT_EQ(tensors_expected[0].shape(), tensors[0].shape());
}

class ConstantFoldingCastConstTest : public GrapplerTest {
 protected:
  void ConstantFoldingCastConst(bool fetch_const, bool fetch_cast,
                                bool fetch_const_child, bool fetch_cast_child) {
    if (!fetch_const && !fetch_cast && !fetch_const_child &&
        !fetch_cast_child) {
      return;
    }

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    CreateCastConstGraph(s);
    GrapplerItem item;
    int expected_output_size = SetFetch(&item, fetch_const, fetch_cast,
                                        fetch_const_child, fetch_cast_child);
    TF_CHECK_OK(s.ToGraphDef(&item.graph));

    GraphDef output = ConstantFoldingOptimize(item);
    EXPECT_EQ(expected_output_size, output.node_size());

    EvaluateAndCompareUnoptimized(item.graph, output, item.fetch);
  }

 private:
  void CreateCastConstGraph(const tensorflow::Scope& s) {
    Output const1 = ops::Const(s.WithOpName("const1"), 2, {5, 5});
    Output cast = ops::Cast(s.WithOpName("cast"), const1, DT_FLOAT);
    Output const1_child = ops::Identity(s.WithOpName("const1_child"), const1);
    Output cast_child = ops::Identity(s.WithOpName("cast_child"), cast);
  }

  int SetFetch(GrapplerItem* item, bool fetch_const, bool fetch_cast,
               bool fetch_const_child, bool fetch_cast_child) {
    int expected_output_size = 0;
    if (fetch_const) {
      item->fetch.push_back("const1");
      expected_output_size++;
    }
    if (fetch_cast) {
      item->fetch.push_back("cast");
      expected_output_size++;
    }
    if (fetch_const_child) {
      item->fetch.push_back("const1_child");
      expected_output_size++;
    }
    if (fetch_cast_child) {
      item->fetch.push_back("cast_child");
      expected_output_size++;
    }
    return expected_output_size;
  }

  GraphDef ConstantFoldingOptimize(const GrapplerItem& item) {
    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    TF_EXPECT_OK(status);
    return output;
  }

  void EvaluateAndCompareUnoptimized(const GraphDef& unoptimized_graph,
                                     const GraphDef& optimized_graph,
                                     const std::vector<string>& fetch_nodes) {
    auto tensors_expected = EvaluateNodes(unoptimized_graph, fetch_nodes);
    auto tensors = EvaluateNodes(optimized_graph, fetch_nodes);
    ASSERT_EQ(fetch_nodes.size(), tensors_expected.size());
    ASSERT_EQ(fetch_nodes.size(), tensors.size());
    for (int i = 0; i < fetch_nodes.size(); i++) {
      if (fetch_nodes[i] == "const1" || fetch_nodes[i] == "const1_child") {
        test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
      } else {
        test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
      }
    }
  }
};

TEST_F(ConstantFoldingCastConstTest, CastConstFolding) {
  for (bool fetch_const : {false, true}) {
    for (bool fetch_cast : {false, true}) {
      for (bool fetch_const_child : {false, true}) {
        for (bool fetch_cast_child : {false, true}) {
          ConstantFoldingCastConst(fetch_const, fetch_cast, fetch_const_child,
                                   fetch_cast_child);
        }
      }
    }
  }
}

TEST_F(ConstantFoldingTest, MaterializeConstantValuedNode) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output x =
      ops::Placeholder(scope.WithOpName("x"), DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({1, 2, 3, 4})));
  Output ones_like = ops::OnesLike(scope.WithOpName("ones_like"), x);
  Output zeros_like = ops::ZerosLike(scope.WithOpName("zeros_like"), x);
  Output fill = ops::Fill(scope.WithOpName("fill"), {4, 3, 2, 1}, 42);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch = {"ones_like", "zeros_like", "fill"};
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({1, 2, 3, 4}));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, {{"x", x_t}});

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(output.node_size(), 6);
  for (const auto& node : output.node()) {
    if (node.name() != "x") {
      EXPECT_EQ(node.op(), "Const");
    }
    if (node.name() == "ones_like" || node.name() == "zeros_like") {
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "^x");
    }
    if (node.name() == "fill") {
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0)[0], '^');
      EXPECT_EQ(node.input(1)[0], '^');
    }
  }
  auto tensors = EvaluateNodes(output, item.fetch, {{"x", x_t}});
  ASSERT_EQ(item.fetch.size(), tensors.size());
  ASSERT_EQ(tensors_expected.size(), tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    if (item.fetch[i] == "fill") {
      test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
    } else {
      test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
    }
  }
}

TEST_F(ConstantFoldingTest, MaterializeConstantValuedNodeDisableCompression) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Output x =
      ops::Placeholder(scope.WithOpName("x"), DT_FLOAT,
                       ops::Placeholder::Shape(TensorShape({1, 2, 3, 4})));
  Output ones_like = ops::OnesLike(scope.WithOpName("ones_like"), x);
  Output zeros_like = ops::ZerosLike(scope.WithOpName("zeros_like"), x);
  Output fill = ops::Fill(scope.WithOpName("fill"), {4, 3, 2, 1}, 42);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch = {"ones_like", "zeros_like", "fill"};
  auto x_t = GenerateRandomTensor<DT_FLOAT>(TensorShape({1, 2, 3, 4}));
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, {{"x", x_t}});

  ConstantFolding optimizer(/*cpu_device=*/nullptr, true);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(output.node_size(), 6);
  for (const auto& node : output.node()) {
    if (node.name() == "ones_like") {
      EXPECT_EQ(node.op(), "OnesLike");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "x");
    }
    if (node.name() == "zeros_like") {
      EXPECT_EQ(node.op(), "ZerosLike");
      ASSERT_EQ(node.input_size(), 1);
      EXPECT_EQ(node.input(0), "x");
    }
    if (node.name() == "fill") {
      EXPECT_EQ(node.op(), "Fill");
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "Const/Const");
      EXPECT_EQ(node.input(1), "Const_1/Const");
    }
  }
  auto tensors = EvaluateNodes(output, item.fetch, {{"x", x_t}});
  ASSERT_EQ(item.fetch.size(), tensors.size());
  ASSERT_EQ(tensors_expected.size(), tensors.size());
  for (int i = 0; i < tensors.size(); i++) {
    if (item.fetch[i] == "fill") {
      test::ExpectTensorEqual<int>(tensors_expected[i], tensors[i]);
    } else {
      test::ExpectTensorEqual<float>(tensors_expected[i], tensors[i]);
    }
  }
}

TEST_F(ConstantFoldingTest, MaterializeConstantValuedNodeHugeFill) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output value = ops::Const(scope.WithOpName("value"), 42, {});
  Output shape_const = ops::Const(scope.WithOpName("shape"),
                                  {1024, 1024, 1024, 1024, 1024}, {5});
  Output fill_huge =
      ops::Fill(scope.WithOpName("fill_huge"), shape_const, value);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  // Manually convert the input value format to tensor_content to test this
  // case.
  NodeDef* node = item.graph.mutable_node(0);
  ASSERT_EQ(node->name(), "value");
  TensorProto* t = (*node->mutable_attr())["value"].mutable_tensor();
  t->clear_int_val();
  int val = 42;
  port::CopyFromArray(t->mutable_tensor_content(),
                      reinterpret_cast<const char*>(&val), sizeof(int));
  item.fetch = {"fill_huge"};
  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(output.node_size(), 3);
  for (const auto& node : output.node()) {
    EXPECT_EQ(node.op(), "Const");
    if (node.name() == "fill_huge") {
      ASSERT_EQ(node.input_size(), 2);
      EXPECT_EQ(node.input(0), "^shape");
      EXPECT_EQ(node.input(1), "^value");
    }
  }
}

TEST_F(ConstantFoldingTest, BitcastDenormalFloats) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  Tensor x_t(DT_INT64, TensorShape({2, 2}));
  x_t.flat<int64>()(0) = 9223372036854775807L;
  x_t.flat<int64>()(1) = 1L;
  x_t.flat<int64>()(2) = 9223372036854775807L;
  x_t.flat<int64>()(3) = 1L;
  Output x = ops::Const(scope.WithOpName("x"), x_t);
  Output y = ops::Bitcast(scope.WithOpName("y"), x, DT_FLOAT);
  Output z = ops::Bitcast(scope.WithOpName("z"), y, DT_INT64);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch = {"z"};
  auto tensors_expected = EvaluateNodes(item.graph, item.fetch, {});

  ConstantFolding optimizer(/*cpu_device=*/nullptr);
  GraphDef output;
  Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
  TF_EXPECT_OK(status);

  ASSERT_EQ(output.node_size(), 1);
  const NodeDef& node = output.node(0);
  EXPECT_EQ(node.name(), "z");
  EXPECT_EQ(node.op(), "Const");

  auto tensors = EvaluateNodes(output, item.fetch, {});
  ASSERT_EQ(tensors.size(), 1);
  ASSERT_EQ(tensors_expected.size(), 1);
  test::ExpectTensorEqual<int64>(tensors[0], tensors_expected[0]);
}

TEST_F(ConstantFoldingTest, SimplifyCase) {
  using test::function::NDef;

  for (int index = 0; index < 2; ++index) {
    // Build a graph to compute y = Case(index, x, XTimesTwo(x), NonZero(x))
    GrapplerItem item;
    constexpr char kDevice[] = "/job:localhost/replica:0/task:0/device:CPU:0";
    AttrValue branches;
    auto* f = branches.mutable_list()->add_func();
    f->set_name("XTimesTwo");
    (*f->mutable_attr())["T"].set_type(DT_FLOAT);
    auto* g = branches.mutable_list()->add_func();
    *g = *f;
    g->set_name("NonZero");

    // Add a pair of somewhat arbitrary output shapes to
    // test that they are correctly propagates to the _output_shapes
    // attribute.
    AttrValue output_shapes;
    // The first shape is a scalar.
    output_shapes.mutable_list()->add_shape();
    // The second shape is unknown.
    TensorShapeProto* g_shape = output_shapes.mutable_list()->add_shape();
    g_shape->set_unknown_rank(true);

    const Tensor kZero = test::AsScalar<int32>(0);
    const Tensor kOne = test::AsScalar<int32>(1);
    item.graph = test::function::GDef(
        {NDef("one", "Const", {},
              {{"value", index == 0 ? kZero : kOne}, {"dtype", DT_INT32}},
              kDevice),
         NDef("x", "Placeholder", {}, {{"dtype", DT_FLOAT}}, kDevice),
         NDef("case", "Case", {"one", "x"},
              {{"Tin", DataTypeSlice{DT_FLOAT}},
               {"Tout", DataTypeSlice{DT_FLOAT}},
               {"branches", branches},
               {"output_shapes", output_shapes}},
              kDevice),
         NDef("y", "Identity", {"case"}, {{"T", DT_FLOAT}}, kDevice)},
        // FunctionLib
        {
            test::function::XTimesTwo(),
            test::function::NonZero(),
        });
    VLOG(1) << "Before: " << item.graph.DebugString();

    item.fetch = {"y"};
    const Tensor kTwo = test::AsScalar<float>(2.0f);
    auto tensors_expected =
        EvaluateNodes(item.graph, item.fetch, {{"x", kTwo}});

    ConstantFolding optimizer(/*cpu_device=*/nullptr);
    GraphDef optimized_graph;
    TF_ASSERT_OK(
        optimizer.Optimize(/*cluster=*/nullptr, item, &optimized_graph));
    VLOG(1) << "After: " << optimized_graph.DebugString();

    int pco_count = 0;
    for (const auto& node : optimized_graph.node()) {
      EXPECT_NE(node.op(), "Case");
      if (node.op() == "PartitionedCall") {
        ++pco_count;
        const auto& shape_list = node.attr().at("_output_shapes").list();
        ASSERT_EQ(shape_list.shape_size(), 1);
        EXPECT_EQ(shape_list.shape(0).dim_size(), 0);
        if (index == 0) {
          EXPECT_EQ(node.attr().at("f").func().name(), "XTimesTwo");
          EXPECT_EQ(shape_list.shape(0).unknown_rank(), false);
        } else {
          EXPECT_EQ(node.attr().at("f").func().name(), "NonZero");
          EXPECT_EQ(shape_list.shape(0).unknown_rank(), true);
        }
      }
    }
    EXPECT_EQ(pco_count, 1);

    auto tensors = EvaluateNodes(optimized_graph, item.fetch, {{"x", kTwo}});
    ASSERT_EQ(tensors.size(), tensors_expected.size());
    test::ExpectTensorEqual<float>(tensors[0], tensors_expected[0]);
  }
}

TEST_F(ConstantFoldingTest, SimplifySelect) {
  for (bool scalar_pred : {true, false}) {
    for (bool pred_val : {true, false}) {
      tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
      std::unique_ptr<Tensor> if_t;
      if (scalar_pred) {
        if_t.reset(new Tensor(DT_BOOL, TensorShape()));
      } else {
        if_t.reset(new Tensor(DT_BOOL, TensorShape({2, 2})));
      }
      for (int i = 0; i < (scalar_pred ? 1 : 4); ++i) {
        if_t->flat<bool>()(i) = pred_val;
      }
      Output if_ = ops::Const(scope.WithOpName("if"), *if_t);
      Output then_ =
          ops::Placeholder(scope.WithOpName("then"), DT_FLOAT,
                           ops::Placeholder::Shape(TensorShape({2, 2})));
      Output else_ =
          ops::Placeholder(scope.WithOpName("else"), DT_FLOAT,
                           ops::Placeholder::Shape(TensorShape({2, 2})));
      Output select =
          ops::SelectV2(scope.WithOpName("select"), if_, then_, else_);
      Output id = ops::Identity(scope.WithOpName("id"), select);

      GrapplerItem item;
      TF_CHECK_OK(scope.ToGraphDef(&item.graph));
      item.fetch = {"id"};

      const Tensor kOne =
          test::AsTensor<float>({1.0f, 1.0f, 1.0f, 1.0f}, TensorShape({2, 2}));
      const Tensor kTwo =
          test::AsTensor<float>({2.0f, 2.0f, 2.0f, 2.0f}, TensorShape({2, 2}));
      auto tensors_expected = EvaluateNodes(item.graph, item.fetch,
                                            {{"then", kOne}, {"else", kTwo}});

      // Use aggressive mode to force the shape inference to propagate
      // placeholder shapes.
      ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                                /*cpu_device=*/nullptr);
      GraphDef optimized_graph;
      TF_EXPECT_OK(
          optimizer.Optimize(/*cluster=*/nullptr, item, &optimized_graph));

      ASSERT_EQ(optimized_graph.node_size(), 5);
      bool found = false;
      for (const auto& node : optimized_graph.node()) {
        if (node.name() == "select") {
          found = true;
          EXPECT_EQ(node.op(), "Identity");
          ASSERT_EQ(node.input_size(), 3);
          EXPECT_EQ(node.input(0), pred_val ? "then" : "else");
          EXPECT_EQ(node.input(1), pred_val ? "^if" : "^then");
          EXPECT_EQ(node.input(2), pred_val ? "^else" : "^if");
        }
      }
      EXPECT_TRUE(found);

      auto tensors = EvaluateNodes(optimized_graph, item.fetch,
                                   {{"then", kOne}, {"else", kTwo}});
      ASSERT_EQ(tensors.size(), 1);
      ASSERT_EQ(tensors_expected.size(), 1);
      test::ExpectTensorEqual<float>(tensors[0], tensors_expected[0]);
    }
  }
}

TEST_F(ConstantFoldingTest, SimplifySelect_BroadcastTo) {
  for (TensorShape pred_shape : {TensorShape{2, 1}, TensorShape{2, 2, 1}}) {
    for (bool pred_val : {true, false}) {
      tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
      std::unique_ptr<Tensor> if_t;
      if_t.reset(new Tensor(DT_BOOL, pred_shape));
      for (int i = 0; i < pred_shape.num_elements(); ++i) {
        if_t->flat<bool>()(i) = pred_val;
      }
      Output if_ = ops::Const(scope.WithOpName("if"), *if_t);
      Output then_ =
          ops::Placeholder(scope.WithOpName("then"), DT_FLOAT,
                           ops::Placeholder::Shape(TensorShape({2, 1})));
      Output else_ =
          ops::Placeholder(scope.WithOpName("else"), DT_FLOAT,
                           ops::Placeholder::Shape(TensorShape({2, 4})));
      Output select =
          ops::SelectV2(scope.WithOpName("select"), if_, then_, else_);
      Output id = ops::Identity(scope.WithOpName("id"), select);

      GrapplerItem item;
      TF_CHECK_OK(scope.ToGraphDef(&item.graph));
      item.fetch = {"id"};

      const Tensor kOne =
          test::AsTensor<float>({1.0f, 1.0f}, TensorShape({2, 1}));
      const Tensor kTwo = test::AsTensor<float>(
          {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
          TensorShape({2, 4}));
      auto tensors_expected = EvaluateNodes(item.graph, item.fetch,
                                            {{"then", kOne}, {"else", kTwo}});

      // Use aggressive mode to force the shape inference to propagate
      // placeholder shapes.
      ConstantFolding optimizer(RewriterConfig::AGGRESSIVE,
                                /*cpu_device=*/nullptr);
      GraphDef optimized_graph;
      TF_EXPECT_OK(
          optimizer.Optimize(/*cluster=*/nullptr, item, &optimized_graph));

      ASSERT_EQ(optimized_graph.node_size(), 6);
      bool found = false;
      for (const auto& node : optimized_graph.node()) {
        if (node.name() == "select") {
          found = true;
          EXPECT_EQ(node.op(), "BroadcastTo");
          ASSERT_EQ(node.input_size(), 4);
          EXPECT_EQ(node.input(0), pred_val ? "then" : "else");
          EXPECT_EQ(node.input(1),
                    strings::StrCat("ConstantFolding/select-broadcastto_shape-",
                                    pred_val ? 1 : 2));
          EXPECT_EQ(node.input(2), pred_val ? "^else" : "^if");
          EXPECT_EQ(node.input(3), pred_val ? "^if" : "^then");
        }
      }
      EXPECT_TRUE(found);

      auto tensors = EvaluateNodes(optimized_graph, item.fetch,
                                   {{"then", kOne}, {"else", kTwo}});
      ASSERT_EQ(tensors.size(), 1);
      ASSERT_EQ(tensors_expected.size(), 1);
      ASSERT_EQ(tensors[0].shape(), pred_shape.num_elements() == 2
                                        ? TensorShape({2, 4})
                                        : TensorShape({2, 2, 4}));
      test::ExpectTensorEqual<float>(tensors[0], tensors_expected[0]);
    }
  }
}

TEST_F(ConstantFoldingTest, QuantizationEmulation) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  Output x = ops::Const(scope.WithOpName("x"), {0.0f, 1.0f, 2.0f, 3.0f}, {4});
  Output min_range = ops::Const(scope.WithOpName("min_range"), 0.0f, {});
  Output max_range = ops::Const(scope.WithOpName("max_range"), 3.0f, {});
  Output y = ops::QuantizeAndDequantizeV2(scope.WithOpName("y"), x, min_range,
                                          max_range);
  Output id = ops::Identity(scope.WithOpName("id"), y);

  GrapplerItem item;
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  item.fetch = {"id"};

  std::vector<Tensor> expected_tensors = EvaluateNodes(item.graph, item.fetch);

  for (const bool fold_quantization_emulation : {false, true}) {
    ConstantFolding optimizer(/*cpu_device=*/nullptr,
                              /*disable_compressed_tensor_optimization=*/false,
                              fold_quantization_emulation);
    GraphDef output;
    Status status = optimizer.Optimize(/*cluster=*/nullptr, item, &output);
    int num_quantization_emulation_ops = 0;
    for (const NodeDef& node : output.node()) {
      if (node.op() == "QuantizeAndDequantizeV2") {
        num_quantization_emulation_ops++;
      }
    }
    EXPECT_EQ(fold_quantization_emulation ? 0 : 1,
              num_quantization_emulation_ops);

    std::vector<Tensor> actual_tensors = EvaluateNodes(output, item.fetch);
    for (int i = 0; i < item.fetch.size(); ++i) {
      test::ExpectTensorEqual<float>(expected_tensors[i], actual_tensors[i]);
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
