/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
absl::Status InsertLogging(const GraphDef& input_graph_def,
                           const TransformFuncContext& context,
                           GraphDef* output_graph_def);

class InsertLoggingTest : public ::testing::Test {
 protected:
  void CheckGraphCanRun(const GraphDef& graph_def,
                        const std::vector<string>& output_names) {
    std::unique_ptr<Session> session(NewSession(SessionOptions()));
    TF_ASSERT_OK(session->Create(graph_def));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->Run({}, output_names, {}, &outputs));
  }

  void TestInsertLogging() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor const_tensor(DT_FLOAT, TensorShape({10}));
    test::FillIota<float>(&const_tensor, 1.0f);
    Output const_node1 =
        Const(root.WithOpName("const_node1"), Input::Initializer(const_tensor));
    Output const_node2 =
        Const(root.WithOpName("const_node2"), Input::Initializer(const_tensor));
    Output const_node3 =
        Const(root.WithOpName("const_node3"), Input::Initializer(const_tensor));
    Output add_node2 =
        Add(root.WithOpName("add_node2"), const_node1, const_node2);
    Output add_node3 =
        Add(root.WithOpName("add_node3"), const_node1, const_node3);
    Output mul_node1 = Mul(root.WithOpName("mul_node1"), add_node2, add_node3);
    Output add_node4 =
        Add(root.WithOpName("add_node4"), mul_node1, const_node3);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    CheckGraphCanRun(graph_def, {"add_node4"});

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node4"};
    TF_ASSERT_OK(InsertLogging(graph_def, context, &result));

    CheckGraphCanRun(result, {"add_node4"});

    std::unordered_set<string> print_inputs;
    for (const NodeDef& node : result.node()) {
      if (node.op() == "Print") {
        print_inputs.insert(node.input(0));
      }
    }

    EXPECT_EQ(6, print_inputs.size());
    EXPECT_EQ(1, print_inputs.count("mul_node1:0"));
    EXPECT_EQ(1, print_inputs.count("add_node2:0"));
    EXPECT_EQ(1, print_inputs.count("add_node3:0"));
    EXPECT_EQ(0, print_inputs.count("add_node4:0"));
    EXPECT_EQ(1, print_inputs.count("const_node1:0"));
    EXPECT_EQ(1, print_inputs.count("const_node2:0"));
    EXPECT_EQ(1, print_inputs.count("const_node3:0"));
  }

  void TestInsertLoggingByOpType() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor const_tensor(DT_FLOAT, TensorShape({10}));
    test::FillIota<float>(&const_tensor, 1.0f);
    Output const_node1 =
        Const(root.WithOpName("const_node1"), Input::Initializer(const_tensor));
    Output const_node2 =
        Const(root.WithOpName("const_node2"), Input::Initializer(const_tensor));
    Output const_node3 =
        Const(root.WithOpName("const_node3"), Input::Initializer(const_tensor));
    Output add_node2 =
        Add(root.WithOpName("add_node2"), const_node1, const_node2);
    Output add_node3 =
        Add(root.WithOpName("add_node3"), const_node1, const_node3);
    Output mul_node1 = Mul(root.WithOpName("mul_node1"), add_node2, add_node3);
    Output add_node4 =
        Add(root.WithOpName("add_node4"), mul_node1, const_node3);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    CheckGraphCanRun(graph_def, {"add_node4"});

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node4"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"op", {"Mul", "Add"}}));
    TF_ASSERT_OK(InsertLogging(graph_def, context, &result));

    CheckGraphCanRun(result, {"add_node4"});

    std::unordered_set<string> print_inputs;
    for (const NodeDef& node : result.node()) {
      if (node.op() == "Print") {
        print_inputs.insert(node.input(0));
      }
    }

    EXPECT_EQ(3, print_inputs.size());
    EXPECT_EQ(1, print_inputs.count("mul_node1:0"));
    EXPECT_EQ(1, print_inputs.count("add_node2:0"));
    EXPECT_EQ(1, print_inputs.count("add_node3:0"));
    EXPECT_EQ(0, print_inputs.count("add_node4:0"));
    EXPECT_EQ(0, print_inputs.count("const_node1:0"));
    EXPECT_EQ(0, print_inputs.count("const_node2:0"));
    EXPECT_EQ(0, print_inputs.count("const_node3:0"));
  }

  void TestInsertLoggingByPrefix() {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    Tensor const_tensor(DT_FLOAT, TensorShape({10}));
    test::FillIota<float>(&const_tensor, 1.0f);
    Output const_node1 =
        Const(root.WithOpName("const_node1"), Input::Initializer(const_tensor));
    Output const_node2 =
        Const(root.WithOpName("const_node2"), Input::Initializer(const_tensor));
    Output const_node3 =
        Const(root.WithOpName("const_node3"), Input::Initializer(const_tensor));
    Output add_node2 =
        Add(root.WithOpName("add_node2"), const_node1, const_node2);
    Output add_node3 =
        Add(root.WithOpName("add_node3"), const_node1, const_node3);
    Output mul_node1 = Mul(root.WithOpName("mul_node1"), add_node2, add_node3);
    Output add_node4 =
        Add(root.WithOpName("add_node4"), mul_node1, const_node3);
    GraphDef graph_def;
    TF_ASSERT_OK(root.ToGraphDef(&graph_def));
    CheckGraphCanRun(graph_def, {"add_node4"});

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node4"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"prefix", {"add_node"}}));
    TF_ASSERT_OK(InsertLogging(graph_def, context, &result));

    CheckGraphCanRun(result, {"add_node4"});

    std::unordered_set<string> print_inputs;
    for (const NodeDef& node : result.node()) {
      if (node.op() == "Print") {
        print_inputs.insert(node.input(0));
      }
    }

    EXPECT_EQ(2, print_inputs.size());
    EXPECT_EQ(0, print_inputs.count("mul_node1:0"));
    EXPECT_EQ(1, print_inputs.count("add_node2:0"));
    EXPECT_EQ(1, print_inputs.count("add_node3:0"));
    EXPECT_EQ(0, print_inputs.count("add_node4:0"));
    EXPECT_EQ(0, print_inputs.count("const_node1:0"));
    EXPECT_EQ(0, print_inputs.count("const_node2:0"));
    EXPECT_EQ(0, print_inputs.count("const_node3:0"));
  }
};

TEST_F(InsertLoggingTest, TestInsertLogging) { TestInsertLogging(); }

TEST_F(InsertLoggingTest, TestInsertLoggingByOpType) {
  TestInsertLoggingByOpType();
}

TEST_F(InsertLoggingTest, TestInsertLoggingByPrefix) {
  TestInsertLoggingByPrefix();
}

}  // namespace graph_transforms
}  // namespace tensorflow
