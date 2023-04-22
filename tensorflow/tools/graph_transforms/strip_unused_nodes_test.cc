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
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declare here, so we don't need a public header.
Status StripUnusedNodes(const GraphDef& input_graph_def,
                        const TransformFuncContext& context,
                        GraphDef* output_graph_def);

class StripUnusedNodesTest : public ::testing::Test {
 protected:
  void TestSimpleAdd() {
    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("a_node");
    add_node->add_input("b_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(graph_def, {{}, {"add_node"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ(1, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }

  void TestCommonAncestor() {
    GraphDef graph_def;

    NodeDef* add_node1 = graph_def.add_node();
    add_node1->set_name("add_node1");
    add_node1->set_op("Add");
    add_node1->add_input("add_node2");
    add_node1->add_input("add_node3");

    NodeDef* add_node2 = graph_def.add_node();
    add_node2->set_name("add_node2");
    add_node2->set_op("Add");
    add_node2->add_input("const_node1");
    add_node2->add_input("const_node2");

    NodeDef* add_node3 = graph_def.add_node();
    add_node3->set_name("add_node3");
    add_node3->set_op("Add");
    add_node3->add_input("const_node1");
    add_node3->add_input("const_node3");

    NodeDef* const_node1 = graph_def.add_node();
    const_node1->set_name("const_node1");
    const_node1->set_op("Const");

    NodeDef* const_node2 = graph_def.add_node();
    const_node2->set_name("const_node2");
    const_node2->set_op("Const");

    NodeDef* const_node3 = graph_def.add_node();
    const_node3->set_name("const_node3");
    const_node3->set_op("Const");

    NodeDef* dangling_input = graph_def.add_node();
    dangling_input->set_name("dangling_input");
    dangling_input->set_op("Const");

    NodeDef* add_node4 = graph_def.add_node();
    add_node4->set_name("add_node4");
    add_node4->set_op("Add");
    add_node4->add_input("add_node2");
    add_node4->add_input("add_node3");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(
        graph_def, {{"dangling_input"}, {"add_node1"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node1"));
    EXPECT_EQ(1, node_lookup.count("add_node2"));
    EXPECT_EQ(1, node_lookup.count("add_node3"));
    EXPECT_EQ(0, node_lookup.count("add_node4"));
    EXPECT_EQ(1, node_lookup.count("const_node1"));
    EXPECT_EQ(1, node_lookup.count("const_node2"));
    EXPECT_EQ(1, node_lookup.count("const_node3"));
    EXPECT_EQ(0, node_lookup.count("const_node4"));
    EXPECT_EQ(1, node_lookup.count("dangling_input"));
  }

  void TestSimplePlaceholder() {
    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("mul_node");
    add_node->add_input("a_node");

    NodeDef* mul_node = graph_def.add_node();
    mul_node->set_name("mul_node");
    mul_node->set_op("Mul");
    mul_node->add_input("b_node");
    mul_node->add_input("c_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(
        StripUnusedNodes(graph_def, {{"mul_node"}, {"add_node"}}, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("mul_node"));
    EXPECT_EQ("Placeholder", node_lookup["mul_node"]->op());
    EXPECT_EQ(DT_FLOAT, node_lookup["mul_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({}),
              TensorShape(node_lookup["mul_node"]->attr().at("shape").shape()));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ(0, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }

  void TestPlaceholderDefaultArgs() {
    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("mul_node");
    add_node->add_input("a_node");

    NodeDef* mul_node = graph_def.add_node();
    mul_node->set_name("mul_node");
    mul_node->set_op("Mul");
    mul_node->add_input("b_node");
    mul_node->add_input("c_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(graph_def,
                                  {{"mul_node"},
                                   {"add_node"},
                                   {{"type", {"int32"}}, {"shape", {"1,2,3"}}}},
                                  &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("mul_node"));
    EXPECT_EQ("Placeholder", node_lookup["mul_node"]->op());
    EXPECT_EQ(DT_INT32, node_lookup["mul_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({1, 2, 3}),
              TensorShape(node_lookup["mul_node"]->attr().at("shape").shape()));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ(0, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }

  void TestPlaceholderNamedArgs() {
    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("mul_node");
    add_node->add_input("a_node");

    NodeDef* mul_node = graph_def.add_node();
    mul_node->set_name("mul_node");
    mul_node->set_op("Mul");
    mul_node->add_input("b_node");
    mul_node->add_input("c_node");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* c_node = graph_def.add_node();
    c_node->set_name("c_node");
    c_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(StripUnusedNodes(graph_def,
                                  {{"mul_node", "a_node"},
                                   {"add_node"},
                                   {{"name", {"a_node", "mul_node"}},
                                    {"type_for_name", {"int64", "quint8"}},
                                    {"shape_for_name", {"1,2", "1, 2, 3"}}}},
                                  &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ(1, node_lookup.count("mul_node"));
    EXPECT_EQ("Placeholder", node_lookup["mul_node"]->op());
    EXPECT_EQ(DT_QUINT8, node_lookup["mul_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({1, 2, 3}),
              TensorShape(node_lookup["mul_node"]->attr().at("shape").shape()));
    EXPECT_EQ(1, node_lookup.count("a_node"));
    EXPECT_EQ("Placeholder", node_lookup["a_node"]->op());
    EXPECT_EQ(DT_INT64, node_lookup["a_node"]->attr().at("dtype").type());
    EXPECT_EQ(TensorShape({1, 2}),
              TensorShape(node_lookup["a_node"]->attr().at("shape").shape()));
    EXPECT_EQ(0, node_lookup.count("b_node"));
    EXPECT_EQ(0, node_lookup.count("c_node"));
  }
};

TEST_F(StripUnusedNodesTest, TestSimpleAdd) { TestSimpleAdd(); }

TEST_F(StripUnusedNodesTest, TestCommonAncestor) { TestCommonAncestor(); }

TEST_F(StripUnusedNodesTest, TestSimplePlaceholder) { TestSimplePlaceholder(); }

TEST_F(StripUnusedNodesTest, TestPlaceholderDefaultArgs) {
  TestPlaceholderDefaultArgs();
}

TEST_F(StripUnusedNodesTest, TestPlaceholderNamedArgs) {
  TestPlaceholderNamedArgs();
}

}  // namespace graph_transforms
}  // namespace tensorflow
