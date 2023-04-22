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
Status RenameAttribute(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def);

class RenameAttributeTest : public ::testing::Test {
 protected:
  void TestRenameAttribute() {
    GraphDef graph_def;

    NodeDef* mul_node1 = graph_def.add_node();
    mul_node1->set_name("mul_node1");
    mul_node1->set_op("Mul");
    mul_node1->add_input("add_node2");
    mul_node1->add_input("add_node3");
    AddNodeAttr("foo", 23, mul_node1);
    AddNodeAttr("bar", "something", mul_node1);

    NodeDef* add_node2 = graph_def.add_node();
    add_node2->set_name("add_node2");
    add_node2->set_op("Add");
    add_node2->add_input("const_node1");
    add_node2->add_input("const_node2");
    AddNodeAttr("foo", 46, add_node2);
    AddNodeAttr("bob", 23, add_node2);
    AddNodeAttr("bar", "something else", add_node2);

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

    NodeDef* add_node4 = graph_def.add_node();
    add_node4->set_name("add_node4");
    add_node4->set_op("Add");
    add_node4->add_input("add_node2");
    add_node4->add_input("add_node3");

    GraphDef wildcard_result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"mul_node1"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"op_name", {string("*")}}));
    context.params.insert(std::pair<string, std::vector<string>>(
        {"old_attribute_name", {string("foo")}}));
    context.params.insert(std::pair<string, std::vector<string>>(
        {"new_attribute_name", {string("baz")}}));
    TF_ASSERT_OK(RenameAttribute(graph_def, context, &wildcard_result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(wildcard_result, &node_lookup);
    EXPECT_EQ(0, node_lookup.at("mul_node1")->attr().count("foo"));
    EXPECT_EQ(1, node_lookup.at("mul_node1")->attr().count("baz"));
    EXPECT_EQ(1, node_lookup.at("mul_node1")->attr().count("bar"));
    EXPECT_EQ(0, node_lookup.at("add_node2")->attr().count("foo"));
    EXPECT_EQ(1, node_lookup.at("add_node2")->attr().count("baz"));
    EXPECT_EQ(1, node_lookup.at("add_node2")->attr().count("bar"));
    EXPECT_EQ(1, node_lookup.at("add_node2")->attr().count("bob"));

    GraphDef targeted_result;
    TransformFuncContext targeted_context;
    targeted_context.input_names = {};
    targeted_context.output_names = {"mul_node1"};
    targeted_context.params.insert(
        std::pair<string, std::vector<string>>({"op_name", {string("Mul")}}));
    targeted_context.params.insert(std::pair<string, std::vector<string>>(
        {"old_attribute_name", {string("foo")}}));
    targeted_context.params.insert(std::pair<string, std::vector<string>>(
        {"new_attribute_name", {string("baz")}}));
    TF_ASSERT_OK(
        RenameAttribute(graph_def, targeted_context, &targeted_result));

    MapNamesToNodes(targeted_result, &node_lookup);
    EXPECT_EQ(0, node_lookup.at("mul_node1")->attr().count("foo"));
    EXPECT_EQ(1, node_lookup.at("mul_node1")->attr().count("baz"));
    EXPECT_EQ(1, node_lookup.at("mul_node1")->attr().count("bar"));
    EXPECT_EQ(1, node_lookup.at("add_node2")->attr().count("foo"));
    EXPECT_EQ(0, node_lookup.at("add_node2")->attr().count("baz"));
    EXPECT_EQ(1, node_lookup.at("add_node2")->attr().count("bar"));
    EXPECT_EQ(1, node_lookup.at("add_node2")->attr().count("bob"));
  }
};

TEST_F(RenameAttributeTest, TestRenameAttribute) { TestRenameAttribute(); }

}  // namespace graph_transforms
}  // namespace tensorflow
