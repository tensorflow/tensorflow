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
absl::Status RemoveNodes(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def);

class RemoveNodesTest : public ::testing::Test {
 protected:
  void TestRemoveNodes() {
    GraphDef graph_def;

    NodeDef* add_node1 = graph_def.add_node();
    add_node1->set_name("add_node1");
    add_node1->set_op("Add");
    add_node1->add_input("add_node2");
    add_node1->add_input("add_node3");

    NodeDef* add_node2 = graph_def.add_node();
    add_node2->set_name("add_node2");
    add_node2->set_op("Add");
    add_node2->add_input("identity_node1");
    add_node2->add_input("identity_node2");

    NodeDef* add_node3 = graph_def.add_node();
    add_node3->set_name("add_node3");
    add_node3->set_op("Add");
    add_node3->add_input("identity_node1");
    add_node3->add_input("const_node3");

    NodeDef* identity_node1 = graph_def.add_node();
    identity_node1->set_name("identity_node1");
    identity_node1->set_op("Identity");
    identity_node1->add_input("const_node1");

    NodeDef* identity_node2 = graph_def.add_node();
    identity_node2->set_name("identity_node2");
    identity_node2->set_op("Identity");
    identity_node2->add_input("const_node2");

    NodeDef* identity_node3 = graph_def.add_node();
    identity_node3->set_name("identity_node3");
    identity_node3->set_op("Identity");
    identity_node3->add_input("const_node3");

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

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node1"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"op", {string("Identity")}}));
    TF_ASSERT_OK(RemoveNodes(graph_def, context, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node1"));
    EXPECT_EQ("add_node2", node_lookup.at("add_node1")->input(0));
    EXPECT_EQ("add_node3", node_lookup.at("add_node1")->input(1));
    EXPECT_EQ(1, node_lookup.count("add_node2"));
    EXPECT_EQ("const_node1", node_lookup.at("add_node2")->input(0));
    EXPECT_EQ("const_node2", node_lookup.at("add_node2")->input(1));
    EXPECT_EQ(1, node_lookup.count("add_node3"));
    EXPECT_EQ("const_node1", node_lookup.at("add_node3")->input(0));
    EXPECT_EQ("const_node3", node_lookup.at("add_node3")->input(1));
    EXPECT_EQ(1, node_lookup.count("add_node4"));
    EXPECT_EQ("add_node2", node_lookup.at("add_node4")->input(0));
    EXPECT_EQ("add_node3", node_lookup.at("add_node4")->input(1));
    EXPECT_EQ(0, node_lookup.count("identity_node1"));
    EXPECT_EQ(0, node_lookup.count("identity_node2"));
    EXPECT_EQ(0, node_lookup.count("identity_node3"));
    EXPECT_EQ(1, node_lookup.count("const_node1"));
    EXPECT_EQ("Const", node_lookup.at("const_node1")->op());
    EXPECT_EQ(1, node_lookup.count("const_node2"));
    EXPECT_EQ("Const", node_lookup.at("const_node2")->op());
    EXPECT_EQ(1, node_lookup.count("const_node3"));
    EXPECT_EQ("Const", node_lookup.at("const_node3")->op());
  }

  void TestRemoveOutputNodes() {
    GraphDef graph_def;

    NodeDef* const_node1 = graph_def.add_node();
    const_node1->set_name("const_node1");
    const_node1->set_op("Const");

    NodeDef* const_node2 = graph_def.add_node();
    const_node2->set_name("const_node2");
    const_node2->set_op("Const");

    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("const_node1");
    add_node->add_input("const_node2");

    NodeDef* identity_node = graph_def.add_node();
    identity_node->set_name("identity_node");
    identity_node->set_op("Identity");
    identity_node->add_input("add_node");

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"identity_node"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"op", {string("Identity")}}));
    TF_ASSERT_OK(RemoveNodes(graph_def, context, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ("const_node1", node_lookup.at("add_node")->input(0));
    EXPECT_EQ("const_node2", node_lookup.at("add_node")->input(1));
    EXPECT_EQ(1, node_lookup.count("identity_node"));
    EXPECT_EQ("add_node", node_lookup.at("identity_node")->input(0));
  }

  void TestRemoveChainedNodes() {
    GraphDef graph_def;

    NodeDef* const_node1 = graph_def.add_node();
    const_node1->set_name("const_node1");
    const_node1->set_op("Const");

    NodeDef* identity_node1 = graph_def.add_node();
    identity_node1->set_name("identity_node1");
    identity_node1->set_op("Identity");
    identity_node1->add_input("const_node1");

    NodeDef* identity_node2 = graph_def.add_node();
    identity_node2->set_name("identity_node2");
    identity_node2->set_op("Identity");
    identity_node2->add_input("identity_node1");

    NodeDef* identity_node3 = graph_def.add_node();
    identity_node3->set_name("identity_node3");
    identity_node3->set_op("Identity");
    identity_node3->add_input("identity_node2");

    NodeDef* const_node2 = graph_def.add_node();
    const_node2->set_name("const_node2");
    const_node2->set_op("Const");

    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("identity_node3");
    add_node->add_input("const_node2");

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"identity_node"};
    context.params.insert(
        std::pair<string, std::vector<string>>({"op", {string("Identity")}}));
    TF_ASSERT_OK(RemoveNodes(graph_def, context, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ("const_node1", node_lookup.at("add_node")->input(0));
    EXPECT_EQ("const_node2", node_lookup.at("add_node")->input(1));
    EXPECT_EQ(0, node_lookup.count("identity_node1"));
    EXPECT_EQ(0, node_lookup.count("identity_node2"));
    EXPECT_EQ(0, node_lookup.count("identity_node3"));
  }

  void TestRemoveMultipleInputs() {
    GraphDef graph_def;

    NodeDef* const_node1 = graph_def.add_node();
    const_node1->set_name("const_node1");
    const_node1->set_op("Const");

    NodeDef* const_node2 = graph_def.add_node();
    const_node2->set_name("const_node2");
    const_node2->set_op("Const");

    NodeDef* const_node3 = graph_def.add_node();
    const_node3->set_name("const_node3");
    const_node3->set_op("Const");

    NodeDef* const_node4 = graph_def.add_node();
    const_node4->set_name("const_node4");
    const_node4->set_op("Const");

    NodeDef* fake_quant_node = graph_def.add_node();
    fake_quant_node->set_name("fake_quant_node");
    fake_quant_node->set_op("FakeQuantWithMinMaxVars");
    fake_quant_node->add_input("const_node1");
    fake_quant_node->add_input("const_node2");
    fake_quant_node->add_input("const_node3");

    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("fake_quant_node");
    add_node->add_input("const_node4");

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"add_node"};
    context.params.insert(std::pair<string, std::vector<string>>(
        {"op", {string("FakeQuantWithMinMaxVars")}}));
    context.params.insert(
        std::pair<string, std::vector<string>>({"max_inputs", {string("3")}}));
    TF_ASSERT_OK(RemoveNodes(graph_def, context, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    ASSERT_EQ(1, node_lookup.count("const_node1"));
    ASSERT_EQ(1, node_lookup.count("const_node4"));
    ASSERT_EQ(0, node_lookup.count("fake_quant_node"));
    ASSERT_EQ(1, node_lookup.count("add_node"));
    EXPECT_EQ("const_node1", node_lookup.at("add_node")->input(0));
    EXPECT_EQ("const_node4", node_lookup.at("add_node")->input(1));
  }
};

TEST_F(RemoveNodesTest, TestRemoveNodes) { TestRemoveNodes(); }

TEST_F(RemoveNodesTest, TestRemoveOutputNodes) { TestRemoveOutputNodes(); }

TEST_F(RemoveNodesTest, TestRemoveChainedNodes) { TestRemoveChainedNodes(); }

TEST_F(RemoveNodesTest, TestRemoveMultipleInputs) {
  TestRemoveMultipleInputs();
}

}  // namespace graph_transforms
}  // namespace tensorflow
