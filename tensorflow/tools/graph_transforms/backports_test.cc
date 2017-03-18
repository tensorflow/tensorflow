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
Status BackportConcatV2Transform(const GraphDef& input_graph_def,
                                 const TransformFuncContext& context,
                                 GraphDef* output_graph_def);

class BackportConcatV2Test : public ::testing::Test {
 protected:
  void TestBackportConcatV2() {
    GraphDef graph_def;

    NodeDef* mul_node1 = graph_def.add_node();
    mul_node1->set_name("mul_node1");
    mul_node1->set_op("Mul");
    mul_node1->add_input("add_node2");
    mul_node1->add_input("add_node3");

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

    NodeDef* concat_node = graph_def.add_node();
    concat_node->set_name("concat_node");
    concat_node->set_op("ConcatV2");
    concat_node->add_input("const_node1");
    concat_node->add_input("const_node2");
    concat_node->add_input("const_node3");
    SetNodeAttr("Tidx", DT_INT32, concat_node);

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"concat_node"};
    TF_ASSERT_OK(BackportConcatV2Transform(graph_def, context, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ(1, node_lookup.count("concat_node"));
    EXPECT_EQ("Concat", node_lookup.at("concat_node")->op());
    EXPECT_EQ(0, node_lookup.at("concat_node")->attr().count("Tidx"));
    EXPECT_EQ("const_node3", node_lookup.at("concat_node")->input(0));
    EXPECT_EQ("const_node1", node_lookup.at("concat_node")->input(1));
    EXPECT_EQ("const_node2", node_lookup.at("concat_node")->input(2));
    EXPECT_EQ(1, node_lookup.count("const_node1"));
    EXPECT_EQ("Const", node_lookup.at("const_node1")->op());
    EXPECT_EQ(1, node_lookup.count("const_node2"));
    EXPECT_EQ("Const", node_lookup.at("const_node2")->op());
    EXPECT_EQ(1, node_lookup.count("const_node3"));
    EXPECT_EQ("Const", node_lookup.at("const_node3")->op());
  }
};

TEST_F(BackportConcatV2Test, TestBackportConcatV2) { TestBackportConcatV2(); }

}  // namespace graph_transforms
}  // namespace tensorflow
