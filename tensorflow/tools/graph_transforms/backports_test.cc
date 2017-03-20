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
Status BackportTensorArrayV3Transform(const GraphDef& input_graph_def,
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

TEST(BackportTensorArrayV3Test, TestBackportTensorArrayV3) {
  GraphDef graph_def;

  NodeDef* size_node = graph_def.add_node();
  size_node->set_name("size_node");
  size_node->set_op("Const");
  Tensor size_tensor(DT_INT32, {});
  size_tensor.flat<int32>()(0) = 1;
  SetNodeTensorAttr<float>("value", size_tensor, size_node);

  NodeDef* tensor_array_node = graph_def.add_node();
  tensor_array_node->set_name("tensor_array_node");
  tensor_array_node->set_op("TensorArrayV3");
  tensor_array_node->add_input("size_node");
  SetNodeAttr("dtype", DT_FLOAT, tensor_array_node);
  SetNodeAttr("element_shape", TensorShape({1, 2}), tensor_array_node);
  SetNodeAttr("dynamic_size", false, tensor_array_node);
  SetNodeAttr("clear_after_read", true, tensor_array_node);
  SetNodeAttr("tensor_array_name", "some_name", tensor_array_node);

  NodeDef* handle_output_node = graph_def.add_node();
  handle_output_node->set_name("handle_output_node");
  handle_output_node->set_op("Identity");
  handle_output_node->add_input("tensor_array_node:0");

  NodeDef* flow_output_node = graph_def.add_node();
  flow_output_node->set_name("flow_output_node");
  flow_output_node->set_op("Identity");
  flow_output_node->add_input("tensor_array_node:1");

  NodeDef* tensor_array_grad_node = graph_def.add_node();
  tensor_array_grad_node->set_name("tensor_array_grad_node");
  tensor_array_grad_node->set_op("TensorArrayGradV3");
  tensor_array_grad_node->add_input("tensor_array_node:0");
  tensor_array_grad_node->add_input("tensor_array_node:1");
  SetNodeAttr("source", "foo", tensor_array_grad_node);

  NodeDef* grad_handle_output_node = graph_def.add_node();
  grad_handle_output_node->set_name("grad_handle_output_node");
  grad_handle_output_node->set_op("Identity");
  grad_handle_output_node->add_input("tensor_array_grad_node:0");

  NodeDef* grad_flow_output_node = graph_def.add_node();
  grad_flow_output_node->set_name("grad_flow_output_node");
  grad_flow_output_node->set_op("Identity");
  grad_flow_output_node->add_input("tensor_array_grad_node:1");

  GraphDef result;
  TransformFuncContext context;
  context.input_names = {};
  context.output_names = {"handle_output_node", "grad_handle_output_node"};
  TF_ASSERT_OK(BackportTensorArrayV3Transform(graph_def, context, &result));

  std::map<string, const NodeDef*> node_lookup;
  MapNamesToNodes(result, &node_lookup);
  ASSERT_EQ(1, node_lookup.count("tensor_array_node"));
  EXPECT_EQ("TensorArrayV2", node_lookup.at("tensor_array_node")->op());
  EXPECT_EQ("TensorArrayGradV2",
            node_lookup.at("tensor_array_grad_node")->op());

  for (const NodeDef& node : result.node()) {
    for (const string& input : node.input()) {
      EXPECT_NE("tensor_array_node:1", input);
    }
  }
}

TEST(BackportTensorArrayV3Test, TestBackportTensorArrayV3Subtypes) {
  const std::vector<string> v3_ops = {
      "TensorArrayWriteV3",   "TensorArrayReadV3",   "TensorArrayGatherV3",
      "TensorArrayScatterV3", "TensorArrayConcatV3", "TensorArraySplitV3",
      "TensorArraySizeV3",    "TensorArrayCloseV3"};
  for (const string& v3_op : v3_ops) {
    GraphDef graph_def;
    NodeDef* v3_node = graph_def.add_node();
    v3_node->set_name("v3_node");
    v3_node->set_op(v3_op);

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {""};
    TF_ASSERT_OK(BackportTensorArrayV3Transform(graph_def, context, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    ASSERT_EQ(1, node_lookup.count("v3_node"));
    EXPECT_TRUE(StringPiece(node_lookup.at("v3_node")->op()).ends_with("V2"));
  }
}

}  // namespace graph_transforms
}  // namespace tensorflow
