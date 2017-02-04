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
Status RemoveDevice(const GraphDef& input_graph_def,
                    const TransformFuncContext& context,
                    GraphDef* output_graph_def);

class RemoveDeviceTest : public ::testing::Test {
 protected:
  void TestRemoveDevice() {
    GraphDef graph_def;

    NodeDef* mul_node1 = graph_def.add_node();
    mul_node1->set_name("mul_node1");
    mul_node1->set_op("Mul");
    mul_node1->set_device("//cpu:0");
    mul_node1->add_input("add_node2");
    mul_node1->add_input("add_node3");

    NodeDef* add_node2 = graph_def.add_node();
    add_node2->set_name("add_node2");
    add_node2->set_op("Add");
    add_node2->add_input("const_node1");
    add_node2->add_input("const_node2");
    add_node2->set_device("//gpu:1");

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

    GraphDef result;
    TransformFuncContext context;
    context.input_names = {};
    context.output_names = {"mul_node1"};
    TF_ASSERT_OK(RemoveDevice(graph_def, context, &result));

    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);
    EXPECT_EQ("", node_lookup.at("mul_node1")->device());
    EXPECT_EQ("", node_lookup.at("add_node2")->device());
  }
};

TEST_F(RemoveDeviceTest, TestRemoveDevice) { TestRemoveDevice(); }

}  // namespace graph_transforms
}  // namespace tensorflow
