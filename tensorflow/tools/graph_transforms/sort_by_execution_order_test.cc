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

class SortByExecutionOrderTest : public ::testing::Test {
 protected:
  void GetOrder(const GraphDef& graph_def, std::map<string, int>* order) {
    for (int i = 0; i < graph_def.node_size(); ++i) {
      const NodeDef& node = graph_def.node(i);
      (*order)[node.name()] = i;
    }
  }

  void TestSimpleAdd() {
    GraphDef graph_def;
    NodeDef* add_node = graph_def.add_node();
    add_node->set_name("add_node");
    add_node->set_op("Add");
    add_node->add_input("a_node");
    add_node->add_input("b_node");

    NodeDef* b_node = graph_def.add_node();
    b_node->set_name("b_node");
    b_node->set_op("Const");

    NodeDef* a_node = graph_def.add_node();
    a_node->set_name("a_node");
    a_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(2, order["add_node"]);
    EXPECT_GT(2, order["a_node"]);
    EXPECT_GT(2, order["b_node"]);
  }

  void TestSimpleLinear() {
    GraphDef graph_def;

    NodeDef* negative_node = graph_def.add_node();
    negative_node->set_name("negative_node");
    negative_node->set_op("Negative");
    negative_node->add_input("sqrt_node");

    NodeDef* relu_node = graph_def.add_node();
    relu_node->set_name("relu_node");
    relu_node->set_op("Relu");
    relu_node->add_input("const_node");

    NodeDef* sqrt_node = graph_def.add_node();
    sqrt_node->set_name("sqrt_node");
    sqrt_node->set_op("Sqrt");
    sqrt_node->add_input("relu_node");

    NodeDef* const_node = graph_def.add_node();
    const_node->set_name("const_node");
    const_node->set_op("Const");

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(3, order["negative_node"]);
    EXPECT_EQ(2, order["sqrt_node"]);
    EXPECT_EQ(1, order["relu_node"]);
    EXPECT_EQ(0, order["const_node"]);
  }

  void TestSimpleTree() {
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
    add_node3->add_input("const_node3");
    add_node3->add_input("const_node4");

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

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(6, order["add_node1"]);
    EXPECT_GT(6, order["add_node2"]);
    EXPECT_GT(6, order["add_node3"]);
    EXPECT_GT(5, order["const_node1"]);
    EXPECT_GT(5, order["const_node2"]);
    EXPECT_GT(5, order["const_node3"]);
    EXPECT_GT(5, order["const_node4"]);
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

    GraphDef result;
    TF_ASSERT_OK(SortByExecutionOrder(graph_def, &result));

    std::map<string, int> order;
    GetOrder(result, &order);
    EXPECT_EQ(5, order["add_node1"]);
    EXPECT_GT(5, order["add_node2"]);
    EXPECT_GT(5, order["add_node3"]);
    EXPECT_GT(4, order["const_node2"]);
    EXPECT_GT(4, order["const_node3"]);
    EXPECT_GT(3, order["const_node1"]);
  }
};

TEST_F(SortByExecutionOrderTest, TestSimpleAdd) { TestSimpleAdd(); }

TEST_F(SortByExecutionOrderTest, TestSimpleLinear) { TestSimpleLinear(); }

TEST_F(SortByExecutionOrderTest, TestSimpleTree) { TestSimpleTree(); }

TEST_F(SortByExecutionOrderTest, TestCommonAncestor) { TestCommonAncestor(); }

}  // namespace graph_transforms
}  // namespace tensorflow
