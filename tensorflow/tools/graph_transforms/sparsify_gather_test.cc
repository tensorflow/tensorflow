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
Status SparsifyGather(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def);

class SparsifyGatherTest : public ::testing::Test {
 protected:
  NodeDef* CreateNode(const string& name, const string& op,
                      const std::vector<NodeDef*>& inputs,
                      GraphDef* graph_def) {
    NodeDef* node_def = graph_def->add_node();
    node_def->set_name(name);
    node_def->set_op(op);
    std::for_each(inputs.begin(), inputs.end(), [&node_def](NodeDef* input) {
      node_def->add_input(input->name());
    });
    return node_def;
  }

  void TestSinglePartitionConst() {
    GraphDef graph_def;

    // Build the graph.
    NodeDef* input_node = CreateNode("ids", "Const", {}, &graph_def);
    NodeDef* const_node = CreateNode("const", "Const", {}, &graph_def);
    SetNodeAttr("dtype", DT_FLOAT, const_node);
    // Set 'Const' node value.
    Tensor weights(DT_FLOAT, TensorShape({4, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001, 1.2, 0.001});
    SetNodeTensorAttr<float>("value", weights, const_node);

    NodeDef* identity_node =
        CreateNode("const/read", "Identity", {const_node}, &graph_def);
    CreateNode("gather", "Gather", {identity_node, input_node}, &graph_def);
    CreateNode("group_deps", "NoOp", {}, &graph_def);

    // Run the op.
    GraphDef result;
    TransformFuncContext context;
    context.input_names = {"ids"};
    context.output_names = {"gather"};
    TF_ASSERT_OK(SparsifyGather(graph_def, context, &result));

    // Validation begins.
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);

    // Check nodes.
    EXPECT_EQ(1, node_lookup.count("ids"));
    EXPECT_EQ("Const", node_lookup.at("ids")->op());

    // Nodes in "const" scope.
    EXPECT_EQ(1, node_lookup.count("const/indices"));
    EXPECT_EQ("Const", node_lookup.at("const/indices")->op());
    Tensor expected_indices_tensor(DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_indices_tensor, {0, 2, 3});
    test::ExpectTensorEqual<int64>(
        expected_indices_tensor,
        GetNodeTensorAttr(*(node_lookup.at("const/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("const/values"));
    EXPECT_EQ("Const", node_lookup.at("const/values")->op());
    Tensor expected_values_tensor(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor,
        GetNodeTensorAttr(*(node_lookup.at("const/values")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("const/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("const/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("const/InitializeTable"));
    EXPECT_EQ("InitializeTable", node_lookup.at("const/InitializeTable")->op());

    // Nodes in "gather" scope.
    EXPECT_EQ(1, node_lookup.count("gather/LookupTableFind"));
    EXPECT_EQ("LookupTableFind",
              node_lookup.at("gather/LookupTableFind")->op());

    EXPECT_EQ(1, node_lookup.count("gather/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather/Const")->op());
    Tensor expected_gather_default_tensor(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected_gather_default_tensor, {0.0});
    test::ExpectTensorNear<float>(
        expected_gather_default_tensor,
        GetNodeTensorAttr(*(node_lookup.at("gather/Const")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("gather/ExpandDims/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather/ExpandDims/Const")->op());
    Tensor expected_expand_dims_tensor(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&expected_expand_dims_tensor, {-1});
    test::ExpectTensorEqual<int32>(
        expected_expand_dims_tensor,
        GetNodeTensorAttr(*(node_lookup.at("gather/ExpandDims/Const")),
                          "value"));

    EXPECT_EQ(1, node_lookup.count("gather"));
    EXPECT_EQ("ExpandDims", node_lookup.at("gather")->op());

    EXPECT_EQ(1, node_lookup.count("group_deps"));
    EXPECT_EQ("NoOp", node_lookup.at("group_deps")->op());

    // Check connections
    EXPECT_EQ("const/HashTable",
              node_lookup.at("const/InitializeTable")->input(0));
    EXPECT_EQ("const/indices",
              node_lookup.at("const/InitializeTable")->input(1));
    EXPECT_EQ("const/values",
              node_lookup.at("const/InitializeTable")->input(2));

    EXPECT_EQ("const/HashTable",
              node_lookup.at("gather/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather/LookupTableFind")->input(1));
    EXPECT_EQ("gather/Const",
              node_lookup.at("gather/LookupTableFind")->input(2));

    EXPECT_EQ("gather/LookupTableFind", node_lookup.at("gather")->input(0));

    // Check control dependency.
    EXPECT_NE(std::find(node_lookup.at("group_deps")->input().begin(),
                        node_lookup.at("group_deps")->input().end(),
                        "^const/InitializeTable"),
              node_lookup.at("group_deps")->input().end());
  }

  void TestMultiPartitionConst() {
    // The 'ids' node is served input for two 'Gather's.
    GraphDef graph_def;

    // Build Graph:
    // Shared input node
    NodeDef* input_node = CreateNode("ids", "Const", {}, &graph_def);
    // Shared init node
    CreateNode("group_deps", "NoOp", {}, &graph_def);

    // Two partitions
    NodeDef* const_node1 = CreateNode("const1", "Const", {}, &graph_def);
    SetNodeAttr("dtype", DT_FLOAT, const_node1);
    // Set 'Const' node value.
    Tensor weights(DT_FLOAT, TensorShape({4, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001, 1.2, 0.001});
    SetNodeTensorAttr<float>("value", weights, const_node1);

    NodeDef* const_node2 = CreateNode("const2", "Const", {}, &graph_def);
    SetNodeAttr("dtype", DT_FLOAT, const_node2);
    SetNodeTensorAttr<float>("value", weights, const_node2);

    NodeDef* identity_node1 =
        CreateNode("const1/read", "Identity", {const_node1}, &graph_def);
    NodeDef* identity_node2 =
        CreateNode("const2/read", "Identity", {const_node2}, &graph_def);
    CreateNode("gather1", "Gather", {identity_node1, input_node}, &graph_def);
    CreateNode("gather2", "Gather", {identity_node2, input_node}, &graph_def);

    // Run the op.
    GraphDef result;
    TransformFuncContext context;
    context.input_names = {"ids"};
    context.output_names = {"gather1", "gather2"};
    TF_ASSERT_OK(SparsifyGather(graph_def, context, &result));

    // Validation begins.
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);

    // Check nodes.
    // Check shared nodes:
    EXPECT_EQ(1, node_lookup.count("ids"));
    EXPECT_EQ("Const", node_lookup.at("ids")->op());

    EXPECT_EQ(1, node_lookup.count("group_deps"));
    EXPECT_EQ("NoOp", node_lookup.at("group_deps")->op());

    // Nodes in "const1" scope.
    EXPECT_EQ(1, node_lookup.count("const1/indices"));
    EXPECT_EQ("Const", node_lookup.at("const1/indices")->op());
    Tensor expected_indices_tensor1(DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_indices_tensor1, {0, 2, 3});
    test::ExpectTensorEqual<int64>(
        expected_indices_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("const1/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("const1/values"));
    EXPECT_EQ("Const", node_lookup.at("const1/values")->op());
    Tensor expected_values_tensor1(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor1, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("const1/values")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("const1/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("const1/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("const1/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("const1/InitializeTable")->op());

    // Nodes in "gather1" scope.
    EXPECT_EQ(1, node_lookup.count("gather1/LookupTableFind"));
    EXPECT_EQ("LookupTableFind",
              node_lookup.at("gather1/LookupTableFind")->op());

    EXPECT_EQ(1, node_lookup.count("gather1/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather1/Const")->op());
    Tensor expected_gather_default_tensor1(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected_gather_default_tensor1, {0.0});
    test::ExpectTensorNear<float>(
        expected_gather_default_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("gather1/Const")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("gather1/ExpandDims/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather1/ExpandDims/Const")->op());
    Tensor expected_expand_dims_tensor1(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&expected_expand_dims_tensor1, {-1});
    test::ExpectTensorEqual<int32>(
        expected_expand_dims_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("gather1/ExpandDims/Const")),
                          "value"));

    EXPECT_EQ(1, node_lookup.count("gather1"));
    EXPECT_EQ("ExpandDims", node_lookup.at("gather1")->op());

    // Nodes in "const2" scope.
    EXPECT_EQ(1, node_lookup.count("const2/indices"));
    EXPECT_EQ("Const", node_lookup.at("const2/indices")->op());
    Tensor expected_indices_tensor2(DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_indices_tensor2, {0, 2, 3});
    test::ExpectTensorEqual<int64>(
        expected_indices_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("const2/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("const2/values"));
    EXPECT_EQ("Const", node_lookup.at("const2/values")->op());
    Tensor expected_values_tensor2(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor2, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("const2/values")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("const2/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("const2/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("const2/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("const2/InitializeTable")->op());

    // Nodes in "gather2" scope.
    EXPECT_EQ(1, node_lookup.count("gather2/LookupTableFind"));
    EXPECT_EQ("LookupTableFind",
              node_lookup.at("gather2/LookupTableFind")->op());

    EXPECT_EQ(1, node_lookup.count("gather2/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather2/Const")->op());
    Tensor expected_gather_default_tensor2(DT_FLOAT, TensorShape({}));
    test::FillValues<float>(&expected_gather_default_tensor2, {0.0});
    test::ExpectTensorNear<float>(
        expected_gather_default_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("gather2/Const")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("gather2/ExpandDims/Const"));
    EXPECT_EQ("Const", node_lookup.at("gather2/ExpandDims/Const")->op());
    Tensor expected_expand_dims_tensor2(DT_INT32, TensorShape({}));
    test::FillValues<int32>(&expected_expand_dims_tensor2, {-1});
    test::ExpectTensorEqual<int32>(
        expected_expand_dims_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("gather2/ExpandDims/Const")),
                          "value"));

    EXPECT_EQ(1, node_lookup.count("gather2"));
    EXPECT_EQ("ExpandDims", node_lookup.at("gather2")->op());

    // Check connections
    EXPECT_EQ("const1/HashTable",
              node_lookup.at("const1/InitializeTable")->input(0));
    EXPECT_EQ("const1/indices",
              node_lookup.at("const1/InitializeTable")->input(1));
    EXPECT_EQ("const1/values",
              node_lookup.at("const1/InitializeTable")->input(2));

    EXPECT_EQ("const2/HashTable",
              node_lookup.at("const2/InitializeTable")->input(0));
    EXPECT_EQ("const2/indices",
              node_lookup.at("const2/InitializeTable")->input(1));
    EXPECT_EQ("const2/values",
              node_lookup.at("const2/InitializeTable")->input(2));

    EXPECT_EQ("const1/HashTable",
              node_lookup.at("gather1/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather1/LookupTableFind")->input(1));
    EXPECT_EQ("gather1/Const",
              node_lookup.at("gather1/LookupTableFind")->input(2));
    EXPECT_EQ("gather1/LookupTableFind", node_lookup.at("gather1")->input(0));

    EXPECT_EQ("const2/HashTable",
              node_lookup.at("gather2/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather2/LookupTableFind")->input(1));
    EXPECT_EQ("gather2/Const",
              node_lookup.at("gather2/LookupTableFind")->input(2));
    EXPECT_EQ("gather2/LookupTableFind", node_lookup.at("gather2")->input(0));

    // Check control deps.
    EXPECT_EQ(2, node_lookup.at("group_deps")->input_size());
    EXPECT_NE(std::find(node_lookup.at("group_deps")->input().begin(),
                        node_lookup.at("group_deps")->input().end(),
                        "^const1/InitializeTable"),
              node_lookup.at("group_deps")->input().end());

    EXPECT_NE(std::find(node_lookup.at("group_deps")->input().begin(),
                        node_lookup.at("group_deps")->input().end(),
                        "^const2/InitializeTable"),
              node_lookup.at("group_deps")->input().end());
  }
};

TEST_F(SparsifyGatherTest, TestSinglePartitionConst) {
  TestSinglePartitionConst();
}

TEST_F(SparsifyGatherTest, TestMultiPartitionConst) {
  TestMultiPartitionConst();
}

}  // namespace graph_transforms
}  // namespace tensorflow
