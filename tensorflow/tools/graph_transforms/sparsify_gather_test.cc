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
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declarations so we don't need a public header.
Status SparsifyGather(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def);
Status ReadTensorFromCheckpoint(
    const string& tensor_name, const std::unique_ptr<BundleReader>& ckpt_reader,
    const string& shape_and_slice, Tensor* tensor);

class SparsifyGatherTest : public ::testing::Test {
 protected:
  NodeDef* CreateNode(const StringPiece name, const StringPiece op,
                      const std::vector<NodeDef*>& inputs, GraphDef* graph_def,
                      bool control_dep = false) {
    NodeDef* node_def = graph_def->add_node();
    node_def->set_name(name.ToString());
    node_def->set_op(op.ToString());
    if (!control_dep) {
      std::for_each(inputs.begin(), inputs.end(), [&node_def](NodeDef* input) {
        node_def->add_input(input->name());
      });
    } else {
      std::for_each(inputs.begin(), inputs.end(), [&node_def](NodeDef* input) {
        node_def->add_input(strings::StrCat("^", input->name()));
      });
    }
    return node_def;
  }

  void MakeGather(StringPiece name, bool gather_v2, NodeDef* params,
                  NodeDef* indices, GraphDef* graph_def) {
    if (gather_v2) {
      NodeDef* axis_node =
          CreateNode(strings::StrCat(name, "_axis"), "Const", {}, graph_def);
      Tensor axis_t(DT_INT32, TensorShape({}));
      axis_t.scalar<int32>()() = 0;
      SetNodeTensorAttr<int32>("value", axis_t, axis_node);
      CreateNode(name, "GatherV2", {params, indices, axis_node}, graph_def);
    } else {
      CreateNode(name, "Gather", {params, indices}, graph_def);
    }
  }

  void TestSinglePartition(bool gather_v2, bool include_shared_init,
                           bool test_variable,
                           const string& shared_init_name = "group_deps") {
    GraphDef graph_def;

    const auto checkpoint_path =
        io::JoinPath(testing::TmpDir(), "checkpoint_single");
    // Build the graph.
    NodeDef* input_node = CreateNode("ids", "Const", {}, &graph_def);
    NodeDef* w_node;
    NodeDef* zeros_node;
    NodeDef* assign_node;

    Tensor weights(DT_FLOAT, TensorShape({4, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001, 1.2, 0.001});

    if (!test_variable) {
      w_node = CreateNode("w/part_1", "Const", {}, &graph_def);
      SetNodeTensorAttr<float>("value", weights, w_node);
    } else {
      w_node = CreateNode("w/part_1", "VariableV2", {}, &graph_def);

      zeros_node =
          CreateNode("w/part_1/Initializer/zeros", "Const", {}, &graph_def);
      assign_node = CreateNode("w/part_1/Assign", "Assign",
                               {w_node, zeros_node}, &graph_def);

      NodeDef* save_const_node =
          CreateNode("save/Const", "Const", {}, &graph_def);

      NodeDef* tensor_names_node =
          CreateNode("save/RestoreV2/tensor_names", "Const", {}, &graph_def);
      NodeDef* tensor_shapes_slices_node = CreateNode(
          "save/RestoreV2/shape_and_slices", "Const", {}, &graph_def);

      Tensor shapes_slices_val(DT_STRING, TensorShape({1}));
      shapes_slices_val.flat<string>()(0) = "4 1 0,4:0,1";
      SetNodeTensorAttr<string>("value", shapes_slices_val,
                                tensor_shapes_slices_node);

      NodeDef* restore_node = CreateNode(
          "save/RestoreV2", "RestoreV2",
          {save_const_node, tensor_names_node, tensor_shapes_slices_node},
          &graph_def);
      CreateNode("save/Assign", "Assign", {w_node, restore_node}, &graph_def);

      BundleWriter writer(Env::Default(), checkpoint_path);
      TF_ASSERT_OK(writer.Add("w", weights));
      TF_ASSERT_OK(writer.Finish());
    }
    SetNodeAttr("dtype", DT_FLOAT, w_node);

    NodeDef* identity_node =
        CreateNode("w/read", "Identity", {w_node}, &graph_def);
    MakeGather("gather", gather_v2, identity_node, input_node, &graph_def);
    if (include_shared_init) {
      if (!test_variable) {
        CreateNode(shared_init_name, "NoOp", {}, &graph_def);
      } else {
        CreateNode(shared_init_name, "NoOp", {assign_node}, &graph_def, true);
      }
    }

    // Run the op.
    GraphDef result;
    TransformFuncContext context;
    context.input_names = {"ids"};
    context.output_names = {"gather"};
    if (test_variable) {
      context.params["input_checkpoint"] = {checkpoint_path};
    }
    if (shared_init_name != "group_deps") {
      context.params["group_init_node"] = {shared_init_name};
    }
    TF_ASSERT_OK(SparsifyGather(graph_def, context, &result));

    // Validation begins.
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);

    // Check nodes.
    EXPECT_EQ(0, node_lookup.count("w/part_1/Initializer/zeros"));
    EXPECT_EQ(0, node_lookup.count("w/part_1/Assign"));

    EXPECT_EQ(1, node_lookup.count("ids"));
    EXPECT_EQ("Const", node_lookup.at("ids")->op());

    EXPECT_EQ(1, node_lookup.count("w/part_1/indices"));
    EXPECT_EQ("Const", node_lookup.at("w/part_1/indices")->op());
    Tensor expected_indices_tensor(DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_indices_tensor, {0, 2, 3});
    test::ExpectTensorEqual<int64>(
        expected_indices_tensor,
        GetNodeTensorAttr(*(node_lookup.at("w/part_1/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("w/part_1/values"));
    EXPECT_EQ("Const", node_lookup.at("w/part_1/values")->op());
    Tensor expected_values_tensor(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor,
        GetNodeTensorAttr(*(node_lookup.at("w/part_1/values")), "value"), 1e-5);

    EXPECT_EQ(1, node_lookup.count("w/part_1/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("w/part_1/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("w/part_1/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("w/part_1/InitializeTable")->op());

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

    EXPECT_EQ(1, node_lookup.count(shared_init_name));
    EXPECT_EQ("NoOp", node_lookup.at(shared_init_name)->op());

    // Check connections
    EXPECT_EQ("w/part_1/HashTable",
              node_lookup.at("w/part_1/InitializeTable")->input(0));
    EXPECT_EQ("w/part_1/indices",
              node_lookup.at("w/part_1/InitializeTable")->input(1));
    EXPECT_EQ("w/part_1/values",
              node_lookup.at("w/part_1/InitializeTable")->input(2));

    EXPECT_EQ("w/part_1/HashTable",
              node_lookup.at("gather/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather/LookupTableFind")->input(1));
    EXPECT_EQ("gather/Const",
              node_lookup.at("gather/LookupTableFind")->input(2));

    EXPECT_EQ("gather/LookupTableFind", node_lookup.at("gather")->input(0));

    // Check control dependency.
    EXPECT_NE(std::find(node_lookup.at(shared_init_name)->input().begin(),
                        node_lookup.at(shared_init_name)->input().end(),
                        "^w/part_1/InitializeTable"),
              node_lookup.at(shared_init_name)->input().end());
    EXPECT_EQ(1, node_lookup.at(shared_init_name)->input().size());
  }

  void TestMultiPartition(bool gather_v2, bool include_shared_init,
                          bool test_variable,
                          const string& shared_init_name = "group_deps") {
    // The 'ids' node is served input for two 'Gather's.
    GraphDef graph_def;

    const auto checkpoint_path =
        io::JoinPath(testing::TmpDir(), "checkpoint_multiple");
    // Build Graph:
    // Shared input node
    NodeDef* input_node = CreateNode("ids", "Const", {}, &graph_def);

    // Two partitions
    NodeDef* w_node1;
    NodeDef* w_node2;
    NodeDef* zeros_node1;
    NodeDef* zeros_node2;
    NodeDef* assign_node1;
    NodeDef* assign_node2;

    Tensor weights(DT_FLOAT, TensorShape({4, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001, 1.2, 0.001});
    if (!test_variable) {
      w_node1 = CreateNode("w1/part_1", "Const", {}, &graph_def);
      w_node2 = CreateNode("w2/part_1", "Const", {}, &graph_def);
      SetNodeTensorAttr<float>("value", weights, w_node1);
      SetNodeTensorAttr<float>("value", weights, w_node2);
    } else {
      w_node1 = CreateNode("w1/part_1", "VariableV2", {}, &graph_def);
      zeros_node1 =
          CreateNode("w1/part_1/Initializer/zeros", "Const", {}, &graph_def);
      assign_node1 = CreateNode("w1/part_1/Assign", "Assign",
                                {w_node1, zeros_node1}, &graph_def);

      NodeDef* save_const_node =
          CreateNode("save/Const", "Const", {}, &graph_def);
      NodeDef* tensor_names_node1 =
          CreateNode("save/RestoreV2/tensor_names", "Const", {}, &graph_def);
      NodeDef* tensor_shapes_slices_node1 = CreateNode(
          "save/RestoreV2/shape_and_slices", "Const", {}, &graph_def);

      Tensor shapes_slices_val1(DT_STRING, TensorShape({1}));
      shapes_slices_val1.flat<string>()(0) = "4 1 0,4:0,1";
      SetNodeTensorAttr<string>("value", shapes_slices_val1,
                                tensor_shapes_slices_node1);

      NodeDef* restore_node1 = CreateNode(
          "save/RestoreV2", "RestoreV2",
          {save_const_node, tensor_names_node1, tensor_shapes_slices_node1},
          &graph_def);
      CreateNode("save/Assign", "Assign", {w_node1, restore_node1}, &graph_def);

      w_node2 = CreateNode("w2/part_1", "VariableV2", {}, &graph_def);
      zeros_node2 =
          CreateNode("w2/part_1/Initializer/zeros", "Const", {}, &graph_def);
      assign_node2 = CreateNode("w2/part_1/Assign", "Assign",
                                {w_node2, zeros_node2}, &graph_def);

      NodeDef* tensor_names_node2 =
          CreateNode("save/RestoreV2_1/tensor_names", "Const", {}, &graph_def);
      NodeDef* tensor_shapes_slices_node2 = CreateNode(
          "save/RestoreV2_1/shape_and_slices", "Const", {}, &graph_def);

      Tensor shapes_slices_val2(DT_STRING, TensorShape({1}));
      shapes_slices_val2.flat<string>()(0) = "4 1 0,4:0,1";
      SetNodeTensorAttr<string>("value", shapes_slices_val2,
                                tensor_shapes_slices_node2);

      NodeDef* restore_node2 = CreateNode(
          "save/RestoreV2_1", "RestoreV2",
          {save_const_node, tensor_names_node2, tensor_shapes_slices_node2},
          &graph_def);
      CreateNode("save/Assign_1", "Assign", {w_node2, restore_node2},
                 &graph_def);

      BundleWriter writer(Env::Default(), checkpoint_path);
      TF_ASSERT_OK(writer.Add("w1", weights));
      TF_ASSERT_OK(writer.Add("w2", weights));
      TF_ASSERT_OK(writer.Finish());
    }
    SetNodeAttr("dtype", DT_FLOAT, w_node1);
    SetNodeAttr("dtype", DT_FLOAT, w_node2);

    NodeDef* identity_node1 =
        CreateNode("w1/part_1/read", "Identity", {w_node1}, &graph_def);
    NodeDef* identity_node2 =
        CreateNode("w2/part_1/read", "Identity", {w_node2}, &graph_def);
    MakeGather("gather1", gather_v2, identity_node1, input_node, &graph_def);
    MakeGather("gather2", gather_v2, identity_node2, input_node, &graph_def);

    // Shared init node
    if (include_shared_init) {
      if (!test_variable) {
        CreateNode(shared_init_name, "NoOp", {}, &graph_def);
      } else {
        CreateNode(shared_init_name, "NoOp", {assign_node1, assign_node2},
                   &graph_def, true);
      }
    }

    // Run the op.
    GraphDef result;
    TransformFuncContext context;
    context.input_names = {"ids"};
    context.output_names = {"gather1", "gather2"};
    if (test_variable) {
      context.params["input_checkpoint"] = {checkpoint_path};
    }
    if (shared_init_name != "group_deps") {
      context.params["group_init_node"] = {shared_init_name};
    }
    TF_ASSERT_OK(SparsifyGather(graph_def, context, &result));

    // Validation begins.
    std::map<string, const NodeDef*> node_lookup;
    MapNamesToNodes(result, &node_lookup);

    // Check nodes.
    EXPECT_EQ(0, node_lookup.count("w1/part_1/Initializer/zeros"));
    EXPECT_EQ(0, node_lookup.count("w1/part_1/Assign"));
    EXPECT_EQ(0, node_lookup.count("w2/part_1/Initializer/zeros"));
    EXPECT_EQ(0, node_lookup.count("w2/part_1/Assign"));
    EXPECT_EQ(1, node_lookup.count("ids"));
    EXPECT_EQ("Const", node_lookup.at("ids")->op());

    EXPECT_EQ(1, node_lookup.count(shared_init_name));
    EXPECT_EQ("NoOp", node_lookup.at(shared_init_name)->op());

    EXPECT_EQ(1, node_lookup.count("w1/part_1/indices"));
    EXPECT_EQ("Const", node_lookup.at("w1/part_1/indices")->op());
    Tensor expected_indices_tensor1(DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_indices_tensor1, {0, 2, 3});
    test::ExpectTensorEqual<int64>(
        expected_indices_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("w1/part_1/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("w1/part_1/values"));
    EXPECT_EQ("Const", node_lookup.at("w1/part_1/values")->op());
    Tensor expected_values_tensor1(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor1, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor1,
        GetNodeTensorAttr(*(node_lookup.at("w1/part_1/values")), "value"),
        1e-5);

    EXPECT_EQ(1, node_lookup.count("w1/part_1/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("w1/part_1/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("w1/part_1/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("w1/part_1/InitializeTable")->op());

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

    EXPECT_EQ(1, node_lookup.count("w2/part_1/indices"));
    EXPECT_EQ("Const", node_lookup.at("w2/part_1/indices")->op());
    Tensor expected_indices_tensor2(DT_INT64, TensorShape({3}));
    test::FillValues<int64>(&expected_indices_tensor2, {0, 2, 3});
    test::ExpectTensorEqual<int64>(
        expected_indices_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("w2/part_1/indices")), "value"));

    EXPECT_EQ(1, node_lookup.count("w2/part_1/values"));
    EXPECT_EQ("Const", node_lookup.at("w2/part_1/values")->op());
    Tensor expected_values_tensor2(DT_FLOAT, TensorShape({3}));
    test::FillValues<float>(&expected_values_tensor2, {0.2, 1.2, 0.001});
    test::ExpectTensorNear<float>(
        expected_values_tensor2,
        GetNodeTensorAttr(*(node_lookup.at("w2/part_1/values")), "value"),
        1e-5);

    EXPECT_EQ(1, node_lookup.count("w2/part_1/HashTable"));
    EXPECT_EQ("HashTable", node_lookup.at("w2/part_1/HashTable")->op());

    EXPECT_EQ(1, node_lookup.count("w2/part_1/InitializeTable"));
    EXPECT_EQ("InitializeTable",
              node_lookup.at("w2/part_1/InitializeTable")->op());

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
    EXPECT_EQ("w1/part_1/HashTable",
              node_lookup.at("w1/part_1/InitializeTable")->input(0));
    EXPECT_EQ("w1/part_1/indices",
              node_lookup.at("w1/part_1/InitializeTable")->input(1));
    EXPECT_EQ("w1/part_1/values",
              node_lookup.at("w1/part_1/InitializeTable")->input(2));

    EXPECT_EQ("w2/part_1/HashTable",
              node_lookup.at("w2/part_1/InitializeTable")->input(0));
    EXPECT_EQ("w2/part_1/indices",
              node_lookup.at("w2/part_1/InitializeTable")->input(1));
    EXPECT_EQ("w2/part_1/values",
              node_lookup.at("w2/part_1/InitializeTable")->input(2));

    EXPECT_EQ("w1/part_1/HashTable",
              node_lookup.at("gather1/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather1/LookupTableFind")->input(1));
    EXPECT_EQ("gather1/Const",
              node_lookup.at("gather1/LookupTableFind")->input(2));
    EXPECT_EQ("gather1/LookupTableFind", node_lookup.at("gather1")->input(0));

    EXPECT_EQ("w2/part_1/HashTable",
              node_lookup.at("gather2/LookupTableFind")->input(0));
    EXPECT_EQ("ids", node_lookup.at("gather2/LookupTableFind")->input(1));
    EXPECT_EQ("gather2/Const",
              node_lookup.at("gather2/LookupTableFind")->input(2));
    EXPECT_EQ("gather2/LookupTableFind", node_lookup.at("gather2")->input(0));

    // Check control deps.
    EXPECT_EQ(2, node_lookup.at(shared_init_name)->input_size());
    EXPECT_NE(std::find(node_lookup.at(shared_init_name)->input().begin(),
                        node_lookup.at(shared_init_name)->input().end(),
                        "^w1/part_1/InitializeTable"),
              node_lookup.at(shared_init_name)->input().end());

    EXPECT_NE(std::find(node_lookup.at(shared_init_name)->input().begin(),
                        node_lookup.at(shared_init_name)->input().end(),
                        "^w2/part_1/InitializeTable"),
              node_lookup.at(shared_init_name)->input().end());
  }
  void TestReadTensorSlice() {
    const auto checkpoint_path =
        io::JoinPath(testing::TmpDir(), "checkpoint_slice");

    Tensor weights(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&weights, {0.2, 0.000001});
    BundleWriter writer(Env::Default(), checkpoint_path);
    TF_ASSERT_OK(writer.AddSlice("w", TensorShape({4, 1}),
                                 TensorSlice::ParseOrDie("0,2:0,1"), weights));
    TF_ASSERT_OK(writer.Finish());

    std::unique_ptr<BundleReader> reader(
        new BundleReader(Env::Default(), checkpoint_path));

    Tensor results;
    TF_ASSERT_OK(
        ReadTensorFromCheckpoint("w/part_0", reader, "4 1 0,2:0,1", &results));

    test::ExpectTensorEqual<float>(weights, results);
  }
};

TEST_F(SparsifyGatherTest, TestSinglePartition) {
  TestSinglePartition(false, false, false);
  TestSinglePartition(false, true, false);
  TestSinglePartition(true, false, false);
  TestSinglePartition(true, true, false);
  TestSinglePartition(false, false, true);
  TestSinglePartition(false, true, true);
  TestSinglePartition(true, false, true);
  TestSinglePartition(true, true, true);
  TestSinglePartition(false, true, false, "shared_inits");
  TestSinglePartition(true, true, false, "shared_inits");
  TestSinglePartition(false, true, true, "shared_inits");
  TestSinglePartition(true, true, true, "shared_inits");
}

TEST_F(SparsifyGatherTest, TestMultiPartition) {
  TestMultiPartition(false, false, false);
  TestMultiPartition(false, true, false);
  TestMultiPartition(true, false, false);
  TestMultiPartition(true, true, false);
  TestMultiPartition(false, false, true);
  TestMultiPartition(false, true, true);
  TestMultiPartition(true, false, true);
  TestMultiPartition(true, true, true);
  TestMultiPartition(false, true, false, "shared_inits");
  TestMultiPartition(true, true, false, "shared_inits");
  TestMultiPartition(false, true, true, "shared_inits");
  TestMultiPartition(true, true, true, "shared_inits");
}

TEST_F(SparsifyGatherTest, TestTensorSlice) { TestReadTensorSlice(); }

}  // namespace graph_transforms
}  // namespace tensorflow
