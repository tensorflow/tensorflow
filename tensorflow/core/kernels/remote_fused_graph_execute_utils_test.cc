/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_op_test_utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

constexpr const char* const NAME_A = "a";
constexpr const char* const NAME_B = "b";
constexpr const char* const NAME_A_PLUS_B = "a_plus_b";
constexpr float NODE_A_VAL = 2.0f;
constexpr float NODE_B_VAL = 3.0f;
constexpr float VALUE_TOLERANCE_FLOAT = 1e-8f;

static NodeDef* GetNodeDef(const string& name, GraphDef* def) {
  CHECK_NE(def, nullptr);
  for (NodeDef& node_def : *def->mutable_node()) {
    if (node_def.name() == name) {
      return &node_def;
    }
  }
  return nullptr;
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphA) {
  GraphDef def = RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B);
  std::pair<string, Tensor> input_node_info;
  input_node_info.first = NAME_A;
  input_node_info.second = Tensor(DT_FLOAT, {});
  input_node_info.second.scalar<float>()() = 1.0f;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info};
  std::vector<string> outputs = {NAME_B, NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInference(
      def, inputs, outputs, false /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
  EXPECT_NEAR(1.0f + NODE_B_VAL, output_tensors.at(1).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphAUninitialized) {
  GraphDef def = RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B);
  std::pair<string, Tensor> input_node_info;
  input_node_info.first = NAME_A;
  input_node_info.second = Tensor(DT_FLOAT, {});
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info};
  std::vector<string> outputs = {NAME_B, NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInference(
      def, inputs, outputs, true /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(1).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphAB) {
  GraphDef def = RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B);
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = NODE_A_VAL;
  std::pair<string, Tensor> input_node_info_b;
  input_node_info_b.first = NAME_B;
  input_node_info_b.second = Tensor(DT_FLOAT, {});
  input_node_info_b.second.scalar<float>()() = NODE_B_VAL;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a,
                                                      input_node_info_b};
  std::vector<string> outputs = {NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInference(
      def, inputs, outputs, false /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_A_VAL + NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphForAllNodes) {
  // Set Node "a" as an input with value (= 1.0f)
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = 1.0f;

  // Setup dryrun arguments
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a};
  RemoteFusedGraphExecuteUtils::TensorShapeMap tensor_shape_map;

  GraphDef def = RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B);

  // dryrun
  const Status status = RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
      def, inputs, false /* initialize_by_zero */, &tensor_shape_map);

  ASSERT_TRUE(status.ok()) << status;

  // Assert output node count
  ASSERT_EQ(3, tensor_shape_map.size());
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_B));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A_PLUS_B));

  const RemoteFusedGraphExecuteUtils::TensorShapeType* tst =
      RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                       NAME_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());

  tst = RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                         NAME_A_PLUS_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());
}

TEST(RemoteFusedGraphExecuteUtils, PropagateAndBuildTensorShapeMap) {
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = NODE_A_VAL;
  std::pair<string, Tensor> input_node_info_b;
  input_node_info_b.first = NAME_B;
  input_node_info_b.second = Tensor(DT_FLOAT, {});
  input_node_info_b.second.scalar<float>()() = NODE_B_VAL;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a,
                                                      input_node_info_b};

  RemoteFusedGraphExecuteUtils::TensorShapeMap tensor_shape_map;
  GraphDef def = RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B);
  ImportGraphDefOptions opts;
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions().producer(), graph.op_registry());
  Status status = ImportGraphDef(opts, def, &graph, &shape_refiner);
  ASSERT_TRUE(RemoteFusedGraphExecuteUtils::PropagateShapeInference(
                  def, inputs, &graph, &shape_refiner)
                  .ok());
  ASSERT_TRUE(RemoteFusedGraphExecuteUtils::BuildTensorShapeMapFromGraph(
                  graph, shape_refiner, &tensor_shape_map)
                  .ok());

  ASSERT_EQ(3, tensor_shape_map.size());
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_B));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A_PLUS_B));

  const RemoteFusedGraphExecuteUtils::TensorShapeType* tst =
      RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                       NAME_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());

  tst = RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                         NAME_A_PLUS_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());

  {
    NodeDef* node_def = GetNodeDef(NAME_B, &def);
    TF_ASSERT_OK(
        RemoteFusedGraphExecuteUtils::AddOutputTensorShapeTypeByTensorShapeMap(
            tensor_shape_map, node_def));
    std::vector<DataType> data_types;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_DATA_TYPES,
        &data_types));
    ASSERT_EQ(1, data_types.size());
    EXPECT_EQ(DT_FLOAT, data_types.at(0));

    std::vector<TensorShape> shapes;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_SHAPES, &shapes));
    ASSERT_EQ(1, shapes.size());
    EXPECT_EQ(0, shapes.at(0).dims());
  }

  {
    NodeDef* node_def = GetNodeDef(NAME_A_PLUS_B, &def);
    TF_ASSERT_OK(
        RemoteFusedGraphExecuteUtils::AddOutputTensorShapeTypeByTensorShapeMap(
            tensor_shape_map, node_def));
    std::vector<DataType> data_types;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_DATA_TYPES,
        &data_types));
    ASSERT_EQ(1, data_types.size());
    EXPECT_EQ(DT_FLOAT, data_types.at(0));

    std::vector<TensorShape> shapes;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_SHAPES, &shapes));
    ASSERT_EQ(1, shapes.size());
    EXPECT_EQ(0, shapes.at(0).dims());
  }
}

}  // namespace tensorflow
