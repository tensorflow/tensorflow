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
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

const string NAME_A = "a";
const string NAME_B = "b";
const string NAME_A_PLUS_B = "a_plus_b";
constexpr float NODE_A_VAL = 2.0f;
constexpr float NODE_B_VAL = 3.0f;
constexpr float VALUE_TOLERANCE_FLOAT = 1e-8f;

static Output BuildAddOps(const Scope& scope, const Input& x, const Input& y) {
  EXPECT_TRUE(scope.ok());
  auto _x = ops::AsNodeOut(scope, x);
  EXPECT_TRUE(scope.ok());
  auto _y = ops::AsNodeOut(scope, y);
  EXPECT_TRUE(scope.ok());
  Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("Add");
  auto builder = NodeBuilder(unique_name, "Add").Input(_x).Input(_y);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  EXPECT_TRUE(scope.ok());
  return Output(ret, 0);
}

static GraphDef CreateAddGraphDef() {
  Scope root = Scope::NewRootScope();
  Output node_a = ops::Const(root.WithOpName(NAME_A), NODE_A_VAL);
  Output node_b = ops::Const(root.WithOpName(NAME_B), NODE_B_VAL);
  Output node_add = BuildAddOps(root.WithOpName(NAME_A_PLUS_B), node_a, node_b);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphA) {
  GraphDef def = CreateAddGraphDef();
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
  GraphDef def = CreateAddGraphDef();
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
  GraphDef def = CreateAddGraphDef();
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
  RemoteFusedGraphExecuteUtils::TensorShapeMap output_tensor_info;
  GraphDef def = CreateAddGraphDef();

  // dryrun
  const Status status = RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
      def, inputs, false /* initialize_by_zero */, &output_tensor_info);

  ASSERT_TRUE(status.ok()) << status;

  // Assert output node count
  ASSERT_EQ(3, output_tensor_info.size());
  ASSERT_EQ(1, output_tensor_info.count(NAME_A));
  ASSERT_EQ(1, output_tensor_info.count(NAME_B));
  ASSERT_EQ(1, output_tensor_info.count(NAME_A_PLUS_B));

  EXPECT_EQ(DT_FLOAT, output_tensor_info.at(NAME_B).first);
  EXPECT_EQ(DT_FLOAT, output_tensor_info.at(NAME_A_PLUS_B).first);
  const TensorShape& shape_b = output_tensor_info.at(NAME_B).second;
  const TensorShape& shape_a_b = output_tensor_info.at(NAME_A_PLUS_B).second;
  EXPECT_EQ(0, shape_b.dims());
  EXPECT_EQ(0, shape_a_b.dims());
}

}  // namespace tensorflow
