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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_op_test_utils.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Declared here so we don't have to put it in a public header.
Status FuseRemoteGraph(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def);

Status PlaceRemoteGraphArguments(const GraphDef& input_graph_def,
                                 const TransformFuncContext& context,
                                 GraphDef* output_graph_def);

namespace {
constexpr const char* const REMOTE_FUSED_GRAPH_EXECUTOR_NAME =
    "remote_fused_graph_executor_name";
constexpr const char* const REMOTE_FUSED_GRAPH_NODE_NAME =
    "remote_fused_graph_node_name";
constexpr const char* const REMOTE_FUSED_EXECUTOR_NAME0 =
    "fuse_test_remote_fused_graph_executor0";
constexpr const char* const REMOTE_FUSED_EXECUTOR_NAME1 =
    "fuse_test_remote_fused_graph_executor1";

Status BuildRemoteFusedGraphExecutor0(
    std::unique_ptr<IRemoteFusedGraphExecutor>* executor) {
  executor->reset(
      new TestRemoteFusedGraphExecutor({"Mul"}, REMOTE_FUSED_EXECUTOR_NAME0));
  return Status::OK();
}

Status BuildRemoteFusedGraphExecutor1(
    std::unique_ptr<IRemoteFusedGraphExecutor>* executor) {
  executor->reset(new TestRemoteFusedGraphExecutor(
      {"Const", "Mul"}, REMOTE_FUSED_EXECUTOR_NAME1));
  return Status::OK();
}

class FuseRemoteGraphMultipleAddOpsRewriterTest : public ::testing::Test {
 protected:
  void SetUp() final {
    TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildMultipleAddGraph(
        &input_graph_def_));
    RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar
        hexagon_remote_fused_graph_executor_build(
            REMOTE_FUSED_GRAPH_EXECUTOR_NAME,
            [](std::unique_ptr<IRemoteFusedGraphExecutor>* executor) -> Status {
              return Status::OK();
            });
    RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar
        test_remote_fused_graph_executor_build0(REMOTE_FUSED_EXECUTOR_NAME0,
                                                BuildRemoteFusedGraphExecutor0);

    RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar
        test_remote_fused_graph_executor_build1(REMOTE_FUSED_EXECUTOR_NAME1,
                                                BuildRemoteFusedGraphExecutor1);
  }

  void TearDown() final {}

  Status Fuse() { return FuseInternal(/*only_place_args=*/false); }

  Status PlaceFuseArgs() { return FuseInternal(/*only_place_args*/ true); }

  Status FuseWithPlacedArgs() {
    const std::vector<std::pair<string, Tensor>> input_tensors{
        {"A", {DT_FLOAT, {1, 1, 1, 1}}}};
    return RemoteFusedGraphExecuteUtils::FuseRemoteGraphByPlacedArguments(
        input_graph_def_with_fuse_args_, input_tensors, &output_graph_def_);
  }

  Status FuseInternal(bool only_place_args) {
    TransformFuncContext context;
    context.input_names = inputs_;
    context.output_names = outputs_;

    if (!input_types_.empty()) {
      context.params.insert(std::pair<string, std::vector<string>>(
          {RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_INPUT_TYPES,
           {input_types_}}));
    }
    if (!input_shapes_.empty()) {
      context.params.insert(std::pair<string, std::vector<string>>(
          {RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_INPUT_SHAPES,
           {input_shapes_}}));
    }
    if (!fused_node_names_str_.empty()) {
      context.params.insert(std::pair<string, std::vector<string>>(
          {RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSED_NODES,
           {fused_node_names_str_}}));
    }

    if (!border_inputs_str_.empty()) {
      context.params.insert(std::pair<string, std::vector<string>>(
          {RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_BORDER_INPUTS,
           {border_inputs_str_}}));
    }
    if (!border_outputs_str_.empty()) {
      context.params.insert(std::pair<string, std::vector<string>>(
          {RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_BORDER_OUTPUTS,
           {border_outputs_str_}}));
    }

    if (!fused_op_types_str_.empty()) {
      context.params.insert(std::pair<string, std::vector<string>>(
          {RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSED_OP_TYPES,
           {fused_op_types_str_}}));
    }

    if (fuse_by_executor_) {
      context.params.insert(std::pair<string, std::vector<string>>(
          {RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSE_BY_EXECUTOR,
           {"true"}}));
    }

    context.params.insert(std::pair<string, std::vector<string>>(
        {RemoteFusedGraphExecuteUtils::
             TRANSFORM_ARG_REMOTE_FUSED_GRAPH_EXECUTOR_NAME,
         {remote_fused_graph_executor_name_}}));
    context.params.insert(std::pair<string, std::vector<string>>(
        {RemoteFusedGraphExecuteUtils::
             TRANSFORM_ARG_REMOTE_FUSED_GRAPH_NODE_NAME,
         {REMOTE_FUSED_GRAPH_NODE_NAME}}));

    if (only_place_args) {
      return PlaceRemoteGraphArguments(input_graph_def_, context,
                                       &input_graph_def_with_fuse_args_);
    } else {
      return FuseRemoteGraph(input_graph_def_, context, &output_graph_def_);
    }
  }

  void SetInputShapeType() {
    input_types_ = "float";
    input_shapes_ = "1,1,1,1";
  }

  void ReplaceOpType(const std::unordered_set<string>& op_name,
                     const string& new_op_type) {
    for (NodeDef& node_def : *input_graph_def_.mutable_node()) {
      if (op_name.count(node_def.name()) > 0) {
        node_def.set_op(new_op_type);
      }
    }
  }

  void CheckGraph(int expected_node_count, int expected_cluster_count) {
    EXPECT_EQ(expected_node_count, output_graph_def_.node_size());

    int cluster_count = 0;
    for (const NodeDef& node_def : output_graph_def_.node()) {
      const string& name = node_def.name();
      if (StringPiece(name).starts_with(REMOTE_FUSED_GRAPH_NODE_NAME)) {
        ++cluster_count;
        RemoteFusedGraphExecuteInfo info;
        string serialized_proto;
        TF_ASSERT_OK(
            GetNodeAttr(node_def,
                        RemoteFusedGraphExecuteUtils::
                            ATTR_SERIALIZED_REMOTE_FUSED_GRAPH_EXECUTE_INFO,
                        &serialized_proto));
        info.ParseFromString(serialized_proto);
        CHECK_EQ(remote_fused_graph_executor_name_, info.executor_name());
      }
    }
    EXPECT_EQ(expected_cluster_count, cluster_count);
  }

 public:
  const std::vector<string> inputs_{"A"};
  const std::vector<string> outputs_{"K"};
  GraphDef input_graph_def_;
  string input_types_;
  string input_shapes_;
  GraphDef input_graph_def_with_fuse_args_;
  GraphDef output_graph_def_;
  string fused_node_names_str_;
  string border_inputs_str_;
  string border_outputs_str_;
  string fused_op_types_str_;
  string remote_fused_graph_executor_name_{REMOTE_FUSED_GRAPH_EXECUTOR_NAME};
  bool fuse_by_executor_{false};
};

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByNodesWithShapeType_HIJ) {
  SetInputShapeType();
  fused_node_names_str_ = "H,I,J";
  TF_ASSERT_OK(Fuse());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByNodesWithoutShapeType_HIJ) {
  fused_node_names_str_ = "H,I,J";
  TF_ASSERT_OK(Fuse());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByNodesWithShapeType_ABCDEFGHIJK) {
  SetInputShapeType();
  fused_node_names_str_ = "A,B,C,D,E,F,G,H,I,J,K";
  TF_ASSERT_OK(Fuse());
  CheckGraph(3, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByNodesWithoutShapeType_ABCDEFGHIJK) {
  fused_node_names_str_ = "A,B,C,D,E,F,G,H,I,J,K";
  TF_ASSERT_OK(Fuse());
  CheckGraph(3, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByBorderWithShapeType_FCG_J) {
  SetInputShapeType();
  border_inputs_str_ = "F:0,C:0,G";
  border_outputs_str_ = "J:0";
  TF_ASSERT_OK(Fuse());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByBorderWithoutShapeType_FCG_J) {
  border_inputs_str_ = "F:0,C:0,G";
  border_outputs_str_ = "J:0";
  TF_ASSERT_OK(Fuse());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByBorderWithShapeType_ABCDE_K) {
  SetInputShapeType();
  border_inputs_str_ = "A,B,C,D,E";
  border_outputs_str_ = "K";
  TF_ASSERT_OK(Fuse());
  CheckGraph(7, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByBorderWithoutShapeType_ABCDE_K) {
  border_inputs_str_ = "A,B,C,D,E";
  border_outputs_str_ = "K";
  TF_ASSERT_OK(Fuse());
  CheckGraph(7, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByOpTypes_HIJ) {
  ReplaceOpType({"H", "I", "J"}, "Mul");
  fused_op_types_str_ = "Mul";
  TF_ASSERT_OK(Fuse());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByOpTypes_FGHIJ) {
  ReplaceOpType({"F", "G", "H", "I", "J"}, "Mul");
  fused_op_types_str_ = "Const,Mul";
  TF_ASSERT_OK(Fuse());
  CheckGraph(3, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByExecutor_HIJ) {
  ReplaceOpType({"H", "I", "J"}, "Mul");
  remote_fused_graph_executor_name_ = REMOTE_FUSED_EXECUTOR_NAME0;
  fuse_by_executor_ = true;
  TF_ASSERT_OK(Fuse());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       FuseRemoteGraphByExecutor_FGHIJ) {
  ReplaceOpType({"F", "G", "H", "I", "J"}, "Mul");
  remote_fused_graph_executor_name_ = REMOTE_FUSED_EXECUTOR_NAME1;
  fuse_by_executor_ = true;
  TF_ASSERT_OK(Fuse());
  CheckGraph(3, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest, PlaceAndFuse_HIJ) {
  fused_node_names_str_ = "H,I,J";
  TF_ASSERT_OK(PlaceFuseArgs());
  TF_ASSERT_OK(FuseWithPlacedArgs());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest, PlaceAndFuse_ABCDEFGHIJK) {
  fused_node_names_str_ = "A,B,C,D,E,F,G,H,I,J,K";
  TF_ASSERT_OK(PlaceFuseArgs());
  TF_ASSERT_OK(FuseWithPlacedArgs());
  CheckGraph(3, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest, PlaceAndFuse_FCG_J) {
  border_inputs_str_ = "F:0,C:0,G";
  border_outputs_str_ = "J:0";
  TF_ASSERT_OK(PlaceFuseArgs());
  TF_ASSERT_OK(FuseWithPlacedArgs());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest, PlaceAndFuse_ABCDE_K) {
  SetInputShapeType();
  border_inputs_str_ = "A,B,C,D,E";
  border_outputs_str_ = "K";
  TF_ASSERT_OK(PlaceFuseArgs());
  TF_ASSERT_OK(FuseWithPlacedArgs());
  CheckGraph(7, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest, PlaceAndFuse_MUL_HIJ) {
  SetInputShapeType();
  ReplaceOpType({"H", "I", "J"}, "Mul");
  fused_op_types_str_ = "Mul";

  TF_ASSERT_OK(PlaceFuseArgs());
  TF_ASSERT_OK(FuseWithPlacedArgs());
  CheckGraph(9, 1);
}

TEST_F(FuseRemoteGraphMultipleAddOpsRewriterTest,
       PlaceAndFuse_CONST_MUL_FGHIJ) {
  SetInputShapeType();
  ReplaceOpType({"F", "G", "H", "I", "J"}, "Mul");
  fused_op_types_str_ = "Const,Mul";

  TF_ASSERT_OK(PlaceFuseArgs());
  TF_ASSERT_OK(FuseWithPlacedArgs());
  CheckGraph(3, 1);
}

}  // namespace
}  // namespace graph_transforms
}  // namespace tensorflow
