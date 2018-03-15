/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/graph_transfer_info.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/hexagon/graph_transfer_utils.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_executor.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_ops_definitions.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

const string NAME_A = "a";
const string NAME_B = "b";
const string NAME_A_PLUS_B = "a_plus_b";
constexpr float NODE_A_VAL = 2.0f;
constexpr float NODE_B_VAL = 3.0f;
constexpr float VALUE_TOLERANCE_FLOAT = 1e-8f;

class GraphTransfererTest : public ::testing::Test {
 protected:
  void SetUp() final {}

  GraphTransferer gt_;
};

const RemoteFusedGraphExecuteUtils::TensorShapeMap EMPTY_OUTPUT_TENSOR_MAP;

class TestGraphTransferOpsDefinitions : public IRemoteFusedGraphOpsDefinitions {
 public:
  int GetTotalOpsCount() const final { return op_types_.size(); }

  int GetOpIdFor(const string& op_type, const DataTypeVector&) const final {
    for (int i = 0; i < op_types_.size(); ++i) {
      if (op_types_[i] == op_type) {
        return i;
      }
    }
    return -1;
  }

 private:
  const std::vector<string> op_types_{"INPUT",   "OUTPUT",  "Conv2D",
                                      "MaxPool", "NoOp",    "Add",
                                      "Const",   "Softmax", "Identity"};
} TEST_GRAPH_TRANSFER_OPS_DEFINITIONS;

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

static Output BuildSoftmaxOps(const Scope& scope, const Input& logits) {
  EXPECT_TRUE(scope.ok());
  auto _logits = ops::AsNodeOut(scope, logits);
  EXPECT_TRUE(scope.ok());
  Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("Softmax");
  auto builder = NodeBuilder(unique_name, "Softmax").Input(_logits);
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  EXPECT_TRUE(scope.ok());
  return Output(ret, 0);
}

static Output BuildConv2DOps(const Scope& scope, const Input& input,
                             const Input& filter,
                             const gtl::ArraySlice<int>& strides,
                             const StringPiece& padding) {
  EXPECT_TRUE(scope.ok());
  auto _input = ops::AsNodeOut(scope, input);
  EXPECT_TRUE(scope.ok());
  auto _filter = ops::AsNodeOut(scope, filter);
  EXPECT_TRUE(scope.ok());
  Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("Conv2D");
  auto builder = NodeBuilder(unique_name, "Conv2D")
                     .Input(_input)
                     .Input(_filter)
                     .Attr("strides", strides)
                     .Attr("use_cudnn_on_gpu", true)
                     .Attr("padding", padding)
                     .Attr("data_format", "NHWC");
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  EXPECT_TRUE(scope.ok());
  return Output(ret, 0);
}

static Output BuildMaxPoolOps(const Scope& scope, const Input& input,
                              const gtl::ArraySlice<int>& ksize,
                              const gtl::ArraySlice<int>& strides,
                              const StringPiece& padding) {
  EXPECT_TRUE(scope.ok());
  auto _input = ops::AsNodeOut(scope, input);
  EXPECT_TRUE(scope.ok());
  Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("MaxPool");
  auto builder = NodeBuilder(unique_name, "MaxPool")
                     .Input(_input)
                     .Attr("ksize", ksize)
                     .Attr("strides", strides)
                     .Attr("padding", padding)
                     .Attr("data_format", "NHWC");
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

static GraphDef CreateConvGraphDef() {
  Scope root = Scope::NewRootScope();
  Tensor input_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&input_data, 1.0f);
  Output input =
      ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
  Tensor filter_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&filter_data, 1.0f);
  Output filter =
      ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data));
  const std::vector<int> strides{1, 1, 1, 1};
  Output conv =
      BuildConv2DOps(root.WithOpName("conv"), input, filter, strides, "SAME");
  Output softmax = BuildSoftmaxOps(root.WithOpName("softmax"), conv);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static GraphDef CreatePoolGraphDef() {
  Scope root = Scope::NewRootScope();
  Tensor input_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&input_data, 1.0f);
  Output input =
      ops::Const(root.WithOpName("input"), Input::Initializer(input_data));
  Tensor filter_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&filter_data, 1.0f);
  Output filter =
      ops::Const(root.WithOpName("filter"), Input::Initializer(filter_data));
  const std::vector<int> ksize{1, 1, 1, 1};
  const std::vector<int> padding{0, 0, 0, 0};
  const std::vector<int> strides{1, 1, 1, 1};
  Output max_pool = BuildMaxPoolOps(root.WithOpName("maxpool"), input, ksize,
                                    strides, "SAME");
  Output softmax = BuildSoftmaxOps(root.WithOpName("softmax"), max_pool);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static const GraphTransferInfo::ConstNodeInfo* FindConstNodeInfo(
    const GraphTransferer& gt, const string& name) {
  for (const GraphTransferInfo::ConstNodeInfo& params :
       gt.GetGraphTransferInfo().const_node_info()) {
    if (params.name() == name) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferInfo::NodeInfo* FindNodeInfo(
    const GraphTransferer& gt, const string& name) {
  for (const GraphTransferInfo::NodeInfo& params :
       gt.GetGraphTransferInfo().node_info()) {
    if (params.name() == name) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferInfo::NodeInputInfo* FindNodeInputInfo(
    const GraphTransferer& gt, const int node_id) {
  for (const GraphTransferInfo::NodeInputInfo& params :
       gt.GetGraphTransferInfo().node_input_info()) {
    if (params.node_id() == node_id) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferInfo::NodeOutputInfo* FindNodeOutputInfo(
    const GraphTransferer& gt, const int node_id) {
  for (const GraphTransferInfo::NodeOutputInfo& params :
       gt.GetGraphTransferInfo().node_output_info()) {
    if (params.node_id() == node_id) {
      return &params;
    }
  }
  return nullptr;
}

static void SanityCheckNodes(const GraphTransferer& gt) {
  for (const GraphTransferInfo::NodeInfo& params :
       gt.GetGraphTransferInfo().node_info()) {
    if (params.input_count() > 0) {
      const GraphTransferInfo::NodeInputInfo* input_params =
          FindNodeInputInfo(gt, params.node_id());
      ASSERT_NE(nullptr, input_params);
      EXPECT_EQ(params.input_count(), input_params->node_input_size());
      EXPECT_EQ(params.node_id(), input_params->node_id());
      for (const GraphTransferInfo::NodeInput& node_input :
           input_params->node_input()) {
        EXPECT_GE(node_input.output_port(), 0);
      }
    }
    if (params.output_count() > 0) {
      const GraphTransferInfo::NodeOutputInfo* output_params =
          FindNodeOutputInfo(gt, params.node_id());
      ASSERT_NE(nullptr, output_params);
      EXPECT_EQ(params.output_count(), output_params->max_byte_size_size());
      EXPECT_EQ(params.node_id(), output_params->node_id());
      for (const int max_size : output_params->max_byte_size()) {
        EXPECT_GE(max_size, 0);
      }
    }
  }
}

TEST_F(GraphTransfererTest, LoadAddGraph) {
  GraphDef def = CreateAddGraphDef();
  ASSERT_TRUE(gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                     {}, std::vector<string>{NAME_A_PLUS_B},
                                     false)
                  .ok());
  SanityCheckNodes(gt_);

  const int const_node_count =
      gt_.GetGraphTransferInfo().const_node_info_size();
  ASSERT_EQ(2, const_node_count);
  const GraphTransferInfo::ConstNodeInfo* params_a =
      FindConstNodeInfo(gt_, NAME_A);
  ASSERT_TRUE(params_a != nullptr);
  EXPECT_EQ(NAME_A, params_a->name());
  ASSERT_EQ(4, params_a->shape_size());
  EXPECT_EQ(1, params_a->shape(0));
  EXPECT_EQ(1, params_a->shape(1));
  EXPECT_EQ(1, params_a->shape(2));
  EXPECT_EQ(1, params_a->shape(3));
  EXPECT_EQ(4, params_a->data().length());

  const GraphTransferInfo::ConstNodeInfo* params_b =
      FindConstNodeInfo(gt_, NAME_B);
  ASSERT_TRUE(params_b != nullptr);
  ASSERT_EQ(4, params_b->shape_size());
  EXPECT_EQ(1, params_b->shape(0));
  EXPECT_EQ(1, params_b->shape(1));
  EXPECT_EQ(1, params_b->shape(2));
  EXPECT_EQ(1, params_b->shape(3));
  EXPECT_EQ(4, params_b->data().length());
}

TEST_F(GraphTransfererTest, LoadAddGraphWithOutputTensorMap) {
  GraphDef def = CreateAddGraphDef();
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = 1.0f;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a};
  RemoteFusedGraphExecuteUtils::TensorShapeMap output_tensor_info;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
      def, inputs, {}, &output_tensor_info);
  ASSERT_TRUE(status.ok()) << status;
  const std::vector<string> output_node_names = {NAME_A_PLUS_B};
  status = gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                  inputs, output_node_names, false);
  TF_ASSERT_OK(status);
}

TEST_F(GraphTransfererTest, LoadConvGraph) {
  GraphDef def = CreateConvGraphDef();
  std::vector<std::pair<string, Tensor>> input_node_info_list;
  input_node_info_list.emplace_back(
      std::pair<string, Tensor>{"input", Tensor{DT_FLOAT, {1, 1, 1, 1}}});
  const std::vector<string> output_node_names = {"softmax"};
  ASSERT_TRUE(gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                     input_node_info_list, output_node_names,
                                     false)
                  .ok());
  SanityCheckNodes(gt_);
  const int const_node_count =
      gt_.GetGraphTransferInfo().const_node_info_size();
  ASSERT_EQ(2, const_node_count);
  const int op_node_count = gt_.GetGraphTransferInfo().node_info_size();
  ASSERT_EQ(4, op_node_count);
  const GraphTransferInfo::NodeInfo* params_conv = FindNodeInfo(gt_, "conv");
  ASSERT_TRUE(params_conv != nullptr);
  const int id = params_conv->node_id();
  EXPECT_GE(id, 0);
  EXPECT_EQ("Conv2D", params_conv->type_name());
  EXPECT_EQ(3, params_conv->input_count());
  EXPECT_EQ(1, params_conv->output_count());
  EXPECT_EQ(Padding::SAME, params_conv->padding_id());
}

TEST_F(GraphTransfererTest, LoadMaxPoolGraph) {
  GraphDef def = CreatePoolGraphDef();
  std::vector<std::pair<string, Tensor>> input_node_info_list;
  input_node_info_list.emplace_back(
      std::pair<string, Tensor>{"input", Tensor{DT_FLOAT, {1, 1, 1, 1}}});
  const std::vector<string> output_node_names = {"softmax"};
  ASSERT_TRUE(gt_.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                     input_node_info_list, output_node_names,
                                     false)
                  .ok());
  SanityCheckNodes(gt_);
  const int const_node_count =
      gt_.GetGraphTransferInfo().const_node_info_size();
  ASSERT_EQ(2, const_node_count);
  const int op_node_count = gt_.GetGraphTransferInfo().node_info_size();
  ASSERT_EQ(4, op_node_count);
  const GraphTransferInfo::NodeInfo* params_max_pool =
      FindNodeInfo(gt_, "maxpool");
  ASSERT_TRUE(params_max_pool != nullptr);
  const int id = params_max_pool->node_id();
  EXPECT_GE(id, 0);
  EXPECT_EQ("MaxPool", params_max_pool->type_name());
  EXPECT_EQ(3, params_max_pool->input_count());
  EXPECT_EQ(1, params_max_pool->output_count());
  EXPECT_EQ(Padding::SAME, params_max_pool->padding_id());
}

TEST(HexagonOpsDefinitions, CheckOpsDefinitions) {
  const IRemoteFusedGraphOpsDefinitions& ops_definitions =
      HexagonOpsDefinitions::getInstance();
  const int total_ops_count = ops_definitions.GetTotalOpsCount();
  EXPECT_GT(total_ops_count, 0);
}

TEST(GraphTransferer, LoadGraphFromProtoFile) {
  const IRemoteFusedGraphOpsDefinitions* ops_definitions =
      &TEST_GRAPH_TRANSFER_OPS_DEFINITIONS;
  string filename =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "core/example/testdata/parse_example_graph_def.pbtxt");
  std::vector<std::pair<string, Tensor>> input_node_info_list = {};
  std::vector<string> output_node_names = {};
  bool is_text_proto = true;

  // Keep following comments for debugging purpose for now
  // filename = "v3_stripped_quantized_graph_opt.pb";
  // input_node_info_list.emplace_back(
  // std::pair<string, Tensor>{"Mul", Tensor{DT_FLOAT, {1,299,299,3}}});
  // output_node_names.emplace_back("softmax");
  // is_text_proto = false;
  // ops_definitions = &HexagonOpsDefinitions::getInstance();

  GraphTransferer gt;
  gt.EnableStrictCheckMode(false);
  Status status = gt.LoadGraphFromProtoFile(
      *ops_definitions, filename, input_node_info_list, output_node_names,
      is_text_proto, false, true);
}

TEST_F(GraphTransfererTest, BuildRemoteFusedGraphDefAddGraph) {
  GraphDef def = CreateAddGraphDef();
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = 1.0f;
  std::pair<string, Tensor> input_node_info_b;
  input_node_info_b.first = NAME_B;
  input_node_info_b.second = Tensor(DT_FLOAT, {});
  input_node_info_b.second.scalar<float>()() = 10.0f;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a,
                                                      input_node_info_b};
  std::vector<string> outputs = {NAME_A_PLUS_B};

  GraphDef fused_graph_def = GraphTransferUtils::BuildFusedGraphDef(
      TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, "remote_fused_graph_execute_node",
      inputs, outputs, &def);

  EXPECT_EQ(3, fused_graph_def.node_size());
}

namespace {
// Just compares the max_byte_size attributes present.
void CompareGraphTransferInfo(const GraphTransferInfo& a,
                              const GraphTransferInfo& b) {
  EXPECT_EQ(a.node_output_info_size(), b.node_output_info_size());
  for (int i = 0; i < a.node_output_info_size(); ++i) {
    EXPECT_EQ(a.node_output_info(i).node_id(), b.node_output_info(i).node_id());
    EXPECT_EQ(a.node_output_info(i).max_byte_size_size(),
              b.node_output_info(i).max_byte_size_size());
    for (int j = 0; j < a.node_output_info(i).max_byte_size_size(); ++j) {
      EXPECT_EQ(a.node_output_info(i).max_byte_size(j),
                b.node_output_info(i).max_byte_size(j));
    }
  }
}
}  // anonymous namespace

TEST(GraphTransferer, LoadGraphFromProtoFileShapeInferenceSimple) {
  const IRemoteFusedGraphOpsDefinitions* ops_definitions =
      &TEST_GRAPH_TRANSFER_OPS_DEFINITIONS;
  string filename =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "core/example/testdata/parse_example_graph_def.pbtxt");
  std::vector<std::pair<string, Tensor>> input_node_info_list = {};
  std::vector<string> output_node_names = {};
  bool is_text_proto = true;

  // In order to run with a more complex graph uncomment the following lines
  // filename = "v3_stripped_quantized_graph_opt.pb";
  // input_node_info_list.emplace_back(
  // std::pair<string, Tensor>{"Mul", Tensor{DT_FLOAT, {1,299,299,3}}});
  // output_node_names.emplace_back("softmax");
  // is_text_proto = false;
  // ops_definitions = &HexagonOpsDefinitions::getInstance();

  // First compute using Shape inference.
  GraphTransferer si_gt;
  si_gt.EnableStrictCheckMode(false);
  bool shape_inference_for_unknown_shape = true;
  bool dry_run_for_unknown_shape = false;
  Status status1 = si_gt.LoadGraphFromProtoFile(
      *ops_definitions, filename, input_node_info_list, output_node_names,
      is_text_proto, shape_inference_for_unknown_shape,
      dry_run_for_unknown_shape);
  const GraphTransferInfo& si_graph_transfer_info =
      si_gt.GetGraphTransferInfo();

  // Now compute using dry run.
  GraphTransferer dr_gt;
  dr_gt.EnableStrictCheckMode(false);
  shape_inference_for_unknown_shape = false;
  dry_run_for_unknown_shape = true;
  Status status2 = dr_gt.LoadGraphFromProtoFile(
      *ops_definitions, filename, input_node_info_list, output_node_names,
      is_text_proto, shape_inference_for_unknown_shape,
      dry_run_for_unknown_shape);
  const GraphTransferInfo& dr_graph_transfer_info =
      dr_gt.GetGraphTransferInfo();

  // Now compare both of them.
  CompareGraphTransferInfo(si_graph_transfer_info, dr_graph_transfer_info);
}

}  // namespace tensorflow
