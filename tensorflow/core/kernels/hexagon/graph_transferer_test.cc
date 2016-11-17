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
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"
#include "tensorflow/core/kernels/hexagon/i_graph_transfer_ops_definitions.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

static const string NAME_A = "a";
static const string NAME_B = "b";

class GraphTransfererTest : public ::testing::Test {
 protected:
  void SetUp() final {
    SessionOptions session_options;
    session_options.env = Env::Default();
    _session = std::unique_ptr<Session>(NewSession(session_options));
  }

  std::unique_ptr<Session> _session;
};

static const std::vector<string> OP_TYPES{"INPUT", "OUTPUT", "Conv2D",
                                          "MaxPool"};

class TestGraphTransferOpsDefinitions : public IGraphTransferOpsDefinitions {
 public:
  int GetTotalOpsCount() const final { return OP_TYPES.size(); }
  int GetInputNodeOpId() const final { return GetOpIdFor("INPUT"); }
  int GetOutputNodeOpId() const final { return GetOpIdFor("OUTPUT"); }
  int GetOpIdFor(const string& op_type) const final {
    for (int i = 0; i < OP_TYPES.size(); ++i) {
      if (OP_TYPES[i] == op_type) {
        return i;
      }
    }
    return -1;
  }

 private:
} TEST_GRAPH_TRANSFER_OPS_DEFINITIONS;

static GraphDef CreateAddGraphDef() {
  Scope root = Scope::NewRootScope();
  ops::Output node_a = ops::Const(root.WithOpName(NAME_A), 1);
  ops::Output node_b = ops::Const(root.WithOpName(NAME_B), 2);
  ops::Output node_add = ops::Add(root.WithOpName("a_plus_b"), node_a, node_b);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static GraphDef CreateConvGraphDef() {
  Scope root = Scope::NewRootScope();
  Tensor input_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&input_data, 1.0f);
  ops::Output input =
      ops::Const(root.WithOpName("input"), ops::Input::Initializer(input_data));
  Tensor filter_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&filter_data, 1.0f);
  ops::Output filter = ops::Const(root.WithOpName("filter"),
                                  ops::Input::Initializer(filter_data));
  const std::vector<int> strides{1, 1, 1, 1};
  ops::Output conv =
      ops::Conv2D(root.WithOpName("conv"), input, filter, strides, "SAME");
  ops::Output softmax = ops::Softmax(root.WithOpName("softmax"), conv);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static GraphDef CreatePoolGraphDef() {
  Scope root = Scope::NewRootScope();
  Tensor input_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&input_data, 1.0f);
  ops::Output input =
      ops::Const(root.WithOpName("input"), ops::Input::Initializer(input_data));
  Tensor filter_data(DT_FLOAT, TensorShape({1, 1, 1, 1}));
  test::FillIota<float>(&filter_data, 1.0f);
  ops::Output filter = ops::Const(root.WithOpName("filter"),
                                  ops::Input::Initializer(filter_data));
  const std::vector<int> ksize{1, 1, 1, 1};
  const std::vector<int> padding{0, 0, 0, 0};
  const std::vector<int> strides{1, 1, 1, 1};
  ops::Output max_pool =
      ops::MaxPool(root.WithOpName("maxpool"), input, ksize, strides, "SAME");
  ops::Output softmax = ops::Softmax(root.WithOpName("softmax"), max_pool);
  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

static const GraphTransferer::ConstNodeTransferParams* FindConstNodeParams(
    const GraphTransferer& gt, const string& name) {
  for (const GraphTransferer::ConstNodeTransferParams& params :
       gt.GetConstNodeParams()) {
    if (params.name == name) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferer::NodeTransferParams* FindOpNodeParams(
    const GraphTransferer& gt, const string& name) {
  for (const GraphTransferer::NodeTransferParams& params :
       gt.GetOpNodeParams()) {
    if (params.name == name) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferer::NodeInputParams* FindNodeInputParams(
    const GraphTransferer& gt, const int node_id) {
  for (const GraphTransferer::NodeInputParams& params :
       gt.GetNodeInputParams()) {
    if (params.node_id == node_id) {
      return &params;
    }
  }
  return nullptr;
}

static const GraphTransferer::NodeOutputParams* FindNodeOutputParams(
    const GraphTransferer& gt, const int node_id) {
  for (const GraphTransferer::NodeOutputParams& params :
       gt.GetNodeOutputParams()) {
    if (params.node_id == node_id) {
      return &params;
    }
  }
  return nullptr;
}

static void SanityCheckNodes(const GraphTransferer& gt) {
  for (const GraphTransferer::NodeTransferParams& params :
       gt.GetOpNodeParams()) {
    if (params.inputs_size > 0) {
      const GraphTransferer::NodeInputParams* input_params =
          FindNodeInputParams(gt, params.node_id);
      ASSERT_NE(nullptr, input_params);
      EXPECT_EQ(params.inputs_size,
                input_params->input_node_id_and_output_port_list.size());
      EXPECT_EQ(params.node_id, input_params->node_id);
      for (const std::tuple<int, int>& pair :
           input_params->input_node_id_and_output_port_list) {
        EXPECT_GE(std::get<1>(pair), 0);
      }
    }
    if (params.outputs_size > 0) {
      const GraphTransferer::NodeOutputParams* output_params =
          FindNodeOutputParams(gt, params.node_id);
      ASSERT_NE(nullptr, output_params);
      EXPECT_EQ(params.outputs_size, output_params->max_sizes.size());
      EXPECT_EQ(params.node_id, output_params->node_id);
      for (const int max_size : output_params->max_sizes) {
        EXPECT_GE(max_size, 0);
      }
    }
  }
}

TEST_F(GraphTransfererTest, LoadAddGraph) {
  GraphDef def = CreateAddGraphDef();
  _session->Create(def);

  GraphTransferer gt;
  ASSERT_TRUE(
      gt.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def, {}, {})
          .ok());
  SanityCheckNodes(gt);

  const int const_node_count = gt.GetConstNodeParams().size();
  ASSERT_EQ(2, const_node_count);
  const GraphTransferer::ConstNodeTransferParams* params_a =
      FindConstNodeParams(gt, NAME_A);
  ASSERT_TRUE(params_a != nullptr);
  EXPECT_EQ(NAME_A, params_a->name);
  EXPECT_EQ(1, params_a->shape[0]);
  EXPECT_EQ(1, params_a->shape[1]);
  EXPECT_EQ(1, params_a->shape[2]);
  EXPECT_EQ(1, params_a->shape[3]);
  EXPECT_EQ(10, params_a->data_size);

  const GraphTransferer::ConstNodeTransferParams* params_b =
      FindConstNodeParams(gt, NAME_B);
  ASSERT_TRUE(params_b != nullptr);
  EXPECT_EQ(1, params_b->shape[0]);
  EXPECT_EQ(1, params_b->shape[1]);
  EXPECT_EQ(1, params_b->shape[2]);
  EXPECT_EQ(1, params_b->shape[3]);
  EXPECT_EQ(10, params_b->data_size);
}

TEST_F(GraphTransfererTest, LoadConvGraph) {
  GraphDef def = CreateConvGraphDef();
  _session->Create(def);

  GraphTransferer gt;
  const std::vector<string> input_node_names = {"input"};
  const std::vector<string> output_node_names = {"softmax"};
  ASSERT_TRUE(gt.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                    input_node_names, output_node_names)
                  .ok());
  SanityCheckNodes(gt);
  const int const_node_count = gt.GetConstNodeParams().size();
  ASSERT_EQ(2, const_node_count);
  const int op_node_count = gt.GetOpNodeParams().size();
  ASSERT_EQ(3, op_node_count);
  const GraphTransferer::NodeTransferParams* params_conv =
      FindOpNodeParams(gt, "conv");
  ASSERT_TRUE(params_conv != nullptr);
  const int id = params_conv->node_id;
  EXPECT_GE(id, 0);
  EXPECT_EQ("Conv2D", params_conv->type);
  EXPECT_EQ(3, params_conv->inputs_size);
  EXPECT_EQ(1, params_conv->outputs_size);
  EXPECT_EQ("NN_PAD_SAME", params_conv->padding);
}

TEST_F(GraphTransfererTest, LoadMaxPoolGraph) {
  GraphDef def = CreatePoolGraphDef();
  _session->Create(def);

  GraphTransferer gt;
  const std::vector<string> input_node_names = {"input"};
  const std::vector<string> output_node_names = {"softmax"};
  ASSERT_TRUE(gt.LoadGraphFromProto(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS, def,
                                    input_node_names, output_node_names)
                  .ok());
  SanityCheckNodes(gt);
  const int const_node_count = gt.GetConstNodeParams().size();
  ASSERT_EQ(2, const_node_count);
  const int op_node_count = gt.GetOpNodeParams().size();
  ASSERT_EQ(3, op_node_count);
  const GraphTransferer::NodeTransferParams* params_max_pool =
      FindOpNodeParams(gt, "maxpool");
  ASSERT_TRUE(params_max_pool != nullptr);
  const int id = params_max_pool->node_id;
  EXPECT_GE(id, 0);
  EXPECT_EQ("MaxPool", params_max_pool->type);
  EXPECT_EQ(3, params_max_pool->inputs_size);
  EXPECT_EQ(1, params_max_pool->outputs_size);
  EXPECT_EQ("NN_PAD_SAME", params_max_pool->padding);
}

TEST(HexagonOpsDefinitions, CheckOpsDefinitions) {
  const IGraphTransferOpsDefinitions& ops_definitions =
      HexagonOpsDefinitions::getInstance();
  const int total_ops_count = ops_definitions.GetTotalOpsCount();
  EXPECT_GT(total_ops_count, 0);
  const int input_op_id =
      ops_definitions.GetOpIdFor(IGraphTransferOpsDefinitions::INPUT_OP_NAME);
  EXPECT_GE(input_op_id, 0);
  EXPECT_EQ(input_op_id, ops_definitions.GetInputNodeOpId());
  const int output_op_id =
      ops_definitions.GetOpIdFor(IGraphTransferOpsDefinitions::OUTPUT_OP_NAME);
  EXPECT_GE(output_op_id, 0);
  EXPECT_EQ(output_op_id, ops_definitions.GetOutputNodeOpId());
}

TEST(GraphTransferer, LoadGraphFromProtoFile) {
  string filename =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "core/example/testdata/parse_example_graph_def.pbtxt");
  std::vector<string> input_node_names = {};
  std::vector<string> output_node_names = {};
  bool is_text_proto = true;
  // Keep following comments for debugging purpose for now
  // filename = "";
  // input_node_names = { "Mul" };
  // output_node_names = { "softmax" };
  // is_text_proto = false;
  GraphTransferer gt;
  Status status = gt.LoadGraphFromProtoFile(TEST_GRAPH_TRANSFER_OPS_DEFINITIONS,
                                            filename, input_node_names,
                                            output_node_names, is_text_proto);
  // TODO(satok): Uncomment following assert once we fix the loader problem
  // ASSERT_TRUE(status.ok()) << status;
}

}  // namespace tensorflow
