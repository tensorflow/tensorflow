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
#include "tensorflow/core/lib/core/status.h"
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

TEST_F(GraphTransfererTest, LoadAddGraph) {
  GraphDef def = CreateAddGraphDef();
  _session->Create(def);

  GraphTransferer gt;
  gt.LoadGraphFromProto(def);
  const int const_node_count = gt.GetConstNodeParams().size();
  ASSERT_EQ(2, const_node_count);
  const GraphTransferer::ConstNodeTransferParams* params_a =
      FindConstNodeParams(gt, NAME_A);
  ASSERT_TRUE(params_a != nullptr);
  EXPECT_TRUE(params_a->id > 0 && params_a->id <= const_node_count);
  EXPECT_EQ(NAME_A, params_a->name);
  EXPECT_EQ(1, params_a->shape[0]);
  EXPECT_EQ(1, params_a->shape[1]);
  EXPECT_EQ(1, params_a->shape[2]);
  EXPECT_EQ(1, params_a->shape[3]);
  EXPECT_EQ(10, params_a->data_size);

  const GraphTransferer::ConstNodeTransferParams* params_b =
      FindConstNodeParams(gt, NAME_B);
  ASSERT_TRUE(params_b != nullptr);
  EXPECT_TRUE(params_b->id > 0 && params_b->id <= const_node_count);
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
  gt.LoadGraphFromProto(def);
  const int const_node_count = gt.GetConstNodeParams().size();
  ASSERT_EQ(3, const_node_count);
  const int op_node_count = gt.GetOpNodeParams().size();
  ASSERT_EQ(1, op_node_count);
  const GraphTransferer::NodeTransferParams* params_conv =
      FindOpNodeParams(gt, "conv");
  ASSERT_TRUE(params_conv != nullptr);
  const int id = params_conv->id;
  EXPECT_TRUE(id > 0 && id <= (const_node_count + op_node_count));
  EXPECT_EQ("Conv2D", params_conv->type);
  EXPECT_EQ(3, params_conv->inputs_size);
  EXPECT_EQ(1, params_conv->outputs_size);
  EXPECT_EQ("NN_PAD_SAME", params_conv->padding);
}

TEST_F(GraphTransfererTest, LoadMaxPoolGraph) {
  GraphDef def = CreatePoolGraphDef();
  _session->Create(def);

  GraphTransferer gt;
  gt.LoadGraphFromProto(def);
  const int const_node_count = gt.GetConstNodeParams().size();
  ASSERT_EQ(3, const_node_count);
  const int op_node_count = gt.GetOpNodeParams().size();
  ASSERT_EQ(1, op_node_count);
  const GraphTransferer::NodeTransferParams* params_max_pool =
      FindOpNodeParams(gt, "maxpool");
  ASSERT_TRUE(params_max_pool != nullptr);
  const int id = params_max_pool->id;
  EXPECT_TRUE(id > 0 && id <= (const_node_count + op_node_count));
  EXPECT_EQ("MaxPool", params_max_pool->type);
  EXPECT_EQ(3, params_max_pool->inputs_size);
  EXPECT_EQ(1, params_max_pool->outputs_size);
  EXPECT_EQ("NN_PAD_SAME", params_max_pool->padding);
}
}  // namespace tensorflow
