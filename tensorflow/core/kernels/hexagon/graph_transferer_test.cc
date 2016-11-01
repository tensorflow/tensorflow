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
#include "tensorflow/cc/ops/standard_ops.h"
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

static GraphDef CreateSmallGraphDef() {
  Scope root = Scope::NewRootScope();
  ops::Output node_a = ops::Const(root.WithOpName(NAME_A), 1);
  ops::Output node_b = ops::Const(root.WithOpName(NAME_B), 2);
  ops::Add(root.WithOpName("a_plus_b"), node_a, node_b);

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

TEST_F(GraphTransfererTest, LoadGraph) {
  GraphDef def = CreateSmallGraphDef();
  _session->Create(def);

  GraphTransferer gt;
  gt.LoadGraphFromProto(def);
  ASSERT_EQ(2, gt.GetConstNodeParams().size());
  const GraphTransferer::ConstNodeTransferParams* params_a =
      FindConstNodeParams(gt, NAME_A);
  ASSERT_TRUE(params_a != nullptr);
  EXPECT_TRUE(params_a->id > 0 && params_a->id <= 2);
  EXPECT_EQ(NAME_A, params_a->name);
  EXPECT_EQ(1, params_a->shape[0]);
  EXPECT_EQ(1, params_a->shape[1]);
  EXPECT_EQ(1, params_a->shape[2]);
  EXPECT_EQ(1, params_a->shape[3]);
  EXPECT_EQ(10, params_a->data_size);

  const GraphTransferer::ConstNodeTransferParams* params_b =
      FindConstNodeParams(gt, NAME_B);
  ASSERT_TRUE(params_b != nullptr);
  EXPECT_TRUE(params_b->id > 0 && params_b->id <= 2);
  EXPECT_EQ(1, params_b->shape[0]);
  EXPECT_EQ(1, params_b->shape[1]);
  EXPECT_EQ(1, params_b->shape[2]);
  EXPECT_EQ(1, params_b->shape[3]);
  EXPECT_EQ(10, params_b->data_size);
}

}  // namespace tensorflow
