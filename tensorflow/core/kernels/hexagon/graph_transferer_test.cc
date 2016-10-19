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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/hexagon/graph_transferer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

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
  ops::Output node_a = ops::Const(root.WithOpName("a"), 1);
  ops::Output node_b = ops::Const(root.WithOpName("b"), 2);
  ops::Add(root.WithOpName("a_plus_b"), node_a, node_b);

  GraphDef def;
  TF_CHECK_OK(root.ToGraphDef(&def));
  return def;
}

TEST_F(GraphTransfererTest, LoadGraph) {
  GraphDef def = CreateSmallGraphDef();
  _session->Create(def);

  GraphTransferer gt;
  gt.LoadGraphFromProto(&def);
}

}  // namespace tensorflow
