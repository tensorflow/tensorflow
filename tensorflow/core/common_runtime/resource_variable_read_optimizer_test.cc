// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

REGISTER_OP("VarHandleOp").Output("resource: resource");
REGISTER_OP("ReadVariableOp")
    .Input("resource: resource")
    .Attr("dtype: type")
    .Output("value: dtype");
REGISTER_OP("_UnsafeReadVariable")
    .Input("resource: resource")
    .Attr("dtype: type")
    .Output("value: dtype");

TEST(ReadReplaceTest, Simple) {
  std::unique_ptr<Graph> g(new Graph(OpRegistry::Global()));
  Node* handle;
  TF_ASSERT_OK(NodeBuilder("handle", "VarHandleOp").Finalize(g.get(), &handle));
  Node* read;
  TF_ASSERT_OK(NodeBuilder("read", "ReadVariableOp")
                   .Input(handle)
                   .Attr("dtype", DT_FLOAT)
                   .Finalize(g.get(), &read));
  Node* send;
  TF_ASSERT_OK(NodeBuilder("send", "_Send")
                   .Input(read)
                   .Attr("recv_device", "")
                   .Attr("send_device", "")
                   .Attr("send_device_incarnation", 0)
                   .Attr("tensor_name", "")
                   .Finalize(g.get(), &send));
  Node* other_send;
  TF_ASSERT_OK(NodeBuilder("other_send", "_Send")
                   .Input(read)
                   .Attr("recv_device", "")
                   .Attr("send_device", "")
                   .Attr("send_device_incarnation", 0)
                   .Attr("tensor_name", "")
                   .Finalize(g.get(), &other_send));
  GraphOptimizationPassOptions opts;
  opts.graph = &g;
  OptimizationPassRegistry::Global()->RunGrouping(
      OptimizationPassRegistry::POST_REWRITE_FOR_EXEC, opts);
  int found_reads = 0;
  int found_unsafe_reads = 0;
  for (const Node* n : g->nodes()) {
    if (n->type_string() == "ReadVariableOp") {
      found_reads++;
    } else if (n->type_string() == "_UnsafeReadVariable") {
      found_unsafe_reads++;
      ASSERT_EQ(n->num_inputs(), 1);
      const Node* inp;
      TF_ASSERT_OK(n->input_node(0, &inp));
      EXPECT_EQ(inp->name(), handle->name());
      ASSERT_EQ(n->out_edges().size(), 2);
      for (Node* out : n->out_nodes()) {
        EXPECT_EQ(out->type_string(), "_Send");
      }
    }
  }
  EXPECT_EQ(found_reads, 0);
  EXPECT_EQ(found_unsafe_reads, 1);
}

}  // namespace
}  // namespace tensorflow
