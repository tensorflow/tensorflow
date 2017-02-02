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

#include "tensorflow/core/common_runtime/gpu/gpu_stream_util.h"

#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GpuStreamUtilTest : public OpsTestBase {
 protected:
};

TEST_F(GpuStreamUtilTest, BogusOpts) {
  auto root = Scope::NewRootScope().ExitOnError();
  Graph g(OpRegistry::Global());
  root.ToGraph(&g);
  std::unordered_map<int, int> node_to_stream_id;
  gpu_stream_util::AssignStreamsOpts opts;
  Status status;
  status = gpu_stream_util::AssignStreams(nullptr, opts, &node_to_stream_id);
  EXPECT_FALSE(status.ok());
  status = gpu_stream_util::AssignStreams(&g, opts, nullptr);
  EXPECT_FALSE(status.ok());
  opts.max_streams = 0;
  status = gpu_stream_util::AssignStreams(&g, opts, &node_to_stream_id);
  EXPECT_FALSE(status.ok());
  opts.max_streams = 1;
  opts.compute_stream = 5;
  status = gpu_stream_util::AssignStreams(&g, opts, &node_to_stream_id);
  EXPECT_FALSE(status.ok());
}

TEST_F(GpuStreamUtilTest, EmptyGraph) {
  auto root = Scope::NewRootScope().ExitOnError();
  Graph g(OpRegistry::Global());
  root.ToGraph(&g);
  std::unordered_map<int, int> node_to_stream_id;
  gpu_stream_util::AssignStreamsOpts opts;
  TF_ASSERT_OK(gpu_stream_util::AssignStreams(&g, opts, &node_to_stream_id));
  EXPECT_EQ(2, node_to_stream_id.size());  // _SOURCE and _SINK
}

TEST_F(GpuStreamUtilTest, SimpleGraphOneStream) {
  auto root = Scope::NewRootScope().ExitOnError();
  ops::MatMul(root, {}, {});
  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&g));

  std::unordered_map<int, int> node_to_stream_id;
  gpu_stream_util::AssignStreamsOpts opts;
  TF_ASSERT_OK(gpu_stream_util::AssignStreams(&g, opts, &node_to_stream_id));

  // There should be 5 nodes assigned.
  EXPECT_EQ(5, node_to_stream_id.size());

  // All of them should have stream 0.
  for (const auto& it : node_to_stream_id) {
    EXPECT_EQ(0, it.second);
  }
}

TEST_F(GpuStreamUtilTest, SimpleGraphManyStreams) {
  auto root = Scope::NewRootScope().ExitOnError();
  ops::MatMul(root, {}, {});
  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&g));

  std::unordered_map<int, int> node_to_stream_id;
  gpu_stream_util::AssignStreamsOpts opts;
  opts.max_streams = 3;
  TF_ASSERT_OK(gpu_stream_util::AssignStreams(&g, opts, &node_to_stream_id));

  // There should be 5 nodes assigned.
  EXPECT_EQ(5, node_to_stream_id.size());

  // All of them should have a stream in the range [0..max_streams).
  for (const auto& it : node_to_stream_id) {
    EXPECT_GE(it.second, 0);
    EXPECT_LT(it.second, opts.max_streams);
  }
}

TEST_F(GpuStreamUtilTest, StreamOverrides) {
  auto root = Scope::NewRootScope().ExitOnError();
  ops::_Recv(root.WithOpName("input"), DT_FLOAT, "input", "/cpu:0", 0,
             "/gpu:0");
  Output n = ops::MatMul(root, {}, {});
  ops::_Send(root.WithOpName("output"), n, "output", "/gpu:0", 0, "/cpu:0");
  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&g));

  // Perform stream assignment using a large number of streams, but with
  // op types constrained to specific streams.
  std::unordered_map<int, int> node_to_stream_id;
  gpu_stream_util::AssignStreamsOpts opts;
  opts.max_streams = 100;
  opts.const_stream = 90;
  opts.send_stream = 91;
  opts.recv_stream = 92;
  opts.compute_stream = 93;
  TF_ASSERT_OK(gpu_stream_util::AssignStreams(&g, opts, &node_to_stream_id));

  // There should be 7 nodes assigned.
  EXPECT_EQ(7, node_to_stream_id.size());  // including _SOURCE and _SINK

  // Nodes should be assigned to streams by op type.
  for (const auto& it : node_to_stream_id) {
    Node* n = g.FindNodeId(it.first);
    const string& op = n->type_string();
    const int stream = it.second;
    if (op == "Const") {
      EXPECT_EQ(stream, 90);
    } else if (op == "_Send") {
      EXPECT_EQ(stream, 91);
    } else if (op == "_Recv") {
      EXPECT_EQ(stream, 92);
    } else {  // Compute.
      EXPECT_EQ(stream, 93);
    }
  }
}

}  // namespace
}  // namespace tensorflow
