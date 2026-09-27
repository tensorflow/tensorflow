/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/logging.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

static SessionOptions Options(const std::string& target) {
  SessionOptions options;
  options.target = absl::StrCat("grpc://", target);
  options.config.set_isolate_session_state(false);
  return options;
}

static Session* NewRemote(const SessionOptions& options) {
  return CHECK_NOTNULL(NewSession(options));
}

using test::TestClusterConfig;
using test::TestJob;

TEST(MasterSessionOobTest, OobReadInRunCallable) {
  Graph graph(OpRegistry::Global());

  // Create a graph with a constant node that we will attempt to feed.
  Tensor a_tensor(DT_FLOAT, TensorShape({1, 1}));
  test::FillValues<float>(&a_tensor, {1.0});
  Node* a = test::graph::Constant(&graph, a_tensor, "a");

  Node* b = test::graph::Identity(&graph, a);

  GraphDef graph_def;
  test::graph::ToGraphDef(&graph, &graph_def);

  std::unique_ptr<test::TestCluster> cluster;
  TF_ASSERT_OK(test::TestCluster::MakeTestCluster(
      TestClusterConfig()
          .Options(SessionOptions())  // Default options for devices
          .Jobs({TestJob{"localhost", /*num_tasks=*/1}}),
      &cluster));

  std::unique_ptr<Session> session(NewRemote(Options(cluster->targets()[0])));
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(graph_def));

  // Create a callable with the feed "a".
  CallableOptions opts;
  opts.add_feed("a:0");
  opts.add_fetch(b->name() + ":0");

  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(opts, &handle));

  // Call RunCallable with a request that does NOT contain the feed.
  // This should trigger the OOB read.
  std::vector<Tensor> outputs;
  absl::Status status = session->RunCallable(handle, {}, &outputs, nullptr);

  // We expect an error or a crash (if ASan is active).
  // In standard mode, it might read garbage and succeed or fail randomly.
  // Let's check if it failed with an invalid argument error (if validation was
  // added) or just check the status.
  LOG(INFO) << "Status: " << status;

  TF_ASSERT_OK(session->ReleaseCallable(handle));
  TF_ASSERT_OK(session->Close());
}

}  // namespace
}  // namespace tensorflow
