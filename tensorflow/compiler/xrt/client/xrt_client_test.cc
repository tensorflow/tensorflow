/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xrt/client/xrt_grpc_eager_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

class XrtClientTest : public ::testing::Test {
 protected:
  XrtClientTest() {
    string binary_path =
        absl::StrCat(testing::TensorFlowSrcRoot(),
                     "/compiler/xrt/client/xrt_testlib_server");

    TF_CHECK_OK(test::TestCluster::MakeTestCluster(
        binary_path, SessionOptions(), /*n=*/1, &cluster_));

    CHECK_EQ(cluster_->targets().size(), 1);
    JobDef* job = cluster_def_.add_job();
    job->set_name("localhost");
    (*job->mutable_tasks())[0] = cluster_->targets()[0];
  }
  std::unique_ptr<test::TestCluster> cluster_;
  ClusterDef cluster_def_;
};

// Test some connection basics using XrtGrpcEagerClient directly.
TEST_F(XrtClientTest, XrtGrpcEagerClientWorks) {
  ChannelCreationFunction channel_func =
      ConvertToChannelCreationFunction(NewHostPortGrpcChannel);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<GrpcChannelCache> channel_cache,
                          GetGrpcChannelCache(cluster_def_, channel_func));
  XrtGrpcEagerClientCache client_cache(channel_cache);

  TF_ASSERT_OK_AND_ASSIGN(
      XrtGrpcEagerClient * client,
      client_cache.GetClient("/job:localhost/task:0/replica:0"));

  // Create and destroy a context to verify we can make RPCs.
  eager::CreateContextRequest request;
  ServerDef* server_def = request.mutable_server_def();
  *server_def->mutable_cluster() = cluster_def_;
  server_def->set_job_name("localhost");
  server_def->set_protocol("grpc");
  request.set_keep_alive_secs(60);
  request.set_rendezvous_id(random::New64());

  eager::CreateContextResponse create_response;
  TF_ASSERT_OK(client->SyncCall(&XrtGrpcEagerClient::CreateContextAsync,
                                &request, &create_response));

  eager::CloseContextRequest close_request;
  close_request.set_context_id(create_response.context_id());

  eager::CloseContextResponse close_response;
  TF_ASSERT_OK(client->SyncCall(&XrtGrpcEagerClient::CloseContextAsync,
                                &close_request, &close_response));
}

}  // namespace tensorflow
