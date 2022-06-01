/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/grpc_worker_impl.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/credentials_factory.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::grpc::Channel;
using ::grpc::ChannelArguments;
using ::grpc::ChannelCredentials;
using ::grpc::ClientContext;
using ::tensorflow::data::testing::RangeSquareDataset;

constexpr const char kHostAddress[] = "localhost";
constexpr const char kProtocol[] = "grpc";

class GrpcWorkerImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK(SetUpDispatcherServer());
    TF_ASSERT_OK(SetUpWorkerServer());
    TF_ASSERT_OK(SetUpWorkerClientStub());
  }

  Status SetUpDispatcherServer() {
    experimental::DispatcherConfig config;
    config.set_protocol(kProtocol);
    TF_RETURN_IF_ERROR(NewDispatchServer(config, dispatcher_server_));
    return dispatcher_server_->Start();
  }

  Status SetUpWorkerServer() {
    experimental::WorkerConfig config;
    config.set_protocol(kProtocol);
    config.set_dispatcher_address(GetDispatcherAddress());
    config.set_worker_address(absl::StrCat(kHostAddress, ":%port%"));
    TF_RETURN_IF_ERROR(NewWorkerServer(config, worker_server_));
    return worker_server_->Start();
  }

  Status SetUpWorkerClientStub() {
    std::shared_ptr<ChannelCredentials> credentials;
    TF_RETURN_IF_ERROR(
        CredentialsFactory::CreateClientCredentials(kProtocol, &credentials));
    ChannelArguments args;
    args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
    std::shared_ptr<Channel> channel =
        ::grpc::CreateCustomChannel(GetWorkerAddress(), credentials, args);
    worker_client_stub_ = WorkerService::NewStub(channel);
    return OkStatus();
  }

  std::string GetDispatcherAddress() const {
    return absl::StrCat(kHostAddress, ":", dispatcher_server_->BoundPort());
  }

  std::string GetWorkerAddress() const {
    return absl::StrCat(kHostAddress, ":", worker_server_->BoundPort());
  }

  std::unique_ptr<DispatchGrpcDataServer> dispatcher_server_;
  std::unique_ptr<WorkerGrpcDataServer> worker_server_;
  std::unique_ptr<WorkerService::Stub> worker_client_stub_;
};

TEST_F(GrpcWorkerImplTest, GetWorkerTasks) {
  ClientContext ctx;
  GetWorkerTasksRequest req;
  GetWorkerTasksResponse resp;
  TF_ASSERT_OK(
      FromGrpcStatus(worker_client_stub_->GetWorkerTasks(&ctx, req, &resp)));
  EXPECT_EQ(resp.tasks_size(), 0);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
