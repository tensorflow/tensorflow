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
#include "tensorflow/core/data/service/grpc_dispatcher_impl.h"

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
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/types.h"
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

class GrpcDispatcherImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK(SetUpDispatcherServer());
    TF_ASSERT_OK(SetUpDispatcherClientStub());
  }

  Status SetUpDispatcherServer() {
    experimental::DispatcherConfig config;
    config.set_protocol(kProtocol);
    TF_RETURN_IF_ERROR(NewDispatchServer(config, dispatcher_server_));
    return dispatcher_server_->Start();
  }

  Status SetUpDispatcherClientStub() {
    std::shared_ptr<ChannelCredentials> credentials;
    TF_RETURN_IF_ERROR(
        CredentialsFactory::CreateClientCredentials(kProtocol, &credentials));
    ChannelArguments args;
    args.SetMaxReceiveMessageSize(std::numeric_limits<int32>::max());
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, true);
    std::shared_ptr<Channel> channel =
        ::grpc::CreateCustomChannel(GetDispatcherAddress(), credentials, args);
    dispatcher_client_stub_ = DispatcherService::NewStub(channel);
    return Status::OK();
  }

  std::string GetDispatcherAddress() const {
    return absl::StrCat(kHostAddress, ":", dispatcher_server_->BoundPort());
  }

  StatusOr<GetOrRegisterDatasetResponse> RegisterDataset() {
    GetOrRegisterDatasetRequest request;
    GetOrRegisterDatasetResponse response;
    *request.mutable_dataset() = RangeSquareDataset(/*range=*/10);
    ClientContext context;
    TF_RETURN_IF_ERROR(
        FromGrpcStatus(dispatcher_client_stub_->GetOrRegisterDataset(
            &context, request, &response)));
    return response;
  }

  StatusOr<GetOrCreateJobResponse> CreateJob(const int64 dataset_id) {
    GetOrCreateJobRequest request;
    GetOrCreateJobResponse response;
    request.set_dataset_id(dataset_id);
    request.set_processing_mode(ProcessingModeDef::PARALLEL_EPOCHS);
    ClientContext context;
    TF_RETURN_IF_ERROR(FromGrpcStatus(
        dispatcher_client_stub_->GetOrCreateJob(&context, request, &response)));
    return response;
  }

  StatusOr<WorkerHeartbeatResponse> WorkerHeartbeat() {
    WorkerHeartbeatRequest request;
    WorkerHeartbeatResponse response;
    request.set_worker_address(kHostAddress);
    request.set_transfer_address(kHostAddress);
    ClientContext client_ctx;
    TF_RETURN_IF_ERROR(FromGrpcStatus(dispatcher_client_stub_->WorkerHeartbeat(
        &client_ctx, request, &response)));
    return response;
  }

  StatusOr<ClientHeartbeatResponse> ClientHeartbeat(const int64 job_client_id) {
    ClientHeartbeatRequest request;
    ClientHeartbeatResponse response;
    request.set_job_client_id(job_client_id);
    ClientContext client_ctx;
    TF_RETURN_IF_ERROR(FromGrpcStatus(dispatcher_client_stub_->ClientHeartbeat(
        &client_ctx, request, &response)));
    return response;
  }

  std::unique_ptr<DispatchGrpcDataServer> dispatcher_server_;
  std::unique_ptr<DispatcherService::Stub> dispatcher_client_stub_;
};

TEST_F(GrpcDispatcherImplTest, WorkerHeartbeat) {
  TF_ASSERT_OK_AND_ASSIGN(GetOrRegisterDatasetResponse dataset_response,
                          RegisterDataset());
  TF_ASSERT_OK(CreateJob(dataset_response.dataset_id()).status());
  TF_ASSERT_OK_AND_ASSIGN(WorkerHeartbeatResponse worker_response,
                          WorkerHeartbeat());
  ASSERT_EQ(worker_response.new_tasks().size(), 1);
  EXPECT_EQ(worker_response.new_tasks(0).dataset_id(),
            dataset_response.dataset_id());
}

TEST_F(GrpcDispatcherImplTest, ClientHeartbeat) {
  TF_ASSERT_OK_AND_ASSIGN(GetOrRegisterDatasetResponse dataset_response,
                          RegisterDataset());
  TF_ASSERT_OK_AND_ASSIGN(GetOrCreateJobResponse job_response,
                          CreateJob(dataset_response.dataset_id()));
  TF_ASSERT_OK(WorkerHeartbeat().status());
  TF_ASSERT_OK_AND_ASSIGN(ClientHeartbeatResponse client_response,
                          ClientHeartbeat(job_response.job_client_id()));
  ASSERT_EQ(client_response.task_info().size(), 1);
  EXPECT_EQ(client_response.task_info(0).worker_address(), kHostAddress);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
