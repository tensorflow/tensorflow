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

#include "xla/pjrt/distributed/coordination/grpc_coordination_client.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "grpcpp/channel.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/generic/generic_stub.h"
#include "xla/pjrt/distributed/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_channel.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_state.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {
using tensorflow::BarrierRequest;
using tensorflow::BarrierResponse;
using tensorflow::CancelBarrierRequest;
using tensorflow::CancelBarrierResponse;
using tensorflow::DeleteKeyValueRequest;
using tensorflow::DeleteKeyValueResponse;
using tensorflow::GetAliveTasksRequest;
using tensorflow::GetAliveTasksResponse;
using tensorflow::GetKeyValueDirRequest;
using tensorflow::GetKeyValueDirResponse;
using tensorflow::GetKeyValueRequest;
using tensorflow::GetKeyValueResponse;
using tensorflow::HeartbeatRequest;
using tensorflow::HeartbeatResponse;
using tensorflow::IncrementKeyValueRequest;
using tensorflow::IncrementKeyValueResponse;
using tensorflow::InsertKeyValueRequest;
using tensorflow::InsertKeyValueResponse;
using tensorflow::PollForErrorRequest;
using tensorflow::PollForErrorResponse;
using tensorflow::RegisterTaskRequest;
using tensorflow::RegisterTaskResponse;
using tensorflow::ResetTaskRequest;
using tensorflow::ResetTaskResponse;
using tensorflow::ShutdownTaskRequest;
using tensorflow::ShutdownTaskResponse;
using tensorflow::TryGetKeyValueRequest;
using tensorflow::TryGetKeyValueResponse;
using tensorflow::WatchJobStateRequest;
using tensorflow::WatchJobStateResponse;

class GrpcCoordinationClientThread {
 public:
  GrpcCoordinationClientThread() {
    thread_.reset(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "coordination_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            VLOG(4) << "GrpcCoordinationClientThread got next tag";
            tsl::GrpcClientCQTag* callback_tag =
                static_cast<tsl::GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
            VLOG(4) << "GrpcCoordinationClientThread blocking for next tag";
          }
          VLOG(4) << "GrpcCoordinationClientThread exiting";
        }));
  }

  ~GrpcCoordinationClientThread() {
    completion_queue_.Shutdown();
    thread_.reset();
  }

  ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<tsl::Thread> thread_;
};

class GrpcCoordinationClient : public CoordinationClient {
 public:
  GrpcCoordinationClient(tsl::SharedGrpcChannelPtr channel,
                         ::grpc::CompletionQueue* cq, const std::string& target)
      : stub_(channel), cq_(cq), target_(target) {}
  GrpcCoordinationClient(tsl::SharedGrpcChannelPtr channel,
                         const std::string& target)
      : stub_(channel), target_(target) {
    client_thread_ = std::make_unique<GrpcCoordinationClientThread>();
    cq_ = client_thread_->completion_queue();
  }
  ~GrpcCoordinationClient() override = default;

  void RegisterTaskAsync(tsl::CallOptions* call_opts,
                         const RegisterTaskRequest* request,
                         RegisterTaskResponse* response,
                         tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/RegisterTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/false,
        &target_);
  }

  void ShutdownTaskAsync(tsl::CallOptions* call_opts,
                         const ShutdownTaskRequest* request,
                         ShutdownTaskResponse* response,
                         tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ShutdownTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ResetTaskAsync(const ResetTaskRequest* request,
                      ResetTaskResponse* response,
                      tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ResetTask", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void HeartbeatAsync(tsl::CallOptions* call_opts,
                      const HeartbeatRequest* request,
                      HeartbeatResponse* response,
                      tsl::StatusCallback done) override {
    // Different from other RPCs which do not retry by default, the Heartbeat
    // RPC should retry automatically to tolerate transient network issues.
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/Heartbeat", *request,
        response, std::move(done), call_opts, /*threadpool=*/nullptr,
        /*max_retries=*/3,
        /*fail_fast=*/true, &target_);
  }

  void WatchJobStateAsync(tsl::CallOptions* call_opts,
                          const WatchJobStateRequest* request,
                          WatchJobStateResponse* response,
                          tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/WatchJobState", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                           InsertKeyValueResponse* response,
                           tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/InsertKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueAsync(tsl::CallOptions* call_opts,
                        const GetKeyValueRequest* request,
                        GetKeyValueResponse* response,
                        tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetKeyValue", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void TryGetKeyValueAsync(const TryGetKeyValueRequest* request,
                           TryGetKeyValueResponse* response,
                           tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/TryGetKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void IncrementKeyValueAsync(const IncrementKeyValueRequest* request,
                              IncrementKeyValueResponse* response,
                              tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/IncrementKeyValue",
        *request, response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueDirAsync(const GetKeyValueDirRequest* request,
                           GetKeyValueDirResponse* response,
                           tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetKeyValueDir", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                           DeleteKeyValueResponse* response,
                           tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/DeleteKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void BarrierAsync(tsl::CallOptions* call_opts, const BarrierRequest* request,
                    BarrierResponse* response,
                    tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/Barrier", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void CancelBarrierAsync(const CancelBarrierRequest* request,
                          CancelBarrierResponse* response,
                          tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/CancelBarrier", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetAliveTasksAsync(const GetAliveTasksRequest* request,
                          GetAliveTasksResponse* response,
                          tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetAliveTasks", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void PollForErrorAsync(tsl::CallOptions* call_opts,
                         const PollForErrorRequest* request,
                         PollForErrorResponse* response,
                         tsl::StatusCallback done) override {
    new tsl::RPCState<tsl::protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/PollForError", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  const std::string target_;
  std::unique_ptr<GrpcCoordinationClientThread> client_thread_;
};

}  // namespace

CoordinationClient* NewGrpcCoordinationClient(
    std::shared_ptr<::grpc::Channel> channel) {
  return new GrpcCoordinationClient(channel, /*target=*/"coordination_service");
}

}  // namespace xla
