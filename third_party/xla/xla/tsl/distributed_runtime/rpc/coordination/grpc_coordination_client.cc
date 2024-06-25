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

#include "xla/tsl/distributed_runtime/rpc/coordination/grpc_coordination_client.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grpcpp/channel.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/generic/generic_stub.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_client.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_channel.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_state.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/protobuf/coordination_service.pb.h"

namespace tsl {
namespace {
using tensorflow::BarrierRequest;
using tensorflow::BarrierResponse;
using tensorflow::CancelBarrierRequest;
using tensorflow::CancelBarrierResponse;
using tensorflow::DeleteKeyValueRequest;
using tensorflow::DeleteKeyValueResponse;
using tensorflow::GetKeyValueDirRequest;
using tensorflow::GetKeyValueDirResponse;
using tensorflow::GetKeyValueRequest;
using tensorflow::GetKeyValueResponse;
using tensorflow::GetTaskStateRequest;
using tensorflow::GetTaskStateResponse;
using tensorflow::HeartbeatRequest;
using tensorflow::HeartbeatResponse;
using tensorflow::InsertKeyValueRequest;
using tensorflow::InsertKeyValueResponse;
using tensorflow::RegisterTaskRequest;
using tensorflow::RegisterTaskResponse;
using tensorflow::ReportErrorToServiceRequest;
using tensorflow::ReportErrorToServiceResponse;
using tensorflow::ReportErrorToTaskRequest;
using tensorflow::ReportErrorToTaskResponse;
using tensorflow::ResetTaskRequest;
using tensorflow::ResetTaskResponse;
using tensorflow::ShutdownTaskRequest;
using tensorflow::ShutdownTaskResponse;
using tensorflow::TryGetKeyValueRequest;
using tensorflow::TryGetKeyValueResponse;
using tensorflow::WaitForAllTasksRequest;
using tensorflow::WaitForAllTasksResponse;

class GrpcCoordinationClientThread {
 public:
  GrpcCoordinationClientThread() {
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), "coordination_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            VLOG(4) << "GrpcCoordinationClientThread got next tag";
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
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
  std::unique_ptr<Thread> thread_;
};

class GrpcCoordinationClient : public CoordinationClient {
 public:
  GrpcCoordinationClient(SharedGrpcChannelPtr channel,
                         ::grpc::CompletionQueue* cq, const std::string& target)
      : stub_(channel), cq_(cq), target_(target) {}
  GrpcCoordinationClient(SharedGrpcChannelPtr channel,
                         const std::string& target)
      : stub_(channel), target_(target) {
    client_thread_ = std::make_unique<GrpcCoordinationClientThread>();
    cq_ = client_thread_->completion_queue();
  }
  ~GrpcCoordinationClient() override = default;

  void RegisterTaskAsync(CallOptions* call_opts,
                         const RegisterTaskRequest* request,
                         RegisterTaskResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/RegisterTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/false,
        &target_);
  }

  void WaitForAllTasksAsync(const WaitForAllTasksRequest* request,
                            WaitForAllTasksResponse* response,
                            StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/WaitForAllTasks",
        *request, response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ShutdownTaskAsync(CallOptions* call_opts,
                         const ShutdownTaskRequest* request,
                         ShutdownTaskResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ShutdownTask", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ResetTaskAsync(const ResetTaskRequest* request,
                      ResetTaskResponse* response,
                      StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ResetTask", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void HeartbeatAsync(CallOptions* call_opts, const HeartbeatRequest* request,
                      HeartbeatResponse* response,
                      StatusCallback done) override {
    // Different from other RPCs which do not retry by default, the Heartbeat
    // RPC should retry automatically to tolerate transient network issues.
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/Heartbeat", *request,
        response, std::move(done), call_opts, /*threadpool=*/nullptr,
        /*max_retries=*/3,
        /*fail_fast=*/true, &target_);
  }

  void ReportErrorToTaskAsync(CallOptions* call_opts,
                              const ReportErrorToTaskRequest* request,
                              ReportErrorToTaskResponse* response,
                              StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ReportErrorToTask",
        *request, response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void ReportErrorToServiceAsync(const ReportErrorToServiceRequest* request,
                                 ReportErrorToServiceResponse* response,
                                 StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ReportErrorToService",
        *request, response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetTaskStateAsync(const GetTaskStateRequest* request,
                         GetTaskStateResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetTaskState", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                           InsertKeyValueResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/InsertKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueAsync(CallOptions* call_opts,
                        const GetKeyValueRequest* request,
                        GetKeyValueResponse* response,
                        StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetKeyValue", *request,
        response, std::move(done), call_opts,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void TryGetKeyValueAsync(const TryGetKeyValueRequest* request,
                           TryGetKeyValueResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/TryGetKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueDirAsync(const GetKeyValueDirRequest* request,
                           GetKeyValueDirResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetKeyValueDir", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void DeleteKeyValueAsync(const DeleteKeyValueRequest* request,
                           DeleteKeyValueResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/DeleteKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void BarrierAsync(const BarrierRequest* request, BarrierResponse* response,
                    StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/Barrier", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void CancelBarrierAsync(const CancelBarrierRequest* request,
                          CancelBarrierResponse* response,
                          StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/CancelBarrier", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  const std::string target_;
  std::unique_ptr<GrpcCoordinationClientThread> client_thread_;
};

class GrpcCoordinationClientCache : public CoordinationClientCache {
 public:
  explicit GrpcCoordinationClientCache(
      std::shared_ptr<GrpcChannelCache> channel_cache)
      : next_round_robin_assignment_(0),
        channel_cache_(channel_cache),
        threads_(4) {}

  ~GrpcCoordinationClientCache() override = default;

  CoordinationClient* GetClient(const std::string& target) override {
    absl::MutexLock l(&clients_mu_);
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (channel == nullptr) {
        VLOG(2) << "Coordination client for target " << target << " not found.";
      }
      int assigned_index = AssignClientToThread(target);
      auto coord_client = std::make_unique<GrpcCoordinationClient>(
          channel, threads_[assigned_index].completion_queue(), target);
      it = clients_.emplace(target, std::move(coord_client)).first;
    }
    return it->second.get();
  }

  std::unique_ptr<CoordinationClient> GetOwnedClient(
      const std::string& target) override {
    SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
    if (channel == nullptr) {
      VLOG(2) << "Coordination client for target " << target << " not found.";
    }
    return std::make_unique<GrpcCoordinationClient>(channel, target);
  }

 private:
  absl::Mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      ABSL_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ ABSL_GUARDED_BY(assignment_mu_);

  size_t AssignClientToThread(const std::string& target) {
    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    absl::MutexLock l(&assignment_mu_);
    auto it = target_assignments_.find(target);
    if (it == target_assignments_.end()) {
      it = target_assignments_
               .insert(std::make_pair(
                   target, (next_round_robin_assignment_++) % threads_.size()))
               .first;
    }
    return it->second;
  }

  std::shared_ptr<GrpcChannelCache> channel_cache_;
  mutable absl::Mutex clients_mu_;
  std::unordered_map<std::string, std::unique_ptr<CoordinationClient>> clients_
      ABSL_GUARDED_BY(clients_mu_);
  std::vector<GrpcCoordinationClientThread> threads_;
};

}  // namespace

CoordinationClientCache* NewGrpcCoordinationClientCache(
    std::shared_ptr<GrpcChannelCache> channel_cache) {
  return new GrpcCoordinationClientCache(channel_cache);
}

CoordinationClient* NewGrpcCoordinationClient(
    std::shared_ptr<::grpc::Channel> channel) {
  // TODO(hanyangtay): Pass in the logical task name for better logging.
  return new GrpcCoordinationClient(
      channel, /*target=*/"unknown_target_for_coordination_leader");
}

}  // namespace tsl
