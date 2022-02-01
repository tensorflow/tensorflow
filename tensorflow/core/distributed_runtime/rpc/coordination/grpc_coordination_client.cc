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

#include "tensorflow/core/distributed_runtime/rpc/coordination/grpc_coordination_client.h"

#include <string>
#include <utility>

#include "tensorflow/core/distributed_runtime/coordination/coordination_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"

namespace tensorflow {
namespace {

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
  ~GrpcCoordinationClient() override {}

  void RegisterWorkerAsync(CallOptions* call_opts,
                           const RegisterWorkerRequest* request,
                           RegisterWorkerResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/RegisterWorker", *request,
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

  void HeartbeatAsync(const HeartbeatRequest* request,
                      HeartbeatResponse* response,
                      StatusCallback done) override {
    // Different from other RPCs which do not retry by default, the Heartbeat
    // RPC should retry automatically to tolerate transient network issues.
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/Heartbeat", *request,
        response, std::move(done),
        /*call_opts=*/nullptr, /*threadpool=*/nullptr, /*max_retries=*/3,
        /*fail_fast=*/true, &target_);
  }

  void ReportErrorToAgentAsync(const ReportErrorToAgentRequest* request,
                               ReportErrorToAgentResponse* response,
                               StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/ReportErrorToAgent",
        *request, response, std::move(done), /*call_opts=*/nullptr,
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

  void InsertKeyValueAsync(const InsertKeyValueRequest* request,
                           InsertKeyValueResponse* response,
                           StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/InsertKeyValue", *request,
        response, std::move(done), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);
  }

  void GetKeyValueAsync(const GetKeyValueRequest* request,
                        GetKeyValueResponse* response,
                        StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.CoordinationService/GetKeyValue", *request,
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

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
  const string target_;
  std::unique_ptr<GrpcCoordinationClientThread> client_thread_;
};

class GrpcCoordinationClientCache : public CoordinationClientCache {
 public:
  explicit GrpcCoordinationClientCache(
      std::shared_ptr<GrpcChannelCache> channel_cache)
      : next_round_robin_assignment_(0),
        channel_cache_(channel_cache),
        threads_(4) {}

  ~GrpcCoordinationClientCache() override {}

  CoordinationClient* GetClient(const string& target) override {
    mutex_lock l(clients_mu_);
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
      const string& target) override {
    SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
    if (channel == nullptr) {
      VLOG(2) << "Coordination client for target " << target << " not found.";
    }
    return std::make_unique<GrpcCoordinationClient>(channel, target);
  }

 private:
  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      TF_GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ TF_GUARDED_BY(assignment_mu_);

  size_t AssignClientToThread(const string& target) {
    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performance
    mutex_lock lock(assignment_mu_);
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
  mutable mutex clients_mu_;
  std::unordered_map<std::string, std::unique_ptr<CoordinationClient>> clients_
      TF_GUARDED_BY(clients_mu_);
  std::vector<GrpcCoordinationClientThread> threads_;
};

}  // namespace

CoordinationClientCache* NewGrpcCoordinationClientCache(
    std::shared_ptr<GrpcChannelCache> channel_cache) {
  return new GrpcCoordinationClientCache(channel_cache);
}

}  // namespace tensorflow
