/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.h"

#include "grpcpp/generic/generic_stub.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {
namespace eager {
namespace {
class GrpcEagerClient : public EagerClient {
 public:
  GrpcEagerClient(const tensorflow::SharedGrpcChannelPtr& channel,
                  ::grpc::CompletionQueue* cq)
      : stub_(channel), cq_(cq) {}
  ~GrpcEagerClient() override {}

#define CLIENT_METHOD(method)                                             \
  void method##Async(const method##Request* request,                      \
                     method##Response* response, StatusCallback done)     \
      override {                                                          \
    new RPCState<protobuf::Message>(                                      \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request, \
        response, std::move(done), nullptr, nullptr);                     \
  }

  CLIENT_METHOD(CreateContext);
  CLIENT_METHOD(Enqueue);
  CLIENT_METHOD(WaitQueueDone);
  CLIENT_METHOD(KeepAlive);
  CLIENT_METHOD(CloseContext);
  CLIENT_METHOD(RegisterFunction);
  CLIENT_METHOD(SendTensor);

#undef CLIENT_METHOD

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
};

class GrpcEagerClientCache : public EagerClientCache {
 public:
  explicit GrpcEagerClientCache(
      std::shared_ptr<tensorflow::GrpcChannelCache> cache)
      : next_round_robin_assignment_(0), cache_(cache), threads_(4) {}

  ~GrpcEagerClientCache() override { threads_.clear(); }

  EagerClient* GetClient(const string& target) override {
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      tensorflow::SharedGrpcChannelPtr shared =
          cache_->FindWorkerChannel(target);
      auto worker = std::unique_ptr<EagerClient>(new GrpcEagerClient(
          shared, threads_[AssignClientToThread(target)].completion_queue()));

      it = clients_.emplace(target, std::move(worker)).first;
    }

    return it->second.get();
  }

 private:
  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ GUARDED_BY(assignment_mu_);

  size_t AssignClientToThread(const string& target) {
    // Round-robin target assignment, but keeps the same target on the same
    // polling thread always, as this is important for gRPC performace
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

  class GrpcEagerClientThread {
   public:
    GrpcEagerClientThread() {
      thread_.reset(Env::Default()->StartThread(
          ThreadOptions(), "eager_client_thread", [this]() {
            void* tag;
            bool ok;
            while (completion_queue_.Next(&tag, &ok)) {
              GrpcClientCQTag* callback_tag =
                  static_cast<GrpcClientCQTag*>(tag);
              callback_tag->OnCompleted(ok);
            }
          }));
    }

    ~GrpcEagerClientThread() {
      completion_queue_.Shutdown();
      thread_.reset();
    }

    ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

   private:
    ::grpc::CompletionQueue completion_queue_;
    std::unique_ptr<Thread> thread_;
  };  // GrpcEagerClientThread

  std::shared_ptr<tensorflow::GrpcChannelCache> cache_;
  std::unordered_map<string, std::unique_ptr<EagerClient>> clients_;
  std::vector<GrpcEagerClientThread> threads_;
};

}  // namespace

EagerClientCache* NewGrpcEagerClientCache(
    std::shared_ptr<tensorflow::GrpcChannelCache> channel) {
  return new GrpcEagerClientCache(channel);
}

}  // namespace eager
}  // namespace tensorflow
