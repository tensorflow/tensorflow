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
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace eager {
namespace {

/*
 * Setting environment variable "TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE" to
 * true will turn on asynchronous execution of remote op. It means that when
 * executing an op on a remote worker, client will not block on waiting
 * for the response anymore. Using follow code as example:
 *
 * with tf.device('worker:0'):
 *   a = tf.matmul(...)
 *   b = tf.matmul(...)
 * logging.into('Requests sent')    # Probably not executed yet
 * logging.info('b: %s', b.numpy()) # Block until 'b' finished.
 *
 * Streaming RPC will preserve order as well. So 'a' must be executed before
 * 'b' on 'worker:0'.
 *
 * When turning on this feature, you should explicitly wait for some result
 * from remote workers at the end of you python program. Otherwise, client may
 * shutdown remote workers without waiting all pending ops.
 *
 * TODO(fishx): When exiting client, make sure all pending ops on remote workers
 * are finished.
 *
 * TODO(b/139210648): Move this comment to eager/execute.py when this feature is
 * on by default.
 */
bool EnableStreaming() {
  bool result;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE",
                                 true, &result));
  return result;
}

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
        response, std::move(done), nullptr, nullptr, /*max_retries=*/0,   \
        /*fail_fast=*/true);                                              \
  }

  CLIENT_METHOD(CreateContext);
  CLIENT_METHOD(UpdateContext);
  CLIENT_METHOD(Enqueue);
  CLIENT_METHOD(WaitQueueDone);
  CLIENT_METHOD(KeepAlive);

#undef CLIENT_METHOD

  void CloseContextAsync(const CloseContextRequest* request,
                         CloseContextResponse* response,
                         StatusCallback done) override {
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.eager.EagerService/CloseContext", *request,
        response, std::move(done), nullptr, nullptr);

    VLOG(1) << "Sending RPC to close remote eager context "
            << request->DebugString();

    mutex_lock l(mu_);
    const auto& it = enqueue_dispatchers_.find(request->context_id());
    if (it != enqueue_dispatchers_.end()) {
      it->second.CancelCall();
      enqueue_dispatchers_.erase(it);
    } else if (EnableStreaming()) {
      LOG(ERROR) << "Remote EagerContext with id " << request->context_id()
                 << " does not seem to exist.";
    }
  }

  void StreamingEnqueueAsync(const EnqueueRequest* request,
                             EnqueueResponse* response,
                             StatusCallback done) override {
    if (EnableStreaming()) {
      tf_shared_lock l(mu_);
      auto it = enqueue_dispatchers_.find(request->context_id());
      if (it == enqueue_dispatchers_.end()) {
        auto it_and_bool = enqueue_dispatchers_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(request->context_id()),
            std::forward_as_tuple(
                &stub_, cq_,
                "/tensorflow.eager.EagerService/StreamingEnqueue"));
        it = it_and_bool.first;
      }
      it->second.SendNextRequest(*request, response, std::move(done));
    } else {
      Notification n;
      Status status;
      EnqueueAsync(request, response, [&n, &status](const Status& s) {
        status.Update(s);
        n.Notify();
      });
      n.WaitForNotification();
      done(status);
    }
  }

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;

  mutable mutex mu_;

  std::unordered_map<uint64, StreamingRPCDispatcher<EnqueueResponse>>
      enqueue_dispatchers_ GUARDED_BY(mu_);
};

class GrpcEagerClientCache : public EagerClientCache {
 public:
  explicit GrpcEagerClientCache(
      std::shared_ptr<tensorflow::GrpcChannelCache> cache)
      : next_round_robin_assignment_(0), cache_(cache), threads_(4) {}

  ~GrpcEagerClientCache() override { threads_.clear(); }

  Status GetClient(const string& target, EagerClient** client) override {
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      tensorflow::SharedGrpcChannelPtr shared =
          cache_->FindWorkerChannel(target);
      if (shared == nullptr) {
        return errors::InvalidArgument("Client for target ", target,
                                       " not found.");
      }
      auto worker = std::unique_ptr<EagerClient>(new GrpcEagerClient(
          shared, threads_[AssignClientToThread(target)].completion_queue()));

      it = clients_.emplace(target, std::move(worker)).first;
    }

    *client = it->second.get();
    return Status::OK();
  }

 private:
  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ GUARDED_BY(assignment_mu_);

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

  class GrpcEagerClientThread {
   public:
    GrpcEagerClientThread() {
      thread_.reset(Env::Default()->StartThread(
          ThreadOptions(), "eager_client_thread", [this]() {
            void* tag;
            bool ok;
            while (completion_queue_.Next(&tag, &ok)) {
              VLOG(4) << "GrpcEagerClientThread got next tag";
              GrpcClientCQTag* callback_tag =
                  static_cast<GrpcClientCQTag*>(tag);
              callback_tag->OnCompleted(ok);
              VLOG(4) << "GrpcEagerClientThread blocking for next tag";
            }
            VLOG(4) << "GrpcEagerClientThread exiting";
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
