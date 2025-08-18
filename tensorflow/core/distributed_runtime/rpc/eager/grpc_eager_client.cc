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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "grpcpp/generic/generic_stub.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/error_payloads.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/core_platform_payloads.pb.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace eager {
namespace {

/* Retrieve the global env variable.
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
 * Note that the caller could still disable streaming enqueue, even though
 * EnableStreaminh() returns true, if the caller's executor is set to disable
 * streaming enqueue when the executor was created. EnableStreaming() is
 * determined based on the global env variable, which by default is turned on
 * for the main executor.
 *
 * TODO(fishx): When exiting client, make sure all pending ops on remote workers
 * are finished.
 */
bool EnableStreaming() {
  bool result;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE",
                                 true, &result));
  return result;
}

// Ref-counted thread to handle callbacks for completed requests a GRPC
// completion queue. The thread might be shared by multiple eager clients, and
// each one of them should hold a reference count to ensure that the thread
// outlives the clients.
// To ensure that every tag in completion queue is processed, this thread also
// holds a reference to itself and always wait until ref count is one to exit.
class GrpcEagerClientThread : public core::RefCounted {
 public:
  GrpcEagerClientThread() {
    // Hold a reference to ensure every completion tag gets processed.
    Ref();
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), "eager_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            VLOG(4) << "GrpcEagerClientThread got next tag";
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
            VLOG(4) << "GrpcEagerClientThread blocking for next tag";
            if (RefCountIsOne()) {
              break;
            }
          }
          VLOG(4) << "GrpcEagerClientThread exiting";
          completion_queue_.Shutdown();
          // `this` holds the final reference so cannot directly Unref here.
          // Instead, schedule a separate thread to clean it up.
          Env::Default()->SchedClosure([this]() { this->Unref(); });
        }));
  }

  ~GrpcEagerClientThread() override {}

  ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};

class GrpcEagerClient : public EagerClient {
 public:
  GrpcEagerClient(const tensorflow::SharedGrpcChannelPtr& channel,
                  GrpcEagerClientThread* thread, const string& target)
      : stub_(channel), thread_(thread), target_(target) {
    // Hold a reference to make sure the corresponding EagerClientThread
    // outlives the client.
    thread_->Ref();
    cq_ = thread->completion_queue();
  }
  ~GrpcEagerClient() override { thread_->Unref(); }

  bool allow_multiple_pending_requests() const override {
    return EnableStreaming();
  }

#define CLIENT_METHOD(method)                                             \
  void method##Async(const method##Request* request,                      \
                     method##Response* response, StatusCallback done)     \
      override {                                                          \
    StatusCallback done_wrapped = callback_wrapper(std::move(done));      \
    new RPCState<protobuf::Message>(                                      \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request, \
        response, std::move(done_wrapped), /*call_opts=*/nullptr,         \
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,    \
        &target_);                                                        \
  }

  CLIENT_METHOD(CreateContext);
  CLIENT_METHOD(UpdateContext);
  CLIENT_METHOD(WaitQueueDone);
  CLIENT_METHOD(KeepAlive);

#undef CLIENT_METHOD

#define CLIENT_METHOD_WITH_TIMEOUT_AND_RETRIES(method)                       \
  void method##Async(const method##Request* request,                         \
                     method##Response* response, StatusCallback done,        \
                     int64_t init_timeout_in_ms, int retries) override {     \
    CallOptions* call_ops = nullptr;                                         \
    StatusCallback done_wrapped;                                             \
    if (init_timeout_in_ms > 0) {                                            \
      call_ops = new CallOptions;                                            \
      call_ops->SetTimeout(init_timeout_in_ms);                              \
      auto new_done = [call_ops, done = std::move(done)](const Status& s) {  \
        done(s);                                                             \
        delete call_ops;                                                     \
      };                                                                     \
      done_wrapped = callback_wrapper(new_done);                             \
    } else {                                                                 \
      done_wrapped = callback_wrapper(std::move(done));                      \
    }                                                                        \
    new RPCState<protobuf::Message>(                                         \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request,    \
        response, std::move(done_wrapped), call_ops, /*threadpool=*/nullptr, \
        /*max_retries=*/retries, /*fail_fast=*/true, &target_);              \
  }

  CLIENT_METHOD_WITH_TIMEOUT_AND_RETRIES(CreateContext);

#undef CLIENT_METHOD_WITH_TIMEOUT_AND_RETRIES

#define CLIENT_CANCELABLE_METHOD(method)                                      \
  void method##Async(CallOptions* call_opts, const method##Request* request,  \
                     method##Response* response, StatusCallback done)         \
      override {                                                              \
    StatusCallback done_wrapped = callback_wrapper(std::move(done));          \
    new RPCState<protobuf::Message>(                                          \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request,     \
        response, std::move(done_wrapped), call_opts, /*threadpool=*/nullptr, \
        /*max_retries=*/0, /*fail_fast=*/true, &target_);                     \
  }

  CLIENT_CANCELABLE_METHOD(Enqueue);
  CLIENT_CANCELABLE_METHOD(RunComponentFunction);

#undef CLIENT_CANCELABLE_METHOD

  void CloseContextAsync(const CloseContextRequest* request,
                         CloseContextResponse* response,
                         StatusCallback done) override {
    StatusCallback done_wrapped = callback_wrapper(std::move(done));
    new RPCState<protobuf::Message>(
        &stub_, cq_, "/tensorflow.eager.EagerService/CloseContext", *request,
        response, std::move(done_wrapped), /*call_opts=*/nullptr,
        /*threadpool=*/nullptr, /*max_retries=*/0, /*fail_fast=*/true,
        &target_);

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

  void StreamingEnqueueAsync(bool enable_streaming_enqueue,
                             CallOptions* call_opts,
                             const EnqueueRequest* request,
                             EnqueueResponse* response,
                             StatusCallback done) override {
    StatusCallback done_wrapped = callback_wrapper(std::move(done));
    // Whether streaming enqueue is used is determined based on 2 factors:
    // 1. The global env variable, as checked in EnableStreaming().
    // 2. The flag set in the eager executor.
    // Streaming enqueue is allowed only when the both are enabled.
    if (EnableStreaming() && enable_streaming_enqueue) {
      mutex_lock l(mu_);
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
      // TODO(haoyuzhang): Consider supporting cancellation for streaming RPC?
      it->second.SendNextRequest(*request, response, std::move(done_wrapped));
    } else {
      absl::Notification n;
      absl::Status status;
      EnqueueAsync(call_opts, request, response,
                   [&n, &status](const absl::Status& s) {
                     status.Update(s);
                     n.Notify();
                   });
      n.WaitForNotification();
      done_wrapped(status);
    }
  }

 private:
  ::grpc::GenericStub stub_;
  const GrpcEagerClientThread* thread_;
  const string target_;

  ::grpc::CompletionQueue* cq_;

  mutable mutex mu_;

  std::unordered_map<uint64, StreamingRPCDispatcher<EnqueueResponse>>
      enqueue_dispatchers_ TF_GUARDED_BY(mu_);

  StatusCallback callback_wrapper(StatusCallback done) {
    Ref();
    return [this, done = std::move(done)](const absl::Status& status) {
      done(status);
      this->Unref();
      if (TF_PREDICT_FALSE(!status.ok())) {
        // Retrieve the location where the error was produced.
        auto error_source_payload = status.GetPayload(kErrorSource);

        if (error_source_payload.has_value()) {
          tensorflow::core::platform::ErrorSourceProto error_source_proto;
          error_source_proto.ParseFromString(
              std::string(*error_source_payload));  // NOLINT
          metrics::UpdateEagerClientErrorCounter(
              error_source_proto.ErrorSource_Name(
                  error_source_proto.error_source()),
              absl::StatusCodeToString(status.code()));
        } else {
          metrics::UpdateEagerClientErrorCounter(
              "unknown", absl::StatusCodeToString(status.code()));
        }
      }
    };
  }
};

class GrpcEagerClientCache : public EagerClientCache {
 public:
  explicit GrpcEagerClientCache(
      std::shared_ptr<tensorflow::GrpcChannelCache> cache)
      : next_round_robin_assignment_(0), cache_(cache), threads_(4) {
    for (int i = 0, end = threads_.size(); i < end; i++) {
      threads_[i].reset(new GrpcEagerClientThread());
    }
  }

  ~GrpcEagerClientCache() override { threads_.clear(); }

  absl::Status GetClient(const string& target,
                         core::RefCountPtr<EagerClient>* client) override {
    mutex_lock l(clients_mu_);
    auto it = clients_.find(target);
    if (it == clients_.end()) {
      tensorflow::SharedGrpcChannelPtr shared =
          cache_->FindWorkerChannel(target);
      if (shared == nullptr) {
        return errors::InvalidArgument("Client for target ", target,
                                       " not found.");
      }
      int assigned_index = AssignClientToThread(target);
      GrpcEagerClientThread* thread = threads_[assigned_index].get();
      core::RefCountPtr<EagerClient> worker(
          new GrpcEagerClient(shared, thread, target));
      it = clients_.emplace(target, std::move(worker)).first;
    }

    it->second->Ref();
    client->reset(it->second.get());
    return absl::OkStatus();
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

  std::shared_ptr<tensorflow::GrpcChannelCache> cache_;
  mutable mutex clients_mu_;
  std::unordered_map<string, core::RefCountPtr<EagerClient>> clients_
      TF_GUARDED_BY(clients_mu_);
  std::vector<core::RefCountPtr<GrpcEagerClientThread>> threads_;
};

}  // namespace

EagerClientCache* NewGrpcEagerClientCache(
    std::shared_ptr<tensorflow::GrpcChannelCache> channel) {
  return new GrpcEagerClientCache(channel);
}

}  // namespace eager
}  // namespace tensorflow
