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

#include "tensorflow/compiler/xrt/client/xrt_grpc_eager_client.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_state.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"

namespace tensorflow {

XrtGrpcEagerClient::XrtGrpcEagerClient(const SharedGrpcChannelPtr& channel,
                                       ::grpc::CompletionQueue* cq)
    : stub_(channel), cq_(cq) {}

#define EAGER_CLIENT_METHOD(method)                                       \
  void XrtGrpcEagerClient::method##Async(                                 \
      const eager::method##Request* request,                              \
      eager::method##Response* response, StatusCallback done,             \
      CallOptions* call_opts) {                                           \
    new RPCState<protobuf::Message>(                                      \
        &stub_, cq_, "/tensorflow.eager.EagerService/" #method, *request, \
        response, std::move(done), call_opts, nullptr);                   \
  }

EAGER_CLIENT_METHOD(CreateContext);
EAGER_CLIENT_METHOD(Enqueue);
EAGER_CLIENT_METHOD(WaitQueueDone);
EAGER_CLIENT_METHOD(KeepAlive);
EAGER_CLIENT_METHOD(CloseContext);
EAGER_CLIENT_METHOD(RegisterFunction);
EAGER_CLIENT_METHOD(SendTensor);
#undef EAGER_CLIENT_METHOD

#define WORKER_CLIENT_METHOD(method)                                           \
  void XrtGrpcEagerClient::method##Async(                                      \
      const method##Request* request, method##Response* response,              \
      StatusCallback done, CallOptions* call_opts) {                           \
    new RPCState<protobuf::Message>(                                           \
        &stub_, cq_, "/tensorflow.WorkerService/" #method, *request, response, \
        std::move(done), call_opts, nullptr);                                  \
  }

WORKER_CLIENT_METHOD(GetStatus);
WORKER_CLIENT_METHOD(RecvTensor);
#undef WORKER_CLIENT_METHOD

class XrtGrpcEagerClientThread {
 public:
  XrtGrpcEagerClientThread() {
    thread_.reset(Env::Default()->StartThread(
        ThreadOptions(), "xrt_eager_client_thread", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
          }
        }));
  }

  ~XrtGrpcEagerClientThread() {
    completion_queue_.Shutdown();
    thread_.reset();
  }

  ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

 private:
  ::grpc::CompletionQueue completion_queue_;
  std::unique_ptr<Thread> thread_;
};  // XrtGrpcEagerClientThread

XrtGrpcEagerClientCache::XrtGrpcEagerClientCache(
    std::shared_ptr<tensorflow::GrpcChannelCache> channel_cache)
    : next_round_robin_assignment_(0), cache_(channel_cache), threads_(4) {}

XrtGrpcEagerClientCache::~XrtGrpcEagerClientCache() { threads_.clear(); }

xla::StatusOr<XrtGrpcEagerClient*> XrtGrpcEagerClientCache::GetClient(
    const string& target) {
  auto it = clients_.find(target);
  if (it == clients_.end()) {
    tensorflow::SharedGrpcChannelPtr shared = cache_->FindWorkerChannel(target);
    if (!shared) {
      return errors::NotFound("Unknown target ", target);
    }
    auto worker = absl::make_unique<XrtGrpcEagerClient>(
        shared, threads_[AssignClientToThread(target)].completion_queue());

    it = clients_.emplace(target, std::move(worker)).first;
  }

  return it->second.get();
}

size_t XrtGrpcEagerClientCache::AssignClientToThread(const string& target) {
  // Round-robin target assignment, but keeps the same target on the same
  // polling thread always, as this is important for gRPC performance.
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

xla::StatusOr<std::shared_ptr<GrpcChannelCache>> GetGrpcChannelCache(
    const ClusterDef& cluster_def, ChannelCreationFunction channel_func) {
  GrpcChannelSpec channel_spec;
  for (const JobDef& job : cluster_def.job()) {
    std::map<int, string> host_ports(job.tasks().begin(), job.tasks().end());
    TF_RETURN_IF_ERROR(channel_spec.AddHostPortsJob(job.name(), host_ports));
  }
  return std::shared_ptr<GrpcChannelCache>(
      NewGrpcChannelCache(channel_spec, channel_func));
}

}  // namespace tensorflow
