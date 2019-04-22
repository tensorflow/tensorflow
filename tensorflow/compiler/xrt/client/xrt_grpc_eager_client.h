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

// Self-contained client for communicating with a TensorFlow Eager remote
// service over gRPC.
//
// Unlike, say, the TensorFlow C API, this class is intended to be a
// self-contained, minimal-dependency way to interact with a remote TF eager
// server, containing just enough functionality for the XRT use case.

#ifndef TENSORFLOW_COMPILER_XRT_CLIENT_XRT_GRPC_EAGER_CLIENT_H_
#define TENSORFLOW_COMPILER_XRT_CLIENT_XRT_GRPC_EAGER_CLIENT_H_

#include "grpcpp/generic/generic_stub.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/protobuf/eager_service.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

// This class is a self-contained cousin of the standard EagerClient class.
// Unlike EagerClient, this class includes all of the methods needed by XRT,
// including methods from both EagerService and WorkerService.
// This reduces the dependency footprint, since in particular RecvTensor's
// implementation depends on a bunch of TF framework infrastructure (e.g.,
// Device, Tensor) that we don't need for the XRT client use case.
class XrtGrpcEagerClient {
 public:
  XrtGrpcEagerClient(const SharedGrpcChannelPtr& channel,
                     ::grpc::CompletionQueue* cq);
  ~XrtGrpcEagerClient() = default;

  XrtGrpcEagerClient(const XrtGrpcEagerClient&) = delete;
  XrtGrpcEagerClient(XrtGrpcEagerClient&&) = delete;
  XrtGrpcEagerClient& operator=(const XrtGrpcEagerClient&) = delete;
  XrtGrpcEagerClient& operator=(XrtGrpcEagerClient&&) = delete;

  void CreateContextAsync(const eager::CreateContextRequest* request,
                          eager::CreateContextResponse* response,
                          StatusCallback done,
                          CallOptions* call_opts = nullptr);
  void EnqueueAsync(const eager::EnqueueRequest* request,
                    eager::EnqueueResponse* response, StatusCallback done,
                    CallOptions* call_opts = nullptr);
  void WaitQueueDoneAsync(const eager::WaitQueueDoneRequest* request,
                          eager::WaitQueueDoneResponse* response,
                          StatusCallback done,
                          CallOptions* call_opts = nullptr);
  void KeepAliveAsync(const eager::KeepAliveRequest* request,
                      eager::KeepAliveResponse* response, StatusCallback done,
                      CallOptions* call_opts = nullptr);
  void CloseContextAsync(const eager::CloseContextRequest* request,
                         eager::CloseContextResponse* response,
                         StatusCallback done, CallOptions* call_opts = nullptr);
  void RegisterFunctionAsync(const eager::RegisterFunctionRequest* request,
                             eager::RegisterFunctionResponse* response,
                             StatusCallback done,
                             CallOptions* call_opts = nullptr);
  void SendTensorAsync(const eager::SendTensorRequest* request,
                       eager::SendTensorResponse* response, StatusCallback done,
                       CallOptions* call_opts = nullptr);

  // The following two methods are actually from the WorkerService API, not
  // EagerService, but are necessary for using remote Eager, and we include them
  // here for self-containedness.

  // We use RecvTensor to copy tensors back from a remote worker to the client.
  void RecvTensorAsync(const RecvTensorRequest* request,
                       RecvTensorResponse* response, StatusCallback done,
                       CallOptions* call_opts = nullptr);

  // We use GetStatus to discover device incarnation values for use in
  // RecvTensor.
  // TODO(phawkins): We need to call GetStatus to work around a bug in the
  // TFE server implementation. Remove this API call and use the device
  // information from CreateContext once the bug fix is deployed everywhere.
  void GetStatusAsync(const GetStatusRequest* request,
                      GetStatusResponse* response, StatusCallback done,
                      CallOptions* call_opts = nullptr);

  // Helper method for calling any of the ...Async methods synchronously.
  template <typename Request, typename Response, typename Method>
  Status SyncCall(Method m, const Request* request, Response* response,
                  CallOptions* call_opts = nullptr) {
    absl::Notification done;
    Status status;
    (this->*(m))(
        request, response,
        [&](Status s) {
          status = s;
          done.Notify();
        },
        call_opts);
    done.WaitForNotification();
    return status;
  }

 private:
  ::grpc::GenericStub stub_;
  ::grpc::CompletionQueue* cq_;
};

class XrtGrpcEagerClientThread;

// Simple wrapper class that can be used to retrieve XrtGrpcEagerClients.
class XrtGrpcEagerClientCache {
 public:
  explicit XrtGrpcEagerClientCache(
      std::shared_ptr<tensorflow::GrpcChannelCache> channel_cache);
  ~XrtGrpcEagerClientCache();

  XrtGrpcEagerClientCache(const XrtGrpcEagerClientCache&) = delete;
  XrtGrpcEagerClientCache(XrtGrpcEagerClientCache&&) = delete;
  XrtGrpcEagerClientCache& operator=(const XrtGrpcEagerClientCache&) = delete;
  XrtGrpcEagerClientCache& operator=(XrtGrpcEagerClientCache&&) = delete;

  // Returns a cached client for 'target'. 'target' should be a task name known
  // te the channel cache, e.g., "/job:worker/task:0/replica:0".
  xla::StatusOr<XrtGrpcEagerClient*> GetClient(const string& target);

 private:
  size_t AssignClientToThread(const string& target);

  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ GUARDED_BY(assignment_mu_);

  std::shared_ptr<tensorflow::GrpcChannelCache> cache_;
  std::unordered_map<string, std::unique_ptr<XrtGrpcEagerClient>> clients_;
  std::vector<XrtGrpcEagerClientThread> threads_;
};

// Builds a GrpcChannelCache for a TF cluster `cluster_def`. `channel_func`
// is a function to use to create channels; it is client-provided so clients can
// set up custom authentication, etc.
xla::StatusOr<std::shared_ptr<GrpcChannelCache>> GetGrpcChannelCache(
    const ClusterDef& cluster_def, ChannelCreationFunction channel_func);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_CLIENT_XRT_GRPC_EAGER_CLIENT_H_
