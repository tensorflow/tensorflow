/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_H_

#include <unordered_map>
#include "tensorflow/core/distributed_runtime/recent_request_ids.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/worker.h"

namespace grpc {
class ByteBuffer;
class ServerBuilder;
}  // namespace grpc

namespace tensorflow {

class AsyncServiceInterface;
class ConfigProto;
struct WorkerEnv;
struct WorkerSession;

class GrpcWorker : public Worker {
 public:
  GrpcWorker(WorkerEnv* env, const ConfigProto& config);

  // Specialized version of RecvTensor for gRPC, which avoids a copy.
  virtual void GrpcRecvTensorAsync(CallOptions* opts,
                                   const RecvTensorRequest* request,
                                   ::grpc::ByteBuffer* response,
                                   StatusCallback done);

  virtual void LoggingAsync(const LoggingRequest* request,
                            LoggingResponse* response, StatusCallback done);

  virtual void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                            RecvBufResponse* response, StatusCallback done);

  WorkerEnv* env();

 private:
  RecentRequestIds recent_request_ids_;
  const int32 recv_buf_max_chunk_;
};

std::unique_ptr<GrpcWorker> NewGrpcWorker(WorkerEnv* worker_env,
                                          const ConfigProto& config);

struct GrpcWorkerServiceOptions {
  // Map from GrpcWorkerMethod id to queue depth.  If set this overrides the
  // default queue depth for a method.
  std::unordered_map<int, int> queue_depth;
  int num_worker_threads = 8;
};

// Returns an implementation of WorkerService rpc service.
std::unique_ptr<AsyncServiceInterface> NewGrpcWorkerService(
    GrpcWorker* worker, ::grpc::ServerBuilder* builder,
    GrpcWorkerServiceOptions opts = GrpcWorkerServiceOptions());

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_H_
