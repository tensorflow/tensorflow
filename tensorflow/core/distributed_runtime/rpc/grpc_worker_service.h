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

#include <memory>
#include <unordered_map>

#include "grpcpp/server_builder.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service_impl.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_response_cache.h"
#include "tensorflow/core/distributed_runtime/worker.h"
#include "tensorflow/core/protobuf/worker.pb.h"
#include "tsl/distributed_runtime/rpc/async_service_interface.h"

namespace grpc {
class ByteBuffer;
}  // namespace grpc

namespace tsl {
class AsyncServiceInterface;
}

namespace tensorflow {

class ConfigProto;
struct WorkerEnv;
class WorkerSession;
class RpcResponseCache;

class GrpcWorker : public Worker {
 public:
  GrpcWorker(WorkerEnv* env, const ConfigProto& config);

  // Specialized version of RecvTensor for gRPC, which avoids a copy.
  virtual void GrpcRecvTensorAsync(CallOptions* opts,
                                   const RecvTensorRequest* request,
                                   ::grpc::ByteBuffer* response,
                                   StatusCallback done);

  void LoggingAsync(const LoggingRequest* request, LoggingResponse* response,
                    StatusCallback done) override;

  void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                    RecvBufResponse* response, StatusCallback done) override;

  void CleanupGraphAsync(const CleanupGraphRequest* request,
                         CleanupGraphResponse* response,
                         StatusCallback done) override;

  WorkerEnv* env();

  void EnableResponseCache();

  void RemoveCacheEntryForId(int64_t request_id);

 private:
  std::unique_ptr<RpcResponseCache> response_cache_;
  const int32 recv_buf_max_chunk_;
};

std::unique_ptr<GrpcWorker> NewGrpcWorker(WorkerEnv* worker_env,
                                          const ConfigProto& config);

struct GrpcWorkerServiceOptions {
  // Map from GrpcWorkerMethod id to queue depth.  If set this overrides the
  // default queue depth for a method.
  std::unordered_map<int, int> queue_depth;
  int num_serving_threads = 8;
};

// Returns an implementation of WorkerService rpc service.
std::unique_ptr<tsl::AsyncServiceInterface> NewGrpcWorkerService(
    GrpcWorker* worker, ::grpc::ServerBuilder* builder,
    GrpcWorkerServiceOptions options = GrpcWorkerServiceOptions());

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_SERVICE_H_
