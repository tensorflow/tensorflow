/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_SERVICE_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_SERVICE_H_

#include <atomic>
#include <memory>

#include "grpcpp/server_builder.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_common.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_grpc.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"

namespace tensorflow {
// gRPC service for handling CompilationCache requests.
// To avoid OOMs during execution, this service using the asynchronous raw gRPC
// interface to serialize cache results directly to gRPC byte buffers. This
// allows us to control serialization concurrency and avoids making an extra
// copy of the program cache for each worker.
class TpuCompilationCacheService {
 public:
  using ServiceType = ::tensorflow::tpu::grpc::TpuCompilationCacheService;
  using AsyncService = ServiceType::AsyncService;

  TpuCompilationCacheService(::grpc::ServerBuilder* server_builder,
                             tpu::TpuCompilationCacheInterface* cache);
  ~TpuCompilationCacheService();

  void Start();
  bool Shutdown(int timeout_sec);
  void SetMemoryQuota(size_t max_bytes);

 private:
  void HandleRPCsLoop();

  using GetTpuProgramCall =
      tsl::Call<TpuCompilationCacheService, AsyncService,
                tpu::GetTpuProgramRequest, ::grpc::ByteBuffer>;

  // Schedule the cache fetch into the serving thread pool.
  void HandleGetTpuProgram(GetTpuProgramCall* call);

  // Performs the actual cache fetch and serialization.
  void GetTpuProgram(GetTpuProgramCall* call);

  std::atomic<bool> running_;
  tpu::TpuCompilationCacheInterface* cache_;
  ::grpc::ServerBuilder* server_builder_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<::grpc::ServerCompletionQueue> cq_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::unique_ptr<Thread> polling_thread_;
  AsyncService service_;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILATION_CACHE_SERVICE_H_
