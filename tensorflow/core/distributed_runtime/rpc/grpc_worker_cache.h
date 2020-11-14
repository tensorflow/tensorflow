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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_CACHE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_CACHE_H_

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {

class GrpcWorkerEnv {
 public:
  GrpcWorkerEnv(size_t num_completion_queues, size_t num_threads);

  ~GrpcWorkerEnv();

  thread::ThreadPool* GetThreadPool() const { return threadpool_.get(); }

  size_t CompletionQueueSize() const { return threads_.size(); }

  ::grpc::CompletionQueue* GetCompletionQueue(size_t index) const {
    return threads_.at(index).completion_queue();
  }

 private:
  // Thread wrapping class that drives work over a single gRPC
  // CompletionQueue.
  class GrpcWorkerCacheThread {
   public:
    GrpcWorkerCacheThread();

    ~GrpcWorkerCacheThread();

    ::grpc::CompletionQueue* completion_queue() const {
      return &completion_queue_;
    }

   private:
    mutable ::grpc::CompletionQueue completion_queue_;
    std::unique_ptr<Thread> thread_;
  };

  std::unique_ptr<thread::ThreadPool> threadpool_;
  std::vector<GrpcWorkerCacheThread> threads_;
};

// Create a GrpcWorkerEnv instance that can be used as argument to create
// gRPC worker cache. Caller should take the ownership of the returned instance.
GrpcWorkerEnv* CreateGrpcWorkerEnv();

// The returned WorkerCacheInterface object takes the ownership of "cc".
WorkerCacheInterface* NewGrpcWorkerCache(std::shared_ptr<GrpcChannelCache> cc,
                                         GrpcWorkerEnv* worker_env);

WorkerCacheInterface* NewGrpcWorkerCacheWithLocalWorker(
    std::shared_ptr<GrpcChannelCache> cc, GrpcWorkerEnv* worker_env,
    WorkerInterface* local_worker, const string& local_target);

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_WORKER_CACHE_H_
