/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class GrpcWorkerCache : public WorkerCachePartial {
 public:
  explicit GrpcWorkerCache(GrpcChannelCache* channel_cache)
      : channel_cache_(channel_cache) {
    // TODO(mrry): Investigate possible performance improvements by
    // replacing this thread with a threadpool.
    polling_thread_ = Env::Default()->StartThread(
        ThreadOptions(), "grpc_worker_cache", [this]() {
          void* tag;
          bool ok;
          while (completion_queue_.Next(&tag, &ok)) {
            GrpcClientCQTag* callback_tag = static_cast<GrpcClientCQTag*>(tag);
            callback_tag->OnCompleted(ok);
            delete callback_tag;
          }
        });
  }

  // Explicit destructor to control destruction order.
  ~GrpcWorkerCache() override {
    completion_queue_.Shutdown();
    delete polling_thread_;  // Blocks until thread exits.
    delete channel_cache_;
  }

  void ListWorkers(std::vector<string>* workers) override {
    channel_cache_->ListWorkers(workers);
  }

  WorkerInterface* CreateWorker(const string& target) override {
    SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
    if (!channel) return nullptr;
    WorkerInterface* ret =
        NewGrpcRemoteWorker(channel, &completion_queue_, &logger_);
    return ret;
  }

  void SetLogging(bool v) override { logger_.SetLogging(v); }

  void ClearLogs() override { logger_.ClearLogs(); }

  bool RetrieveLogs(int64 step_id, StepStats* ss) override {
    return logger_.RetrieveLogs(step_id, ss);
  }

 private:
  GrpcChannelCache* channel_cache_;  // Owned.
  ::grpc::CompletionQueue completion_queue_;
  Thread* polling_thread_;  // Owned.
  WorkerCacheLogger logger_;
};

WorkerCacheInterface* NewGrpcWorkerCache(GrpcChannelCache* cc) {
  return new GrpcWorkerCache(cc);
}

}  // namespace tensorflow
