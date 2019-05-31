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

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"

#include <unordered_map>

#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_client_cq_tag.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_remote_worker.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {

class GrpcWorkerCache : public WorkerCachePartial {
 public:
  // TODO(ncteisen): consider adding a config var or flag for this
  static constexpr const size_t kGrpcWorkerCacheThreadCount = 8;

  explicit GrpcWorkerCache(std::shared_ptr<GrpcChannelCache> channel_cache,
                           WorkerInterface* local_worker,
                           const string& local_target)
      : local_target_(local_target),
        local_worker_(local_worker),
        channel_cache_(channel_cache),
        threads_(kGrpcWorkerCacheThreadCount),
        next_round_robin_assignment_(0) {
    // NOTE: We don't yet have any reason to assign NUMA affinity to this
    // ThreadPool.  If there's only a single NIC it shouldn't make any
    // difference since presumably it is handling memory from all nodes.
    ThreadOptions options;
    options.numa_node = port::kNUMANoAffinity;
    const int kNumCallbackThreads = 10;
    callback_threadpool_.reset(new thread::ThreadPool(
        Env::Default(), options, "grpc_wcache_callback", kNumCallbackThreads,
        false /*low_latency_hint*/, nullptr /*allocator*/));
  }

  // Explicit destructor to control destruction order.
  ~GrpcWorkerCache() override {
    threads_.clear();  // Blocks until threads exit.
  }

  void ListWorkers(std::vector<string>* workers) const override {
    channel_cache_->ListWorkers(workers);
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
    channel_cache_->ListWorkersInJob(job_name, workers);
  }

  WorkerInterface* CreateWorker(const string& target) override {
    if (target == local_target_) {
      return local_worker_;
    } else {
      SharedGrpcChannelPtr channel = channel_cache_->FindWorkerChannel(target);
      if (!channel) return nullptr;
      return NewGrpcRemoteWorker(
          channel, threads_[AssignWorkerToThread(target)].completion_queue(),
          callback_threadpool_.get(), &logger_);
    }
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) override {
    if (target == local_target_) {
      CHECK_EQ(worker, local_worker_)
          << "Releasing a worker that was not returned by this WorkerCache";
    } else {
      WorkerCacheInterface::ReleaseWorker(target, worker);
    }
  }

  void SetLogging(bool v) override { logger_.SetLogging(v); }

  void ClearLogs() override { logger_.ClearLogs(); }

  bool RetrieveLogs(int64 step_id, StepStats* ss) override {
    return logger_.RetrieveLogs(step_id, ss);
  }

 private:
  // Thread wrapping class that drives work over a single gRPC
  // CompletionQueue.
  class GrpcWorkerCacheThread {
   public:
    GrpcWorkerCacheThread() {
      thread_.reset(Env::Default()->StartThread(
          ThreadOptions(), "grpc_worker_cache", [this]() {
            void* tag;
            bool ok;
            while (completion_queue_.Next(&tag, &ok)) {
              GrpcClientCQTag* callback_tag =
                  static_cast<GrpcClientCQTag*>(tag);
              callback_tag->OnCompleted(ok);
            }
          }));
    }

    ~GrpcWorkerCacheThread() {
      completion_queue_.Shutdown();
      thread_.reset();
    }

    ::grpc::CompletionQueue* completion_queue() { return &completion_queue_; }

   private:
    ::grpc::CompletionQueue completion_queue_;
    std::unique_ptr<Thread> thread_;
  };  // GrpcWorkerCacheThread

  size_t AssignWorkerToThread(const string& target) {
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

  const string local_target_;
  WorkerInterface* const local_worker_;  // Not owned.
  std::shared_ptr<GrpcChannelCache> channel_cache_;
  WorkerCacheLogger logger_;
  std::vector<GrpcWorkerCacheThread> threads_;

  std::unique_ptr<thread::ThreadPool> callback_threadpool_;

  mutex assignment_mu_;
  std::unordered_map<std::string, size_t> target_assignments_
      GUARDED_BY(assignment_mu_);
  size_t next_round_robin_assignment_ GUARDED_BY(assignment_mu_);
};

}  // namespace

WorkerCacheInterface* NewGrpcWorkerCache(std::shared_ptr<GrpcChannelCache> cc) {
  return new GrpcWorkerCache(cc, nullptr, "");
}

WorkerCacheInterface* NewGrpcWorkerCacheWithLocalWorker(
    std::shared_ptr<GrpcChannelCache> cc, WorkerInterface* local_worker,
    const string& local_target) {
  return new GrpcWorkerCache(cc, local_worker, local_target);
}

}  // namespace tensorflow
