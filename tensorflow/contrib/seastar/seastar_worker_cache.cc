#include "tensorflow/contrib/seastar/seastar_channel_cache.h"
#include "tensorflow/contrib/seastar/seastar_remote_worker.h"
#include "tensorflow/contrib/seastar/seastar_worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/distributed_runtime/worker_cache_logger.h"

namespace tensorflow {
namespace {
class SeastarWorkerCache : public WorkerCachePartial {
public:
  explicit SeastarWorkerCache(SeastarChannelCache* channel_cache,
                              WorkerInterface* local_worker,
                              const string& local_target,
                              WorkerEnv* env)
    : local_target_(local_target),
      local_worker_(local_worker),
      channel_cache_(channel_cache),
      env_(env) {
  }

  virtual ~SeastarWorkerCache() {}

  void ListWorkers(std::vector<string>* workers) const override {
    channel_cache_->ListWorkers(workers);
  }

  void ListWorkersInJob(const string& job_name,
                        std::vector<string>* workers) const override {
    channel_cache_->ListWorkersInJob(job_name, workers);
  }

  WorkerInterface* GetOrCreateWorker(const string& target) override {
    if (target == local_target_) {
      return local_worker_;
    } else {
      seastar::channel* chan = channel_cache_->FindWorkerChannel(target);
      if (!chan) return nullptr;
      return NewSeastarRemoteWorker(chan, &logger_, env_);
    }    
  }

  void ReleaseWorker(const string& target, WorkerInterface* worker) {
    if (target == local_target_) {
      CHECK_EQ(worker, local_worker_)
        << "Releasing a worker that was not returned by this WorkerCache";
    } else {
      WorkerCacheInterface::ReleaseWorker(target, worker);
    }
  }

   Status GetEagerClientCache(
       std::unique_ptr<eager::EagerClientCache>* eager_client_cache) override {
     return errors::Unimplemented(
         "Eager client not yet implemented for this protocol");
   }

  void SetLogging(bool v) override { logger_.SetLogging(v); }
  void ClearLogs() override { logger_.ClearLogs(); }
  bool RetrieveLogs(int64 step_id, StepStats* ss) override {
    return logger_.RetrieveLogs(step_id, ss);
  }

private:
  const string local_target_;
  WorkerInterface* const local_worker_;
  WorkerCacheLogger logger_;
  SeastarChannelCache* channel_cache_;
  WorkerEnv* env_;
};
}

WorkerCacheInterface* NewSeastarWorkerCache(
    SeastarChannelCache* channel_cache, WorkerEnv* env) {
  return new SeastarWorkerCache(channel_cache, nullptr, "", env);
}

WorkerCacheInterface* NewSeastarWorkerCacheWithLocalWorker(
    SeastarChannelCache* channel_cache,
    WorkerInterface* local_worker,
    const string& local_target,
    WorkerEnv* env) {
  return new SeastarWorkerCache(channel_cache, local_worker, local_target, env);
}
}
