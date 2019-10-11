/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/run_handler.h"

#include <algorithm>
#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/run_handler_util.h"
#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace {
static constexpr int32 kMaxConcurrentHandlers = 128;

// TODO(azaks): Refactor with thread:ThreadPool
class RunHandlerEnvironment {
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;

 public:
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };

  RunHandlerEnvironment(Env* env, const ThreadOptions& thread_options,
                        const string& name)
      : env_(env), thread_options_(thread_options), name_(name) {}

  EnvThread* CreateThread(std::function<void()> f) {
    return env_->StartThread(thread_options_, name_, [=]() {
      // Set the processor flag to flush denormals to zero.
      port::ScopedFlushDenormal flush;
      // Set the processor rounding mode to ROUND TO NEAREST.
      port::ScopedSetRound round(FE_TONEAREST);
      if (thread_options_.numa_node != port::kNUMANoAffinity) {
        port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
      }
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (tracing::EventCollector::IsEnabled()) {
      id = tracing::GetUniqueArg();
      tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
    }
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f),
            Context(ContextKind::kThread),
            id,
        }),
    };
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure,
                                 t.f->trace_id);
    t.f->f();
  }
};

typedef typename RunHandlerEnvironment::Task Task;
typedef Eigen::RunQueue<Task, 1024> Queue;

class ThreadWorkSource {
 public:
  ThreadWorkSource()
      : non_blocking_work_sharding_factor_(
            static_cast<int32>(ParamFromEnvWithDefault(
                "TF_RUN_HANDLER_NUM_OF_NON_BLOCKING_QUEUES", 1))),
        non_blocking_work_queues_(non_blocking_work_sharding_factor_),
        blocking_inflight_(0),
        non_blocking_inflight_(0),
        traceme_id_(0) {
    queue_waiters_.next = &queue_waiters_;
    queue_waiters_.prev = &queue_waiters_;
    for (int i = 0; i < NonBlockingWorkShardingFactor(); ++i) {
      non_blocking_work_queues_.emplace_back(new NonBlockingQueue());
    }
  }

  ~ThreadWorkSource() {
    for (int i = 0; i < non_blocking_work_queues_.size(); ++i) {
      delete non_blocking_work_queues_[i];
    }
  }

  Task EnqueueTask(Task t, bool is_blocking) {
    mutex* mu = nullptr;
    Queue* task_queue = nullptr;
    thread_local int64 closure_counter = 0;

    if (!is_blocking) {
      int queue_index = ++closure_counter % non_blocking_work_sharding_factor_;
      task_queue = &(non_blocking_work_queues_[queue_index]->queue);
      mu = &non_blocking_work_queues_[queue_index]->queue_op_mu;
    } else {
      task_queue = &blocking_work_queue_;
      mu = &blocking_queue_op_mu_;
    }

    {
      mutex_lock l(*mu);
      // For a given queue, only one thread can call PushFront.
      t = task_queue->PushFront(std::move(t));
    }

    // Only wake up the thread that can take tasks from both blocking and
    // non-blocking queues. The rational is that we don't want to wake up more
    // threads than the available physical cores for them to compete for
    // resource. The non-blocking threads are used only to compensate for
    // threads that may be blocked on some tasks. There is less need to
    // proactively wake up those threads.
    static int max_rank_to_wakeup = static_cast<int>(
        ParamFromEnvWithDefault("TF_RUN_HANDLER_MAX_RANK_TO_WAKE_UP",
                                static_cast<int32>(ParamFromEnvWithDefault(
                                    "TF_RUN_HANDLER_MAX_CONCURRENT_HANDLERS",
                                    kMaxConcurrentHandlers))));
    if (max_rank_to_wakeup > 0 &&
        rank_.load(std::memory_order_relaxed) <= max_rank_to_wakeup) {
      Waiter* w = nullptr;
      {
        mutex_lock l(waiters_mu_);
        if (queue_waiters_.next != &queue_waiters_) {
          // Remove waiter from the LIFO queue
          w = queue_waiters_.next;

          CHECK(w->prev != w);
          CHECK(w->next != w);

          w->next->prev = w->prev;
          w->prev->next = w->next;

          // Use `w->next == &w` to indicate that the waiter has been removed
          // from the queue.
          w->next = w;
          w->prev = w;
        }
      }
      if (w != nullptr) {
        // We call notify_one() without any locks, so we can miss notifications.
        // The wake up logic is best effort and a thread will wake in short
        // period of time in case a notification is missed.
        w->cv.notify_one();
      }
    }
    VLOG(3) << "Added " << (is_blocking ? "inter" : "intra") << " work from "
            << traceme_id_.load(std::memory_order_relaxed);
    return t;
  }

  Task PopBlockingTask() { return blocking_work_queue_.PopBack(); }

  Task PopNonBlockingTask(int index) {
    return non_blocking_work_queues_[index]->queue.PopBack();
  }

  void WaitForWork(int max_sleep_micros) {
    thread_local Waiter waiter;
    {
      mutex_lock l(waiters_mu_);
      CHECK_EQ(waiter.next, &waiter);
      CHECK_EQ(waiter.prev, &waiter);

      // Add waiter to the LIFO queue
      waiter.prev = &queue_waiters_;
      waiter.next = queue_waiters_.next;
      waiter.next->prev = &waiter;
      waiter.prev->next = &waiter;
    }
    {
      mutex_lock l(waiter.mu);
      // Wait on the condition variable
      waiter.cv.wait_for(l, std::chrono::microseconds(max_sleep_micros));
    }

    mutex_lock l(waiters_mu_);
    // Remove waiter from the LIFO queue. Note even when a waiter wakes up due
    // to a notification we cannot conclude the waiter is not in the queue.
    // This is due to the fact that a thread preempted right before notifying
    // may resume after a waiter got re-added.
    if (waiter.next != &waiter) {
      CHECK(waiter.prev != &waiter);
      waiter.next->prev = waiter.prev;
      waiter.prev->next = waiter.next;
      waiter.next = &waiter;
      waiter.prev = &waiter;
    } else {
      CHECK_EQ(waiter.prev, &waiter);
    }
  }

  int TaskQueueSize(bool is_blocking) {
    if (is_blocking) {
      return blocking_work_queue_.Size();
    } else {
      unsigned total_size = 0;
      for (int i = 0; i < non_blocking_work_sharding_factor_; ++i) {
        total_size += non_blocking_work_queues_[i]->queue.Size();
      }
      return total_size;
    }
  }

  int64 GetTracemeId() { return traceme_id_.load(std::memory_order_relaxed); }

  void SetTracemeId(int64 value) { traceme_id_ = value; }
  void SetRank(int64 value) { rank_ = value; }

  int64 GetInflightTaskCount(bool is_blocking) {
    std::atomic<int64>* counter =
        is_blocking ? &blocking_inflight_ : &non_blocking_inflight_;
    return counter->load(std::memory_order_relaxed);
  }

  void IncrementInflightTaskCount(bool is_blocking) {
    std::atomic<int64>* counter =
        is_blocking ? &blocking_inflight_ : &non_blocking_inflight_;
    counter->fetch_add(1, std::memory_order_relaxed);
  }

  void DecrementInflightTaskCount(bool is_blocking) {
    std::atomic<int64>* counter =
        is_blocking ? &blocking_inflight_ : &non_blocking_inflight_;
    counter->fetch_sub(1, std::memory_order_relaxed);
  }

  unsigned NonBlockingWorkShardingFactor() {
    return non_blocking_work_sharding_factor_;
  }

  std::string ToString() {
    return strings::StrCat("traceme_id = ", GetTracemeId(),
                           ", inter queue size = ", TaskQueueSize(true),
                           ", inter inflight = ", GetInflightTaskCount(true),
                           ", intra queue size = ", TaskQueueSize(false),
                           ", intra inflight = ", GetInflightTaskCount(false));
  }

 private:
  // To reduce cache misses, we use a doubly-linked list of Waiter structs and
  // queue them in LIFO order rather than the FIFO order used by a single
  // condition variable.
  struct Waiter {
    Waiter() {
      next = this;
      prev = this;
    }
    condition_variable cv;
    mutex mu;
    Waiter* next;
    Waiter* prev;
  };

  struct NonBlockingQueue {
    mutex queue_op_mu;
    char pad[128];
    Queue queue;
  };

  int32 non_blocking_work_sharding_factor_;
  Eigen::MaxSizeVector<NonBlockingQueue*> non_blocking_work_queues_;

  std::atomic<int64> blocking_inflight_;
  std::atomic<int64> non_blocking_inflight_;

  Queue blocking_work_queue_;
  mutex blocking_queue_op_mu_;
  char pad_[128];
  mutex waiters_mu_;
  Waiter queue_waiters_ GUARDED_BY(waiters_mu_);
  std::atomic<int64> traceme_id_;
  std::atomic<int64> rank_;
};

class RunHandlerThreadPool {
 public:
  struct PerThread {
    constexpr PerThread() : pool(nullptr), thread_id(-1) {}
    RunHandlerThreadPool* pool;  // Parent pool, or null for normal threads.
    int thread_id;               // Worker thread index in pool.
  };

  RunHandlerThreadPool(int num_blocking_threads, int num_non_blocking_threads,
                       Env* env, const ThreadOptions& thread_options,
                       const string& name)
      : num_threads_(num_blocking_threads + num_non_blocking_threads),
        num_blocking_threads_(num_blocking_threads),
        num_non_blocking_threads_(num_non_blocking_threads),
        thread_data_(num_threads_),
        env_(env, thread_options, name),
        name_(name) {
    VLOG(1) << "Creating RunHandlerThreadPool " << name << " with  "
            << num_blocking_threads_ << " blocking threads and "
            << num_non_blocking_threads_ << " non-blocking threads.";
    cancelled_ = false;

    thread_data_.resize(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
      thread_data_[i].thread.reset(
          env_.CreateThread([this, i, num_blocking_threads]() {
            WorkerLoop(i, i < num_blocking_threads);
          }));
    }
  }

  ~RunHandlerThreadPool() {
    VLOG(1) << "Exiting RunHandlerThreadPool " << name_;

    cancelled_ = true;
    for (size_t i = 0; i < thread_data_.size(); ++i) {
      {
        mutex_lock l(thread_data_[i].mu);
        thread_data_[i].sources_not_empty.notify_all();
      }
      thread_data_[i].thread.reset();
    }
  }

  void AddWorkToQueue(ThreadWorkSource* tws, bool is_blocking,
                      std::function<void()> fn) {
    Task t = env_.CreateTask(std::move(fn));
    t = tws->EnqueueTask(std::move(t), is_blocking);
    if (t.f) {
      VLOG(3) << "Running " << (is_blocking ? "inter" : "intra") << " work for "
              << tws->GetTracemeId();
      env_.ExecuteTask(t);
    }
  }

  // Set work queues from which the thread 'tid' can steal its work.
  // The request with start_request_idx will be attempted first. Other requests
  // will be attempted in FIFO order based on their arrival time.

  // TODO(donglin) Change the task steal order to be round-robin such that if
  // an attempt to steal task from request i failed, then attempt to steal task
  // from the next request in terms of the arrival time. This approach may
  // provide better performance due to less lock retention. The drawback is that
  // the profiler will be a bit harder to read.
  void SetThreadWorkSources(
      int tid, int start_request_idx,
      const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources) {
    mutex_lock l(thread_data_[tid].mu);
    thread_data_[tid].thread_work_sources.resize(0);
    thread_data_[tid].thread_work_sources.emplace_back(
        thread_work_sources[start_request_idx]);
    // The number of shards for the queue. Threads in each shard will prioritize
    // different thread_work_sources. Increase the number of shards could
    // decrease the contention in the queue.
    // For example, when num_shards == 1:
    // thread_work_sources are ordered as start_request_idx, 0, 1, 2, 3, 4 ...
    // for all threads.
    // When num_shards == 2:
    // thread_work_sources are order as start_request_idx, 0, 2, 4 ... 1, 3,
    // 5... for half of the threads and start_request_idx, 1, 3, 5 ... 0, 2,
    // 4... for the other half of the threads.
    int num_shards = ParamFromEnvWithDefault("TF_RUN_HANDLER_QUEUE_SHARDS", 1);
    int token = tid % num_shards;
    for (int i = 0; i < num_shards; ++i) {
      for (int j = token; j < thread_work_sources.size(); j += num_shards) {
        if (j != start_request_idx) {
          thread_data_[tid].thread_work_sources.emplace_back(
              thread_work_sources[j]);
        }
      }
      token = (token + 1) % num_shards;
    }
    thread_data_[tid].sources_not_empty.notify_all();
  }

  PerThread* GetPerThread() {
    thread_local PerThread per_thread_;
    PerThread* pt = &per_thread_;
    return pt;
  }

  int CurrentThreadId() const {
    const PerThread* pt =
        const_cast<RunHandlerThreadPool*>(this)->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

  int NumThreads() const { return num_threads_; }

  int NumBlockingThreads() const { return num_blocking_threads_; }

  int NumNonBlockingThreads() const { return num_non_blocking_threads_; }

  void WorkerLoop(int thread_id, bool may_steal_blocking_work);

  void WaitForWork(bool is_blocking, int thread_id,
                   int32 max_blocking_inflight);

 private:
  struct ThreadData {
    ThreadData()
        : thread_work_sources(static_cast<int32>(
              ParamFromEnvWithDefault("TF_RUN_HANDLER_MAX_CONCURRENT_HANDLERS",
                                      kMaxConcurrentHandlers))) {}
    mutex mu;
    condition_variable sources_not_empty;
    std::unique_ptr<Thread> thread;
    Eigen::MaxSizeVector<ThreadWorkSource*> thread_work_sources GUARDED_BY(mu);
  };

  const int num_threads_;
  const int num_blocking_threads_;
  const int num_non_blocking_threads_;
  Eigen::MaxSizeVector<ThreadData> thread_data_;
  RunHandlerEnvironment env_;
  std::atomic<bool> cancelled_;
  string name_;
};

// Main worker thread loop.
void RunHandlerThreadPool::WorkerLoop(int thread_id,
                                      bool may_steal_blocking_work) {
  PerThread* pt = GetPerThread();
  pt->pool = this;
  pt->thread_id = thread_id;
  static constexpr int32 kMaxBlockingInflight = 10;

  while (!cancelled_) {
    Task t;
    ThreadWorkSource* tws = nullptr;
    bool task_from_blocking_queue = true;
    Eigen::MaxSizeVector<ThreadWorkSource*>* thread_work_sources =
        &thread_data_[thread_id].thread_work_sources;
    {
      // The mutex is not hot since its per thread and can only be held
      // by some other thread when a session run starts/finishes.
      mutex_lock l(thread_data_[thread_id].mu);

      for (int i = 0; i < thread_work_sources->size(); ++i) {
        tws = (*thread_work_sources)[i];
        // We want a smallish numbers of inter threads since
        // otherwise there will be contention in PropagateOutputs.
        // This is best effort policy.
        if (may_steal_blocking_work &&
            tws->GetInflightTaskCount(true) < kMaxBlockingInflight) {
          t = tws->PopBlockingTask();
          if (t.f) {
            break;
          }
        }
        if (i == 0) {
          // Always look for any work from the "primary" work source.
          // This way when we wake up a thread for a new closure we are
          // guaranteed it can be worked on.
          for (int j = 0; j < tws->NonBlockingWorkShardingFactor(); ++j) {
            t = tws->PopNonBlockingTask((j + thread_id) %
                                        tws->NonBlockingWorkShardingFactor());
            if (t.f) {
              task_from_blocking_queue = false;
              break;
            }
          }
          if (t.f) {
            break;
          }
        } else {
          t = tws->PopNonBlockingTask(thread_id %
                                      tws->NonBlockingWorkShardingFactor());
          if (t.f) {
            task_from_blocking_queue = false;
            break;
          }
        }
      }
    }
    if (t.f) {
      profiler::TraceMe activity(
          [=] {
            return strings::StrCat(task_from_blocking_queue ? "inter" : "intra",
                                   " #id = ", tws->GetTracemeId(), " ",
                                   thread_id, "#");
          },
          profiler::TraceMeLevel::kInfo);
      VLOG(2) << "Running " << (task_from_blocking_queue ? "inter" : "intra")
              << " work from " << tws->GetTracemeId();
      tws->IncrementInflightTaskCount(task_from_blocking_queue);
      env_.ExecuteTask(t);
      tws->DecrementInflightTaskCount(task_from_blocking_queue);
    } else {
      profiler::TraceMe activity(
          [=] {
            return strings::StrCat("Sleeping#thread_id=", thread_id, "#");
          },
          profiler::TraceMeLevel::kInfo);
      if (VLOG_IS_ON(4)) {
        mutex_lock l(thread_data_[thread_id].mu);
        for (int i = 0; i < thread_work_sources->size(); ++i) {
          VLOG(4) << "source id " << i << " "
                  << (*thread_work_sources)[i]->ToString();
        }
      }

      WaitForWork(may_steal_blocking_work, thread_id, kMaxBlockingInflight);
    }
  }
}

void RunHandlerThreadPool::WaitForWork(bool is_blocking, int thread_id,
                                       int32 max_blocking_inflight) {
  const int kMaxSleepMicros = 250;

  // The non-blocking thread will just sleep.
  if (!is_blocking) {
    Env::Default()->SleepForMicroseconds(kMaxSleepMicros);
    return;
  }

  ThreadWorkSource* tws = nullptr;
  {
    Eigen::MaxSizeVector<ThreadWorkSource*>* thread_work_sources =
        &thread_data_[thread_id].thread_work_sources;
    mutex_lock l(thread_data_[thread_id].mu);
    while (!cancelled_ && thread_work_sources->empty()) {
      // Wait until there is new request
      thread_data_[thread_id].sources_not_empty.wait(l);
    }
    if (cancelled_) {
      return;
    }
    tws = (*thread_work_sources)[0];
  }

  if (tws->GetInflightTaskCount(true) >= max_blocking_inflight) {
    // Sleep to reduce contention in PropagateOutputs
    Env::Default()->SleepForMicroseconds(kMaxSleepMicros);
  }
  tws->WaitForWork(kMaxSleepMicros);
}

}  // namespace

// Contains the concrete implementation of the RunHandler.
// Externally visible RunHandler class simply forwards the work to this one.
class RunHandler::Impl {
 public:
  explicit Impl(RunHandlerPool::Impl* pool_impl);

  ~Impl() {}

  thread::ThreadPoolInterface* thread_pool_interface() {
    return thread_pool_interface_.get();
  }

  // Stores now time (in microseconds) since unix epoch when the handler is
  // requested via RunHandlerPool::Get().
  uint64 start_time_us() const { return start_time_us_; }
  int64 step_id() const { return step_id_; }
  void ScheduleInterOpClosure(std::function<void()> fn);
  void ScheduleIntraOpClosure(std::function<void()> fn);

  void Reset(int64 step_id);

  RunHandlerPool::Impl* pool_impl() { return pool_impl_; }

  ThreadWorkSource* tws() { return &tws_; }

 private:
  class ThreadPoolInterfaceWrapper : public thread::ThreadPoolInterface {
   public:
    explicit ThreadPoolInterfaceWrapper(Impl* run_handler_impl)
        : run_handler_impl_(run_handler_impl) {}
    ~ThreadPoolInterfaceWrapper() override {}
    void Schedule(std::function<void()> fn) override;
    int NumThreads() const override;
    int CurrentThreadId() const override;

   private:
    RunHandler::Impl* run_handler_impl_ = nullptr;
  };

  RunHandlerPool::Impl* pool_impl_;  // NOT OWNED.
  uint64 start_time_us_;
  int64 step_id_;
  std::unique_ptr<thread::ThreadPoolInterface> thread_pool_interface_;
  ThreadWorkSource tws_;
};

// Contains shared state across all run handlers present in the pool. Also
// responsible for pool management decisions.
// This class is thread safe.
class RunHandlerPool::Impl {
 public:
  explicit Impl(int num_inter_op_threads, int num_intra_op_threads)
      : max_handlers_(static_cast<int32>(ParamFromEnvWithDefault(
            "TF_RUN_HANDLER_MAX_CONCURRENT_HANDLERS", kMaxConcurrentHandlers))),
        run_handler_thread_pool_(new RunHandlerThreadPool(
            num_inter_op_threads, num_intra_op_threads, Env::Default(),
            ThreadOptions(), "tf_run_handler_pool")),
        iterations_(0) {
    VLOG(1) << "Creating a RunHandlerPool with max handlers: " << max_handlers_;
    for (int i = 0; i < max_handlers_; ++i) {
      handlers_.emplace_back(new RunHandler::Impl(this));
      free_handlers_.push_back(handlers_.back().get());
    }
  }

  ~Impl() {
    // Sanity check that all handlers have been returned back to the pool before
    // destruction.
    DCHECK_EQ(handlers_.size(), max_handlers_);
    DCHECK_EQ(free_handlers_.size(), handlers_.size());
    DCHECK_EQ(sorted_active_handlers_.size(), 0);
    // Stop the threads in run_handler_thread_pool_ before freeing other
    // pointers. Otherwise a thread may try to access a pointer after the
    // pointer has been freed.
    run_handler_thread_pool_.reset();
  }

  RunHandlerThreadPool* run_handler_thread_pool() {
    return run_handler_thread_pool_.get();
  }

  std::unique_ptr<RunHandler> Get(int64 step_id) LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    while (free_handlers_.empty()) {
      one_handler_free_.wait(l);
    }
    // Remove the last entry from free_handlers_ and add to the end of
    // sorted_active_handlers_.
    auto* handler_impl = free_handlers_.back();
    handler_impl->Reset(step_id);
    // Sortedness isn't violated if we simply add at the end of the list, since
    // handlers are expected to be obtained in increasing order of time.
    sorted_active_handlers_.push_back(handler_impl);
    DCHECK_LE(sorted_active_handlers_.size(), max_handlers_);
    free_handlers_.pop_back();

    RecomputePoolStatsLocked();
    return WrapUnique<RunHandler>(new RunHandler(handler_impl));
  }

  void ReleaseHandler(RunHandler::Impl* handler) LOCKS_EXCLUDED(mu_) {
    {
      mutex_lock l(mu_);
      DCHECK_GT(sorted_active_handlers_.size(), 0);

      CHECK_EQ(handler->tws()->TaskQueueSize(true), 0);
      CHECK_EQ(handler->tws()->TaskQueueSize(false), 0);

      uint64 now = tensorflow::Env::Default()->NowMicros();
      double elapsed = (now - handler->start_time_us()) / 1000.0;
      time_hist_.Add(elapsed);

      // Erase from and update sorted_active_handlers_. Add it to the end of
      // free_handlers_.
      auto iter = std::find(sorted_active_handlers_.begin(),
                            sorted_active_handlers_.end(), handler);
      DCHECK(iter != sorted_active_handlers_.end())
          << "Unexpected handler: " << handler
          << " is being requested for release";

      // Remove this handler from this list and add it to the list of free
      // handlers.
      sorted_active_handlers_.erase(iter);
      free_handlers_.push_back(handler);
      DCHECK_LE(free_handlers_.size(), max_handlers_);

      RecomputePoolStatsLocked();
    }
    one_handler_free_.notify_one();
  }

 private:
  void RecomputePoolStatsLocked() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Maximum number of handlers pre-created during pool construction time. The
  // number has been chosen expecting each handler might at least want 1
  // inter-op thread for execution (during compute intensive workloads like
  // inference).
  const int max_handlers_;

  std::unique_ptr<RunHandlerThreadPool> run_handler_thread_pool_;
  // Thread compatible part used only by lock under RunHandlerPool.
  // Handlers are sorted by start time.
  // TODO(azaks): sort by the remaining latency budget.
  std::vector<RunHandler::Impl*> sorted_active_handlers_ GUARDED_BY(mu_);
  std::vector<RunHandler::Impl*> free_handlers_ GUARDED_BY(mu_);
  std::vector<std::unique_ptr<RunHandler::Impl>> handlers_ GUARDED_BY(mu_);
  // Histogram of elapsed runtime of every handler (in ms).
  histogram::Histogram time_hist_ GUARDED_BY(mu_);

  int64 iterations_ GUARDED_BY(mu_);
  condition_variable one_handler_free_;
  mutex mu_;
};

void RunHandlerPool::Impl::RecomputePoolStatsLocked() {
  int num_active_requests = sorted_active_handlers_.size();
  if (num_active_requests == 0) return;
  Eigen::MaxSizeVector<ThreadWorkSource*> thread_work_sources(
      num_active_requests);

  thread_work_sources.resize(num_active_requests);
  for (int i = 0; i < num_active_requests; ++i) {
    thread_work_sources[i] = sorted_active_handlers_[i]->tws();
    thread_work_sources[i]->SetRank(i);
  }

  int num_threads = run_handler_thread_pool()->NumThreads();
  int num_blocking_threads = run_handler_thread_pool()->NumBlockingThreads();
  int num_non_blocking_threads = num_threads - num_blocking_threads;

  std::vector<int> request_idx_list = ChooseRequestsWithExponentialDistribution(
      num_active_requests, num_blocking_threads);
  for (int i = 0; i < num_blocking_threads; ++i) {
    VLOG(2) << "Set work for tid=" << i
            << " with start_request_idx=" << request_idx_list[i];
    run_handler_thread_pool()->SetThreadWorkSources(i, request_idx_list[i],
                                                    thread_work_sources);
  }

  request_idx_list = ChooseRequestsWithExponentialDistribution(
      num_active_requests, num_non_blocking_threads);
  for (int i = 0; i < num_non_blocking_threads; ++i) {
    VLOG(2) << "Set work for tid=" << (i + num_blocking_threads)
            << " with start_request_idx=" << request_idx_list[i];
    run_handler_thread_pool()->SetThreadWorkSources(
        i + num_blocking_threads, request_idx_list[i], thread_work_sources);
  }

  if (iterations_++ % 50000 == 10 && VLOG_IS_ON(1)) {
    VLOG(1) << "Printing time histogram: " << time_hist_.ToString();
    VLOG(1) << "Active session runs: " << num_active_requests;
    uint64 now = tensorflow::Env::Default()->NowMicros();
    string times_str = "";
    string ids_str = "";
    for (int i = 0; i < num_active_requests; ++i) {
      if (i > 0) {
        times_str += " ";
        ids_str += " ";
      }

      times_str += strings::StrCat(
          (now - sorted_active_handlers_[i]->start_time_us()) / 1000.0, " ms.");
      ids_str +=
          strings::StrCat(sorted_active_handlers_[i]->tws()->GetTracemeId());
    }
    VLOG(1) << "Elapsed times are: " << times_str;
    VLOG(1) << "Step ids are: " << ids_str;
  }
}

// It is important to return a value such as:
// CurrentThreadId() in [0, NumThreads)
int RunHandler::Impl::ThreadPoolInterfaceWrapper::NumThreads() const {
  return run_handler_impl_->pool_impl_->run_handler_thread_pool()->NumThreads();
}

int RunHandler::Impl::ThreadPoolInterfaceWrapper::CurrentThreadId() const {
  return run_handler_impl_->pool_impl_->run_handler_thread_pool()
      ->CurrentThreadId();
}

void RunHandler::Impl::ThreadPoolInterfaceWrapper::Schedule(
    std::function<void()> fn) {
  return run_handler_impl_->ScheduleIntraOpClosure(std::move(fn));
}

RunHandler::Impl::Impl(RunHandlerPool::Impl* pool_impl)
    : pool_impl_(pool_impl) {
  thread_pool_interface_.reset(new ThreadPoolInterfaceWrapper(this));
  Reset(0);
}

void RunHandler::Impl::ScheduleInterOpClosure(std::function<void()> fn) {
  VLOG(3) << "Scheduling inter work for  " << tws()->GetTracemeId();
  pool_impl_->run_handler_thread_pool()->AddWorkToQueue(tws(), true,
                                                        std::move(fn));
}

void RunHandler::Impl::ScheduleIntraOpClosure(std::function<void()> fn) {
  VLOG(3) << "Scheduling intra work for " << tws()->GetTracemeId();
  pool_impl_->run_handler_thread_pool()->AddWorkToQueue(tws(), false,
                                                        std::move(fn));
}

void RunHandler::Impl::Reset(int64 step_id) {
  start_time_us_ = tensorflow::Env::Default()->NowMicros();
  step_id_ = step_id;
  tws_.SetTracemeId(step_id);
}

RunHandlerPool::RunHandlerPool(int num_inter_op_threads)
    : impl_(new Impl(num_inter_op_threads, 0)) {}

RunHandlerPool::RunHandlerPool(int num_inter_op_threads,
                               int num_intra_op_threads)
    : impl_(new Impl(num_inter_op_threads, num_intra_op_threads)) {}

RunHandlerPool::~RunHandlerPool() {}

std::unique_ptr<RunHandler> RunHandlerPool::Get(int64 step_id) {
  return impl_->Get(step_id);
}

RunHandler::RunHandler(Impl* impl) : impl_(impl) {}

void RunHandler::ScheduleInterOpClosure(std::function<void()> fn) {
  impl_->ScheduleInterOpClosure(std::move(fn));
}

thread::ThreadPoolInterface* RunHandler::AsIntraThreadPoolInterface() {
  return impl_->thread_pool_interface();
}

RunHandler::~RunHandler() { impl_->pool_impl()->ReleaseHandler(impl_); }

}  // namespace tensorflow
