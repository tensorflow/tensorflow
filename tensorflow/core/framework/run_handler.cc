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

class RunHandlerThreadPool {
 public:
  typedef typename RunHandlerEnvironment::Task Task;
  typedef Eigen::RunQueue<Task, 1024> Queue;

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
      thread_data_[i].thread.reset();
    }
  }

  struct ThreadWorkSource {
    ThreadWorkSource()
        : blocking_inflight(0), non_blocking_inflight(0), traceme_id(0) {}
    Queue blocking_work_queue;
    std::atomic<int64> blocking_inflight;
    mutex blocking_mu;
    Queue non_blocking_work_queue;
    std::atomic<int64> non_blocking_inflight;
    mutex non_blocking_mu;
    std::atomic<int64> traceme_id;
  };

  void AddWorkToQueue(Queue* q, mutex* mu, bool inter_work,
                      std::atomic<int64>* traceme_id,
                      std::function<void()> fn) {
    Task t = env_.CreateTask(std::move(fn));
    {
      mutex_lock l(*mu);
      // For a given queue, only one thread can call PushFront.
      t = q->PushFront(std::move(t));
      VLOG(3) << "Added " << (inter_work ? "inter" : "intra") << " work from "
              << traceme_id->load(std::memory_order_relaxed);
    }
    if (t.f) {
      VLOG(3) << "Running " << (inter_work ? "inter" : "intra") << " work from "
              << traceme_id->load(std::memory_order_relaxed);
      env_.ExecuteTask(t);
    }
  }

  // Set work queues from which the thread 'tid' can steal its work.
  void SetThreadWorkSources(
      int tid,
      const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources) {
    mutex_lock l(thread_data_[tid].mu);
    thread_data_[tid].thread_work_sources.resize(0);
    for (int i = 0; i < thread_work_sources.size(); ++i) {
      thread_data_[tid].thread_work_sources.emplace_back(
          thread_work_sources[i]);
    }
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

 private:
  struct ThreadData {
    ThreadData() : thread_work_sources(kMaxConcurrentHandlers) {}
    mutex mu;
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

  while (!cancelled_) {
    Task t;
    bool inter_work = true;
    std::atomic<int64>* inflight_counter = nullptr;
    int64 traceme_id = 0;
    Eigen::MaxSizeVector<ThreadWorkSource*>* thread_work_sources =
        &thread_data_[thread_id].thread_work_sources;
    {
      // The mutex is not hot since its per thread and can only be held
      // by some other thread when a session run starts/finishes.
      mutex_lock l(thread_data_[thread_id].mu);

      for (int i = 0; i < thread_work_sources->size(); ++i) {
        ThreadWorkSource* tws = (*thread_work_sources)[i];
        // We want a smallish numbers of inter threads since
        // otherwise there will be contention in PropagateOutputs.
        // This is best effort policy.
        static constexpr int32 kMaxBlockingInflight = 10;
        if (may_steal_blocking_work &&
            (tws->blocking_inflight.load(std::memory_order_relaxed) <
             kMaxBlockingInflight)) {
          t = tws->blocking_work_queue.PopBack();
          if (t.f) {
            inflight_counter = &(tws->blocking_inflight);
            traceme_id = tws->traceme_id.load(std::memory_order_relaxed);
            break;
          }
        }
        t = tws->non_blocking_work_queue.PopBack();
        if (t.f) {
          inflight_counter = &(tws->non_blocking_inflight);
          traceme_id = tws->traceme_id.load(std::memory_order_relaxed);
          inter_work = false;
          break;
        }
      }
    }
    if (t.f) {
      profiler::TraceMe activity(
          [=] {
            return strings::StrCat(inter_work ? "inter" : "intra", " ",
                                   "#id = ", traceme_id, " ", thread_id, "#");
          },
          profiler::TraceMeLevel::kInfo);
      VLOG(2) << "Running " << (inter_work ? "inter" : "intra") << " work from "
              << traceme_id;
      inflight_counter->fetch_add(1, std::memory_order_relaxed);
      env_.ExecuteTask(t);
      inflight_counter->fetch_sub(1, std::memory_order_relaxed);
    } else {
      profiler::TraceMe activity(
          [=] {
            return strings::StrCat("Sleeping#thread_id=", thread_id, "#");
          },
          profiler::TraceMeLevel::kInfo);
      if (VLOG_IS_ON(4)) {
        mutex_lock l(thread_data_[thread_id].mu);
        for (int i = 0; i < thread_work_sources->size(); ++i) {
          ThreadWorkSource* tws = (*thread_work_sources)[i];
          VLOG(4) << "source id " << i << " traceme_id = "
                  << tws->traceme_id.load(std::memory_order_relaxed)
                  << " inter queue size " << tws->blocking_work_queue.Size()
                  << " inter inflight "
                  << tws->blocking_inflight.load(std::memory_order_relaxed)
                  << " intra queue size " << tws->non_blocking_work_queue.Size()
                  << " intra inflight "
                  << tws->non_blocking_inflight.load(std::memory_order_relaxed);
        }
      }
      Env::Default()->SleepForMicroseconds(250);
    }
  }
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

  RunHandlerThreadPool::ThreadWorkSource* tws() { return &tws_; }

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
  RunHandlerThreadPool::ThreadWorkSource tws_;
};

// Contains shared state across all run handlers present in the pool. Also
// responsible for pool management decisions.
// This class is thread safe.
class RunHandlerPool::Impl {
 public:
  explicit Impl(int num_inter_op_threads, int num_intra_op_threads)
      : max_handlers_(kMaxConcurrentHandlers),
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

      CHECK_EQ(handler->tws()->blocking_work_queue.Size(), 0);
      CHECK_EQ(handler->tws()->non_blocking_work_queue.Size(), 0);

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
  Eigen::MaxSizeVector<RunHandlerThreadPool::ThreadWorkSource*>
      thread_work_sources(num_active_requests);

  thread_work_sources.resize(num_active_requests);

  for (int i = 0; i < num_active_requests; ++i) {
    thread_work_sources[i] = sorted_active_handlers_[i]->tws();
  }
  for (int i = 0; i < run_handler_thread_pool()->NumThreads(); ++i) {
    VLOG(2) << "Setting work for tid = " << i;
    run_handler_thread_pool()->SetThreadWorkSources(i, thread_work_sources);
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
          strings::StrCat(sorted_active_handlers_[i]->tws()->traceme_id.load(
              std::memory_order_relaxed));
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
  VLOG(3) << "Scheduling inter work for  "
          << tws()->traceme_id.load(std::memory_order_relaxed);
  pool_impl_->run_handler_thread_pool()->AddWorkToQueue(
      &tws()->blocking_work_queue, &tws()->blocking_mu, true,
      &tws()->traceme_id, std::move(fn));
}

void RunHandler::Impl::ScheduleIntraOpClosure(std::function<void()> fn) {
  VLOG(3) << "Scheduling inter work for "
          << tws()->traceme_id.load(std::memory_order_relaxed);
  pool_impl_->run_handler_thread_pool()->AddWorkToQueue(
      &tws()->non_blocking_work_queue, &tws()->non_blocking_mu, false,
      &tws()->traceme_id, std::move(fn));
}

void RunHandler::Impl::Reset(int64 step_id) {
  start_time_us_ = tensorflow::Env::Default()->NowMicros();
  step_id_ = step_id;
  tws_.traceme_id = step_id;
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
