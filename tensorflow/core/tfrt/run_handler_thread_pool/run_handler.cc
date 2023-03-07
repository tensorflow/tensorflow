/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>
#define EIGEN_USE_THREADS

#include <optional>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/core/threadpool_interface.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler.h"
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_util.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

typedef typename internal::RunHandlerEnvironment::Task Task;
typedef Eigen::RunQueue<Task, 1024> Queue;

}  // namespace

namespace internal {
RunHandlerEnvironment::RunHandlerEnvironment(
    tensorflow::Env* env, const tensorflow::ThreadOptions& thread_options,
    const std::string& name)
    : env_(env), thread_options_(thread_options), name_(name) {}

RunHandlerEnvironment::EnvThread* RunHandlerEnvironment::CreateThread(
    std::function<void()> f) {
  return env_->StartThread(thread_options_, name_, [=]() {
    // Set the processor flag to flush denormals to zero.
    tensorflow::port::ScopedFlushDenormal flush;
    // Set the processor rounding mode to ROUND TO NEAREST.
    tensorflow::port::ScopedSetRound round(FE_TONEAREST);
    if (thread_options_.numa_node != tensorflow::port::kNUMANoAffinity) {
      tensorflow::port::NUMASetThreadNodeAffinity(thread_options_.numa_node);
    }
    f();
  });
}

RunHandlerEnvironment::Task RunHandlerEnvironment::CreateTask(TaskFunction f) {
  uint64_t id = 0;
  if (tensorflow::tracing::EventCollector::IsEnabled()) {
    id = tensorflow::tracing::GetUniqueArg();
    tensorflow::tracing::RecordEvent(
        tensorflow::tracing::EventCategory::kScheduleClosure, id);
  }
  return Task{
      std::unique_ptr<TaskImpl>(new TaskImpl{
          std::move(f),
          tensorflow::Context(tensorflow::ContextKind::kThread),
          id,
      }),
  };
}

void RunHandlerEnvironment::ExecuteTask(const Task& t) {
  tensorflow::WithContext wc(t.f->context);
  tensorflow::tracing::ScopedRegion region(
      tensorflow::tracing::EventCategory::kRunClosure, t.f->trace_id);
  t.f->f();
}

void WaitOnWaiter(Waiter* waiter, Waiter* queue_head, tensorflow::mutex* mutex,
                  int sleep_micros, bool adaptive_sleep_time) {
  {
    tensorflow::mutex_lock l(*mutex);
    CHECK_EQ(waiter->next, waiter);  // Crash OK.
    CHECK_EQ(waiter->prev, waiter);  // Crash OK.

    // Add waiter to the LIFO queue
    waiter->prev = queue_head;
    waiter->next = queue_head->next;
    waiter->next->prev = waiter;
    waiter->prev->next = waiter;
  }
  {
    tensorflow::mutex_lock l(waiter->mu);
    waiter->num_waiting_threads++;
    int max_sleep_micros = adaptive_sleep_time
                               ? sleep_micros * waiter->num_waiting_threads
                               : sleep_micros;
    waiter->cv.wait_for(l, std::chrono::microseconds(max_sleep_micros));
    waiter->num_waiting_threads--;
  }

  tensorflow::mutex_lock l(*mutex);
  // Remove waiter from the LIFO queue. Note even when a waiter wakes up due
  // to a notification we cannot conclude the waiter is not in the queue.
  // This is due to the fact that a thread preempted right before notifying
  // may resume after a waiter got re-added.
  if (waiter->next != waiter) {
    CHECK(waiter->prev != waiter);  // Crash OK.
    waiter->next->prev = waiter->prev;
    waiter->prev->next = waiter->next;
    waiter->next = waiter;
    waiter->prev = waiter;
  } else {
    CHECK_EQ(waiter->prev, waiter);  // Crash OK.
  }
}

ThreadWorkSource::ThreadWorkSource()
    : non_blocking_work_sharding_factor_(
          static_cast<int32_t>(ParamFromEnvWithDefault(
              "TF_RUN_HANDLER_NUM_OF_NON_BLOCKING_QUEUES", 1))),
      non_blocking_work_queues_(non_blocking_work_sharding_factor_),
      blocking_inflight_(0),
      non_blocking_inflight_(0),
      pending_tasks_(0),
      traceme_id_(0),
      version_(0),
      sub_thread_pool_waiter_(nullptr) {
  queue_waiters_.next = &queue_waiters_;
  queue_waiters_.prev = &queue_waiters_;
  for (int i = 0; i < NonBlockingWorkShardingFactor(); ++i) {
    non_blocking_work_queues_.emplace_back(new NonBlockingQueue());
  }
}

ThreadWorkSource::~ThreadWorkSource() {
  for (int i = 0; i < non_blocking_work_queues_.size(); ++i) {
    delete non_blocking_work_queues_[i];
  }
}

Task ThreadWorkSource::EnqueueTask(Task t, bool is_blocking,
                                   bool enable_wake_up) {
  uint64_t id = t.f->trace_id;
  tensorflow::profiler::TraceMe activity(
      [id, is_blocking] {
        return tensorflow::profiler::TraceMeEncode(
            "Enqueue", {{"id", id}, {"is_blocking", is_blocking}});
      },
      tensorflow::profiler::TraceMeLevel::kInfo);
  tensorflow::mutex* mu = nullptr;
  Queue* task_queue = nullptr;
  thread_local int64_t closure_counter = 0;

  if (!is_blocking) {
    int queue_index = ++closure_counter % non_blocking_work_sharding_factor_;
    task_queue = &(non_blocking_work_queues_[queue_index]->queue);
    mu = &non_blocking_work_queues_[queue_index]->queue_op_mu;
  } else {
    task_queue = &blocking_work_queue_;
    mu = &blocking_queue_op_mu_;
  }

  {
    tensorflow::mutex_lock l(*mu);
    // For a given queue, only one thread can call PushFront.
    t = task_queue->PushFront(std::move(t));
  }
  IncrementPendingTaskCount();

  if (enable_wake_up) {
    // Try to wake up waiting thread if there is any for the given sub thread
    // pool. We could potentially wake up threads in other sub thread pool if we
    // cannot find any waiting threads in the given sub thread pool.
    // The wake up logic is best effort as the thread may be right before being
    // added to the waiting queue or start waiting on the condition variable.
    // However a thread will wake in short period of time in case a notification
    // is missed.
    Waiter* w = nullptr;

    Waiter* waiter_queue;
    tensorflow::mutex* waiter_queue_mu;
    {
      // When we use multiple sub thread pools, free threads wait on sub
      // thread pool waiting queues. Wake up threads from sub thread waiting
      // queues.
      // The waiting queues are defined at RunHandlerPool.
      // Get the waiter_queue and corresponding mutex. Note, the thread work
      // source may change afterwards if a new request comes or an old request
      // finishes.
      tensorflow::tf_shared_lock lock(run_handler_waiter_mu_);
      waiter_queue = sub_thread_pool_waiter_;
      waiter_queue_mu = sub_thread_pool_waiter_mu_;
    }
    {
      tensorflow::mutex_lock l(*waiter_queue_mu);
      if (waiter_queue->next != waiter_queue) {
        // Remove waiter from the LIFO queue
        w = waiter_queue->next;

        CHECK(w->prev != w);  // Crash OK.
        CHECK(w->next != w);  // Crash OK.

        w->next->prev = w->prev;
        w->prev->next = w->next;

        // Use `w->next == &w` to indicate that the waiter has been removed
        // from the queue.
        w->next = w;
        w->prev = w;
      }
    }
    if (w != nullptr) {
      w->cv.notify_one();
    }
  }
  VLOG(3) << "Added " << (is_blocking ? "inter" : "intra") << " work from "
          << traceme_id_.load(std::memory_order_relaxed);
  return t;
}

Task ThreadWorkSource::PopBlockingTask() {
  return blocking_work_queue_.PopBack();
}

Task ThreadWorkSource::PopNonBlockingTask(int start_index,
                                          bool search_from_all_queue) {
  Task t;
  unsigned sharding_factor = NonBlockingWorkShardingFactor();
  for (unsigned j = 0; j < sharding_factor; ++j) {
    t = non_blocking_work_queues_[(start_index + j) % sharding_factor]
            ->queue.PopBack();
    if (t.f) {
      return t;
    }
    if (!search_from_all_queue) {
      break;
    }
  }
  return t;
}

int ThreadWorkSource::TaskQueueSize(bool is_blocking) {
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

int64_t ThreadWorkSource::GetTracemeId() {
  return traceme_id_.load(std::memory_order_relaxed);
}

void ThreadWorkSource::SetTracemeId(int64_t value) { traceme_id_ = value; }

void ThreadWorkSource::SetWaiter(uint64_t version, Waiter* waiter,
                                 tensorflow::mutex* mutex) {
  {
    tensorflow::tf_shared_lock lock(run_handler_waiter_mu_);
    // Most of the request won't change sub pool for recomputation.
    // Optimization for avoiding holding exclusive lock to reduce contention.
    if (sub_thread_pool_waiter_ == waiter) {
      return;
    }
    // If the current version is a newer version, no need to update.
    if (version_ > version) {
      return;
    }
  }

  tensorflow::mutex_lock l(run_handler_waiter_mu_);
  sub_thread_pool_waiter_ = waiter;
  sub_thread_pool_waiter_mu_ = mutex;
  version_ = version;
}

int64_t ThreadWorkSource::GetInflightTaskCount(bool is_blocking) {
  std::atomic<int64_t>* counter =
      is_blocking ? &blocking_inflight_ : &non_blocking_inflight_;
  return counter->load(std::memory_order_relaxed);
}

void ThreadWorkSource::IncrementInflightTaskCount(bool is_blocking) {
  std::atomic<int64_t>* counter =
      is_blocking ? &blocking_inflight_ : &non_blocking_inflight_;
  counter->fetch_add(1, std::memory_order_relaxed);
}

void ThreadWorkSource::DecrementInflightTaskCount(bool is_blocking) {
  std::atomic<int64_t>* counter =
      is_blocking ? &blocking_inflight_ : &non_blocking_inflight_;
  counter->fetch_sub(1, std::memory_order_relaxed);
}

int64_t ThreadWorkSource::GetPendingTaskCount() {
  return pending_tasks_.load(std::memory_order_acquire);
}

void ThreadWorkSource::IncrementPendingTaskCount() {
  pending_tasks_.fetch_add(1, std::memory_order_relaxed);
}

void ThreadWorkSource::DecrementPendingTaskCount() {
  // std::memory_order_release prevents reorder with op execution.
  pending_tasks_.fetch_sub(1, std::memory_order_release);
}

unsigned ThreadWorkSource::NonBlockingWorkShardingFactor() {
  return non_blocking_work_sharding_factor_;
}

std::string ThreadWorkSource::ToString() {
  return tensorflow::strings::StrCat(
      "traceme_id = ", GetTracemeId(),
      ", inter queue size = ", TaskQueueSize(true),
      ", inter inflight = ", GetInflightTaskCount(true),
      ", intra queue size = ", TaskQueueSize(false),
      ", intra inflight = ", GetInflightTaskCount(false));
}

RunHandlerThreadPool::RunHandlerThreadPool(
    Options options, tensorflow::Env* env,
    const tensorflow::ThreadOptions& thread_options, const std::string& name,
    Eigen::MaxSizeVector<tensorflow::mutex>* waiters_mu,
    Eigen::MaxSizeVector<Waiter>* queue_waiters)
    : num_threads_(options.num_blocking_threads +
                   options.num_non_blocking_threads),
      num_blocking_threads_(options.num_blocking_threads),
      num_non_blocking_threads_(options.num_non_blocking_threads),
      adaptive_sleep_time_(options.use_adaptive_waiting_time),
      wait_if_no_active_request_(options.wait_if_no_active_request),
      non_blocking_thread_sleep_time_(
          options.non_blocking_threads_sleep_time_micro_sec),
      blocking_thread_max_waiting_time_(
          options.blocking_threads_max_sleep_time_micro_sec),
      enable_wake_up_(options.enable_wake_up),
      thread_data_(num_threads_),
      env_(env, thread_options, name),
      name_(name),
      waiters_mu_(waiters_mu),
      queue_waiters_(queue_waiters),
      num_threads_in_sub_thread_pool_(options.num_threads_in_sub_thread_pool),
      sub_thread_pool_end_request_percentage_(
          options.sub_thread_request_percentage) {
  thread_data_.resize(num_threads_);
  for (int i = 0; i < num_threads_; ++i) {
    thread_data_[i].new_thread_work_sources =
        std::make_unique<Eigen::MaxSizeVector<ThreadWorkSource*>>(
            options.max_concurrent_handler);
    thread_data_[i].current_thread_work_sources =
        std::make_unique<Eigen::MaxSizeVector<ThreadWorkSource*>>(
            options.max_concurrent_handler);
  }
  VLOG(1) << "Creating RunHandlerThreadPool " << name << " with  "
          << num_blocking_threads_ << " blocking threads and "
          << num_non_blocking_threads_ << " non-blocking threads.";
}

RunHandlerThreadPool::~RunHandlerThreadPool() {
  VLOG(1) << "Exiting RunHandlerThreadPool " << name_;

  cancelled_ = true;
  for (size_t i = 0; i < thread_data_.size(); ++i) {
    {
      tensorflow::mutex_lock l(thread_data_[i].mu);
      thread_data_[i].sources_not_empty.notify_all();
    }
    thread_data_[i].thread.reset();
  }
}

void RunHandlerThreadPool::Start() {
  cancelled_ = false;
  int num_blocking_threads = num_blocking_threads_;
  for (int i = 0; i < num_threads_; i++) {
    int sub_thread_pool_id = num_threads_in_sub_thread_pool_.size() - 1;
    for (int j = 0; j < num_threads_in_sub_thread_pool_.size(); ++j) {
      if (i < num_threads_in_sub_thread_pool_[j]) {
        sub_thread_pool_id = j;
        break;
      }
    }
    thread_data_[i].sub_thread_pool_id = sub_thread_pool_id;
    thread_data_[i].thread.reset(
        env_.CreateThread([this, i, num_blocking_threads]() {
          WorkerLoop(i, i < num_blocking_threads);
        }));
  }
}

void RunHandlerThreadPool::StartOneThreadForTesting() {
  cancelled_ = false;
  thread_data_[0].sub_thread_pool_id = 0;
  thread_data_[0].thread.reset(
      env_.CreateThread([this]() { WorkerLoop(0, true); }));
}

void RunHandlerThreadPool::AddWorkToQueue(ThreadWorkSource* tws,
                                          bool is_blocking, TaskFunction fn) {
  Task t = env_.CreateTask(std::move(fn));
  t = tws->EnqueueTask(std::move(t), is_blocking, enable_wake_up_);
  if (t.f) {
    VLOG(3) << "Running " << (is_blocking ? "inter" : "intra") << " work for "
            << tws->GetTracemeId();
    env_.ExecuteTask(t);
  }
}

void RunHandlerThreadPool::SetThreadWorkSources(
    int tid, uint64_t version,
    const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources) {
  tensorflow::mutex_lock l(thread_data_[tid].mu);
  if (version > thread_data_[tid].new_version) {
    thread_data_[tid].new_version = version;
  } else {
    // A newer version is already updated. No need to update.
    return;
  }
  auto original_size = thread_data_[tid].current_thread_work_sources->size();
  thread_data_[tid].new_thread_work_sources->resize(0);
  for (int i = 0; i < thread_work_sources.size(); ++i) {
    thread_data_[tid].new_thread_work_sources->emplace_back(
        thread_work_sources[i]);
  }
  if (original_size == 0) {
    thread_data_[tid].sources_not_empty.notify_all();
  }
}

RunHandlerThreadPool::PerThread* RunHandlerThreadPool::GetPerThread() {
  thread_local RunHandlerThreadPool::PerThread per_thread_;
  RunHandlerThreadPool::PerThread* pt = &per_thread_;
  return pt;
}

int RunHandlerThreadPool::CurrentThreadId() const {
  const PerThread* pt = const_cast<RunHandlerThreadPool*>(this)->GetPerThread();
  if (pt->pool == this) {
    return pt->thread_id;
  } else {
    return -1;
  }
}

int RunHandlerThreadPool::NumThreads() const { return num_threads_; }

int RunHandlerThreadPool::NumBlockingThreads() const {
  return num_blocking_threads_;
}

int RunHandlerThreadPool::NumNonBlockingThreads() const {
  return num_non_blocking_threads_;
}

RunHandlerThreadPool::ThreadData::ThreadData()
    : new_version(0), current_index(0), current_version(0) {}

Task RunHandlerThreadPool::FindTask(
    int searching_range_start, int searching_range_end, int thread_id,
    int sub_thread_pool_id, int max_blocking_inflight,
    bool may_steal_blocking_work,
    const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources,
    bool* task_from_blocking_queue, ThreadWorkSource** tws) {
  Task t;
  int current_index = thread_data_[thread_id].current_index;
  *task_from_blocking_queue = false;

  for (int i = 0; i < searching_range_end - searching_range_start; ++i) {
    if (current_index >= searching_range_end ||
        current_index < searching_range_start) {
      current_index = searching_range_start;
    }
    *tws = thread_work_sources[current_index];
    ++current_index;

    // For blocking thread, search for blocking tasks first.
    if (may_steal_blocking_work &&
        (*tws)->GetInflightTaskCount(true) < max_blocking_inflight) {
      t = (*tws)->PopBlockingTask();
      if (t.f) {
        *task_from_blocking_queue = true;
        break;
      }
    }

    // Search for non-blocking tasks.
    t = (*tws)->PopNonBlockingTask(thread_id, true);
    if (t.f) {
      break;
    }
  }
  thread_data_[thread_id].current_index = current_index;
  return t;
}

// Main worker thread loop.
void RunHandlerThreadPool::WorkerLoop(int thread_id,
                                      bool may_steal_blocking_work) {
  PerThread* pt = GetPerThread();
  pt->pool = this;
  pt->thread_id = thread_id;
  static constexpr int32_t kMaxBlockingInflight = 10;

  while (!cancelled_) {
    Task t;
    ThreadWorkSource* tws = nullptr;
    bool task_from_blocking_queue = true;
    int sub_thread_pool_id;
    // Get the current thread work sources.
    {
      tensorflow::mutex_lock l(thread_data_[thread_id].mu);
      if (thread_data_[thread_id].current_version <
          thread_data_[thread_id].new_version) {
        thread_data_[thread_id].current_version =
            thread_data_[thread_id].new_version;
        thread_data_[thread_id].current_thread_work_sources.swap(
            thread_data_[thread_id].new_thread_work_sources);
      }
    }
    Eigen::MaxSizeVector<ThreadWorkSource*>* thread_work_sources =
        thread_data_[thread_id].current_thread_work_sources.get();
    sub_thread_pool_id = thread_data_[thread_id].sub_thread_pool_id;
    int active_requests = thread_work_sources->size();
    if (may_steal_blocking_work) {
      // Each thread will first look for tasks from requests that belongs to
      // its sub thread pool.
      int search_range_start =
          sub_thread_pool_id == 0
              ? 0
              : active_requests *
                    sub_thread_pool_end_request_percentage_[sub_thread_pool_id -
                                                            1];
      int search_range_end =
          active_requests *
          sub_thread_pool_end_request_percentage_[sub_thread_pool_id];
      search_range_end = std::min(
          active_requests, std::max(search_range_end, search_range_start + 1));

      t = FindTask(search_range_start, search_range_end, thread_id,
                   sub_thread_pool_id, kMaxBlockingInflight,
                   /*may_steal_blocking_work=*/true, *thread_work_sources,
                   &task_from_blocking_queue, &tws);
      if (!t.f) {
        // Search from all requests if the thread cannot find tasks from
        // requests that belong to its own sub thread pool.
        t = FindTask(0, active_requests, thread_id, sub_thread_pool_id,
                     kMaxBlockingInflight,
                     /*may_steal_blocking_work=*/true, *thread_work_sources,
                     &task_from_blocking_queue, &tws);
      }
    } else {
      // For non-blocking threads, it will always search from all pending
      // requests.
      t = FindTask(0, active_requests, thread_id, sub_thread_pool_id,
                   kMaxBlockingInflight,
                   /*may_steal_blocking_work=*/false, *thread_work_sources,
                   &task_from_blocking_queue, &tws);
    }
    if (t.f) {
      VLOG(2) << "Running " << (task_from_blocking_queue ? "inter" : "intra")
              << " work from " << tws->GetTracemeId();
      tws->IncrementInflightTaskCount(task_from_blocking_queue);
      env_.ExecuteTask(t);
      tws->DecrementInflightTaskCount(task_from_blocking_queue);
      tws->DecrementPendingTaskCount();
    } else {
      tensorflow::profiler::TraceMe activity(
          [thread_id] {
            return tensorflow::profiler::TraceMeEncode(
                "Sleeping", {{"thread_id", thread_id}});
          },
          tensorflow::profiler::TraceMeLevel::kInfo);
      if (VLOG_IS_ON(4)) {
        for (int i = 0; i < thread_work_sources->size(); ++i) {
          VLOG(4) << "source id " << i << " "
                  << (*thread_work_sources)[i]->ToString();
        }
      }
      WaitForWorkInSubThreadPool(thread_id, may_steal_blocking_work,
                                 sub_thread_pool_id);
    }
  }
}

void RunHandlerThreadPool::WaitForWorkInSubThreadPool(int thread_id,
                                                      bool is_blocking,
                                                      int sub_thread_pool_id) {
  if (wait_if_no_active_request_) {
    tensorflow::mutex_lock l(thread_data_[thread_id].mu);
    if (thread_data_[thread_id].new_version >
        thread_data_[thread_id].current_version) {
      thread_data_[thread_id].current_thread_work_sources.swap(
          thread_data_[thread_id].new_thread_work_sources);
      thread_data_[thread_id].current_version =
          thread_data_[thread_id].new_version;
    }
    Eigen::MaxSizeVector<ThreadWorkSource*>* thread_work_sources =
        thread_data_[thread_id].current_thread_work_sources.get();
    bool already_wait = false;
    while (!cancelled_ && thread_work_sources->empty()) {
      // Wait until there is new request.
      thread_data_[thread_id].sources_not_empty.wait(l);
      if (thread_data_[thread_id].new_version >
          thread_data_[thread_id].current_version) {
        thread_data_[thread_id].current_thread_work_sources.swap(
            thread_data_[thread_id].new_thread_work_sources);
        thread_data_[thread_id].current_version =
            thread_data_[thread_id].new_version;
        thread_work_sources =
            thread_data_[thread_id].current_thread_work_sources.get();
      }
      already_wait = true;
    }
    if (already_wait || cancelled_) {
      return;
    }
  }

  // The non-blocking thread will just sleep.
  if (!is_blocking) {
    tensorflow::Env::Default()->SleepForMicroseconds(
        non_blocking_thread_sleep_time_);
    return;
  }

  if (enable_wake_up_) {
    thread_local Waiter waiter;
    WaitOnWaiter(&waiter, &(*queue_waiters_)[sub_thread_pool_id],
                 &(*waiters_mu_)[sub_thread_pool_id],
                 blocking_thread_max_waiting_time_, adaptive_sleep_time_);
  } else {
    tensorflow::Env::Default()->SleepForMicroseconds(
        blocking_thread_max_waiting_time_);
  }
}

}  // namespace internal

// Contains the concrete implementation of the RunHandler.
// Externally visible RunHandler class simply forwards the work to this one.
class RunHandler::Impl {
 public:
  explicit Impl(RunHandlerPool::Impl* pool_impl);

  ~Impl() {}

  // Stores now time (in microseconds) since unix epoch when the handler is
  // requested via RunHandlerPool::Get().
  uint64_t start_time_us() const { return start_time_us_; }
  int64_t step_id() const { return step_id_; }
  void ScheduleInterOpClosure(TaskFunction fn);
  void ScheduleIntraOpClosure(TaskFunction fn);

  void Reset(int64_t step_id, const RunHandlerOptions& options);

  RunHandlerPool::Impl* pool_impl() { return pool_impl_; }

  tensorflow::thread::ThreadPoolInterface* thread_pool_interface() {
    return &eigen_thread_pool_;
  }

  internal::ThreadWorkSource* tws() { return &tws_; }

  int64_t priority() const { return options_.priority; }

 private:
  class RunHandlerEigenThreadPool
      : public tensorflow::thread::ThreadPoolInterface {
   public:
    explicit RunHandlerEigenThreadPool(RunHandler::Impl* run_handler)
        : run_handler_(run_handler) {
      DCHECK(run_handler);
    }

    void Schedule(std::function<void()> fn) override {
      run_handler_->ScheduleIntraOpClosure(tensorflow::tfrt_stub::WrapWork(
          run_handler_->tws()->GetTracemeId(), "intra", std::move(fn)));
    }

    int NumThreads() const override;
    int CurrentThreadId() const override;

   private:
    RunHandler::Impl* run_handler_;
  };

  RunHandlerPool::Impl* pool_impl_;  // NOT OWNED.
  RunHandlerEigenThreadPool eigen_thread_pool_;
  uint64_t start_time_us_;
  int64_t step_id_;
  internal::ThreadWorkSource tws_;
  RunHandlerOptions options_;
};

// Contains shared state across all run handlers present in the pool. Also
// responsible for pool management decisions.
// This class is thread safe.
class RunHandlerPool::Impl {
 public:
  explicit Impl(Options options)
      : max_handlers_(options.max_concurrent_handler),
        waiters_mu_(options.num_sub_thread_pool),
        queue_waiters_(options.num_sub_thread_pool),
        run_handler_thread_pool_(new internal::RunHandlerThreadPool(
            internal::RunHandlerThreadPool::Options(
                options.num_inter_op_threads, options.num_intra_op_threads,
                options.wait_if_no_active_request,
                options.non_blocking_threads_sleep_time_micro_sec,
                options.blocking_threads_max_sleep_time_micro_sec,
                options.use_adaptive_waiting_time, options.enable_wake_up,
                options.max_concurrent_handler,
                options.num_threads_in_sub_thread_pool,
                options.sub_thread_request_percentage),
            tensorflow::Env::Default(), tensorflow::ThreadOptions(),
            "tf_run_handler_pool", &waiters_mu_, &queue_waiters_)),
        iterations_(0),
        version_(0),
        wait_if_no_active_request_(options.wait_if_no_active_request),
        sub_thread_pool_end_request_percentage_(
            options.sub_thread_request_percentage) {
    VLOG(1) << "Creating a RunHandlerPool with max handlers: " << max_handlers_;
    free_handlers_.reserve(max_handlers_);
    handlers_.reserve(max_handlers_);
    for (int i = 0; i < max_handlers_; ++i) {
      handlers_.emplace_back(new RunHandler::Impl(this));
      free_handlers_.push_back(handlers_.back().get());
    }
    queue_waiters_.resize(options.num_sub_thread_pool);
    waiters_mu_.resize(options.num_sub_thread_pool);
    for (auto& queue_waiter : queue_waiters_) {
      queue_waiter.next = &queue_waiter;
      queue_waiter.prev = &queue_waiter;
    }
    run_handler_thread_pool_->Start();
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

  internal::RunHandlerThreadPool* run_handler_thread_pool() {
    return run_handler_thread_pool_.get();
  }

  bool has_free_handler() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !free_handlers_.empty();
  }

  std::unique_ptr<RunHandler> Get(int64_t step_id, int64_t timeout_in_ms,
                                  const RunHandlerOptions& options)
      TF_LOCKS_EXCLUDED(mu_) {
    thread_local std::unique_ptr<
        Eigen::MaxSizeVector<internal::ThreadWorkSource*>>
        thread_work_sources =
            std::unique_ptr<Eigen::MaxSizeVector<internal::ThreadWorkSource*>>(
                new Eigen::MaxSizeVector<internal::ThreadWorkSource*>(
                    max_handlers_));
    uint64_t version;
    int num_active_requests;
    RunHandler::Impl* handler_impl;
    {
      tensorflow::mutex_lock l(mu_);
      if (!has_free_handler()) {
        tensorflow::profiler::TraceMe activity(
            [step_id] {
              return tensorflow::profiler::TraceMeEncode(
                  "WaitingForHandler", {{"step_id", step_id}});
            },
            tensorflow::profiler::TraceMeLevel::kInfo);
        if (timeout_in_ms == 0) {
          mu_.Await(tensorflow::Condition(this, &Impl::has_free_handler));
        } else if (!mu_.AwaitWithDeadline(
                       tensorflow::Condition(this, &Impl::has_free_handler),
                       tensorflow::EnvTime::NowNanos() +
                           timeout_in_ms * 1000 * 1000)) {
          return nullptr;
        }
      }
      // Remove the last entry from free_handlers_ and add to the end of
      // sorted_active_handlers_.
      handler_impl = free_handlers_.back();
      handler_impl->Reset(step_id, options);
      free_handlers_.pop_back();

      num_active_requests = sorted_active_handlers_.size() + 1;
      thread_work_sources->resize(num_active_requests);
      int priority = options.priority;
      auto it = sorted_active_handlers_.cbegin();
      bool new_handler_inserted = false;
      for (int i = 0; i < num_active_requests; ++i) {
        if (!new_handler_inserted && (it == sorted_active_handlers_.cend() ||
                                      priority > (*it)->priority())) {
          sorted_active_handlers_.insert(it, handler_impl);
          new_handler_inserted = true;
          // Point to the newly added handler.
          --it;
        }
        (*thread_work_sources)[i] = (*it)->tws();
        ++it;
      }
      version = ++version_;
    }
    RecomputePoolStats(num_active_requests, version, *thread_work_sources);
    return tensorflow::WrapUnique<RunHandler>(new RunHandler(handler_impl));
  }

  void ReleaseHandler(RunHandler::Impl* handler) TF_LOCKS_EXCLUDED(mu_) {
    tensorflow::mutex_lock l(mu_);
    DCHECK_GT(sorted_active_handlers_.size(), 0);

    CHECK_EQ(handler->tws()->TaskQueueSize(true), 0);   // Crash OK.
    CHECK_EQ(handler->tws()->TaskQueueSize(false), 0);  // Crash OK.

    uint64_t now = tensorflow::EnvTime::NowMicros();
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
    LogInfo();

    // We do not recompute pool stats all the time. The side effect is that
    // there may be empty thread work sources in the queue. However, any new
    // requests will trigger recomputation.
    if (wait_if_no_active_request_ && sorted_active_handlers_.empty()) {
      thread_local auto thread_work_sources =
          std::make_unique<Eigen::MaxSizeVector<internal::ThreadWorkSource*>>(
              max_handlers_);
      thread_work_sources->resize(0);
      auto version = ++version_;
      RecomputePoolStats(0, version, *thread_work_sources);
    }
  }

  std::vector<int64_t> GetActiveHandlerPrioritiesForTesting()
      TF_LOCKS_EXCLUDED(mu_) {
    tensorflow::mutex_lock l(mu_);
    std::vector<int64_t> ret;
    for (const auto& handler_impl : sorted_active_handlers_) {
      ret.push_back(handler_impl->priority());
    }
    return ret;
  }

  void Quiesce() TF_LOCKS_EXCLUDED(mu_) {
    while (true) {
      {
        tensorflow::tf_shared_lock l(mu_);
        bool all_empty = true;
        for (const auto& handler : sorted_active_handlers_) {
          if (handler->tws()->GetPendingTaskCount() != 0) {
            all_empty = false;
            break;
          }
        }
        if (all_empty) {
          break;
        }
      }
      // Sleep
      const int sleep_time = 50000;
      tensorflow::Env::Default()->SleepForMicroseconds(sleep_time);
    }
  }

 private:
  void RecomputePoolStats(
      int num_active_requests, uint64_t version,
      const Eigen::MaxSizeVector<internal::ThreadWorkSource*>&
          thread_work_sources);

  void LogInfo() TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Maximum number of handlers pre-created during pool construction time. The
  // number has been chosen expecting each handler might at least want 1
  // inter-op thread for execution (during compute intensive workloads like
  // inference).
  const int max_handlers_;

  Eigen::MaxSizeVector<tensorflow::mutex> waiters_mu_;
  Eigen::MaxSizeVector<internal::Waiter> queue_waiters_;

  std::unique_ptr<internal::RunHandlerThreadPool> run_handler_thread_pool_;
  // Thread compatible part used only by lock under RunHandlerPool.
  // Handlers are sorted by start time.
  // TODO(azaks): sort by the remaining latency budget.
  // TODO(chaox): Consider other data structure for maintaining the sorted
  // active handlers if the searching overhead(currently O(n)) becomes the
  // bottleneck.
  std::list<RunHandler::Impl*> sorted_active_handlers_ TF_GUARDED_BY(mu_);
  std::vector<RunHandler::Impl*> free_handlers_ TF_GUARDED_BY(mu_);
  std::vector<std::unique_ptr<RunHandler::Impl>> handlers_ TF_GUARDED_BY(mu_);

  // Histogram of elapsed runtime of every handler (in ms).
  tensorflow::histogram::Histogram time_hist_ TF_GUARDED_BY(mu_);

  int64_t iterations_ TF_GUARDED_BY(mu_);
  tensorflow::mutex mu_;
  int64_t version_ TF_GUARDED_BY(mu_);
  bool wait_if_no_active_request_;
  const std::vector<double> sub_thread_pool_end_request_percentage_;
};

void RunHandlerPool::Impl::RecomputePoolStats(
    int num_active_requests, uint64_t version,
    const Eigen::MaxSizeVector<internal::ThreadWorkSource*>&
        thread_work_sources) {
  int sub_thread_pool_id = 0;
  for (int i = 0; i < num_active_requests; ++i) {
    while (
        sub_thread_pool_id <
            sub_thread_pool_end_request_percentage_.size() - 1 &&
        i >= num_active_requests *
                 sub_thread_pool_end_request_percentage_[sub_thread_pool_id]) {
      sub_thread_pool_id++;
    }
    thread_work_sources[i]->SetWaiter(version,
                                      &queue_waiters_[sub_thread_pool_id],
                                      &waiters_mu_[sub_thread_pool_id]);
  }

  int num_threads = run_handler_thread_pool()->NumThreads();
  int num_blocking_threads = run_handler_thread_pool()->NumBlockingThreads();
  int num_non_blocking_threads = num_threads - num_blocking_threads;

  for (int i = 0; i < num_blocking_threads + num_non_blocking_threads; ++i) {
    run_handler_thread_pool()->SetThreadWorkSources(i, version,
                                                    thread_work_sources);
  }
}

void RunHandlerPool::Impl::LogInfo() {
  if (iterations_++ % 50000 == 10 && VLOG_IS_ON(1)) {
    int num_active_requests = sorted_active_handlers_.size();
    VLOG(1) << "Printing time histogram: " << time_hist_.ToString();
    VLOG(1) << "Active session runs: " << num_active_requests;
    uint64_t now = tensorflow::Env::Default()->NowMicros();
    std::string times_str = "";
    std::string ids_str = "";
    auto it = sorted_active_handlers_.cbegin();
    for (int i = 0; i < num_active_requests; ++i) {
      if (i > 0) {
        times_str += " ";
        ids_str += " ";
      }

      times_str += tensorflow::strings::StrCat(
          (now - (*it)->start_time_us()) / 1000.0, " ms.");
      ids_str += tensorflow::strings::StrCat((*it)->tws()->GetTracemeId());
      ++it;
    }
    VLOG(1) << "Elapsed times are: " << times_str;
    VLOG(1) << "Step ids are: " << ids_str;
  }
}

RunHandler::Impl::Impl(RunHandlerPool::Impl* pool_impl)
    : pool_impl_(pool_impl), eigen_thread_pool_(this) {
  Reset(0, RunHandlerOptions());
}

void RunHandler::Impl::ScheduleInterOpClosure(TaskFunction fn) {
  VLOG(3) << "Scheduling inter work for  " << tws()->GetTracemeId();
  pool_impl_->run_handler_thread_pool()->AddWorkToQueue(tws(), true,
                                                        std::move(fn));
}

void RunHandler::Impl::ScheduleIntraOpClosure(TaskFunction fn) {
  VLOG(3) << "Scheduling intra work for " << tws()->GetTracemeId();
  pool_impl_->run_handler_thread_pool()->AddWorkToQueue(tws(), false,
                                                        std::move(fn));
}

void RunHandler::Impl::Reset(int64_t step_id,
                             const RunHandlerOptions& options) {
  start_time_us_ = tensorflow::Env::Default()->NowMicros();
  step_id_ = step_id;
  options_ = options;
  tws_.SetTracemeId(step_id);
}

int RunHandler::Impl::RunHandlerEigenThreadPool::NumThreads() const {
  return run_handler_->pool_impl_->run_handler_thread_pool()->NumThreads();
}

int RunHandler::Impl::RunHandlerEigenThreadPool::CurrentThreadId() const {
  return run_handler_->pool_impl_->run_handler_thread_pool()->CurrentThreadId();
}

RunHandlerPool::RunHandlerPool(Options options) : impl_(new Impl(options)) {}

RunHandlerPool::~RunHandlerPool() {}

std::unique_ptr<RunHandler> RunHandlerPool::Get(
    int64_t step_id, int64_t timeout_in_ms, const RunHandlerOptions& options) {
  return impl_->Get(step_id, timeout_in_ms, options);
}

std::vector<int64_t> RunHandlerPool::GetActiveHandlerPrioritiesForTesting()
    const {
  return impl_->GetActiveHandlerPrioritiesForTesting();
}

void RunHandlerPool::Quiesce() const { impl_->Quiesce(); }

RunHandler::RunHandler(Impl* impl) : impl_(impl) {}

void RunHandler::ScheduleInterOpClosure(TaskFunction fn) {
  impl_->ScheduleInterOpClosure(std::move(fn));
}

void RunHandler::ScheduleIntraOpClosure(TaskFunction fn) {
  impl_->ScheduleInterOpClosure(std::move(fn));
}

int RunHandler::NumThreads() const {
  return impl_->pool_impl()->run_handler_thread_pool()->NumThreads();
}

int64_t RunHandler::step_id() const { return impl_->step_id(); }

tensorflow::thread::ThreadPoolInterface*
RunHandler::AsIntraThreadPoolInterface() const {
  return impl_->thread_pool_interface();
}

RunHandler::~RunHandler() { impl_->pool_impl()->ReleaseHandler(impl_); }

int RunHandlerWorkQueue::GetParallelismLevel() const {
  return run_handler_->NumThreads();
}

void RunHandlerWorkQueue::AddTask(TaskFunction work) {
  run_handler_->ScheduleInterOpClosure(tensorflow::tfrt_stub::WrapWork(
      run_handler_->step_id(), "inter", std::move(work)));
}

std::optional<TaskFunction> RunHandlerWorkQueue::AddBlockingTask(
    TaskFunction work, bool allow_queuing) {
  LOG_EVERY_N_SEC(ERROR, 10)
      << "RunHandlerWorkQueue::AddBlockingTask() is not supposed to be called.";
  return work;
}

void RunHandlerWorkQueue::Await(ArrayRef<RCReference<AsyncValue>> values) {
  tfrt::Await(values);
}

bool RunHandlerWorkQueue::IsInWorkerThread() const {
  // Simply return true here as this method is not used in savedmodel workflow
  // and soon deprecated.
  //
  // TODO(b/198671794): Remove this method once it is removed from base.
  return true;
}

}  // namespace tf
}  // namespace tfrt
