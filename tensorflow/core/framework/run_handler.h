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

#ifndef TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_
#define TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace Eigen {
struct ThreadPoolDevice;
}

namespace tensorflow {

class RunHandler;

// RunHandlerPool is a fixed size pool of pre-allocated RunHandlers
// that can be used for tracking inter-op work for a given Session::Run().
// RunHandler(s) in the pool are initially 'inactive'. A RunHandler becomes
// 'active' when its unique_ptr is returned by Get() and is being used by a
// client. It becomes 'inactive' once more when its unique_ptr gets destroyed.
//
// Expected usage:
//
// * Create a single RunHandlerPool (say run_handler_pool_).
//
// * When a Session::Run() is invoked, obtain a handler by:
// auto handler = run_handler_pool_->Get();
//
// * Use handler for scheduling all inter-op work by:
// handler->ScheduleInterOpClosure(closure);
//
// This class is thread safe.
class RunHandlerPool {
 public:
  explicit RunHandlerPool(int num_inter_op_threads);

  RunHandlerPool(int num_inter_op_threads, int num_intra_op_threads);
  ~RunHandlerPool();

  // Returns an inactive RunHandler from the pool.
  //
  // RunHandlers in RunHandlerPool are initially 'inactive'.
  // A RunHandler becomes 'active' when its unique_ptr its returned by Get()
  // and is being used by a client.  It becomes 'inactive' once more when the
  // unique_ptr is destroyed.
  //
  // Will block unless there is an inactive handler.
  std::unique_ptr<RunHandler> Get(
      int64 step_id = 0, int64 timeout_in_ms = 0,
      const RunOptions::Experimental::RunHandlerPoolOptions& options =
          RunOptions::Experimental::RunHandlerPoolOptions());

  // Get the priorities for active handlers. The return result is with the same
  // order of the active handler list.
  std::vector<int64> GetActiveHandlerPrioritiesForTesting() const;

 private:
  class Impl;
  friend class RunHandler;

  std::unique_ptr<Impl> impl_;
};

// RunHandler can be used to schedule inter/intra-op closures to run on a global
// pool shared across all Session::Run(s). The closures are enqueued to a
// handler specific queue, from which the work is stolen in a priority order
// (time of the Get() call).
//
// It can only be created via RunHandlerPool::Get().
//
// This class can be used instead of directly scheduling closures on a global
// pool since it maintains a global view across all sessions and optimizes pool
// scheduling to improve (median and tail) latency.
//
// This class is thread safe.
class RunHandler {
 public:
  void ScheduleInterOpClosure(std::function<void()> fn);
  thread::ThreadPoolInterface* AsIntraThreadPoolInterface();

  ~RunHandler();

 private:
  class Impl;
  friend class RunHandlerPool::Impl;

  explicit RunHandler(Impl* impl);

  Impl* impl_;  // NOT OWNED.
};

namespace internal {

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
                        const string& name);

  EnvThread* CreateThread(std::function<void()> f);

  Task CreateTask(std::function<void()> f);

  void ExecuteTask(const Task& t);
};

typedef typename RunHandlerEnvironment::Task Task;
typedef Eigen::RunQueue<Task, 1024> Queue;

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

class ThreadWorkSource {
 public:
  ThreadWorkSource();

  ~ThreadWorkSource();

  Task EnqueueTask(Task t, bool is_blocking);

  Task PopBlockingTask();

  Task PopNonBlockingTask(int start_index, bool search_from_all_queue);

  void WaitForWork(int max_sleep_micros);

  int TaskQueueSize(bool is_blocking);

  int64 GetTracemeId();

  void SetTracemeId(int64 value);

  void SetWaiter(uint64 version, Waiter* waiter, mutex* mutex);

  int64 GetInflightTaskCount(bool is_blocking);

  void IncrementInflightTaskCount(bool is_blocking);

  void DecrementInflightTaskCount(bool is_blocking);

  unsigned NonBlockingWorkShardingFactor();

  std::string ToString();

 private:
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
  Waiter queue_waiters_ TF_GUARDED_BY(waiters_mu_);
  std::atomic<int64> traceme_id_;

  mutex run_handler_waiter_mu_;
  uint64 version_ TF_GUARDED_BY(run_handler_waiter_mu_);
  mutex* sub_thread_pool_waiter_mu_ TF_GUARDED_BY(run_handler_waiter_mu_);
  Waiter* sub_thread_pool_waiter_ TF_GUARDED_BY(run_handler_waiter_mu_);
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
                       const string& name,
                       Eigen::MaxSizeVector<mutex>* waiters_mu,
                       Eigen::MaxSizeVector<Waiter>* queue_waiters);

  ~RunHandlerThreadPool();

  void Start();

  void StartOneThreadForTesting();

  void AddWorkToQueue(ThreadWorkSource* tws, bool is_blocking,
                      std::function<void()> fn);

  // Set work queues from which the thread 'tid' can steal its work.
  // The request with start_request_idx will be attempted first. Other requests
  // will be attempted in FIFO order based on their arrival time.
  void SetThreadWorkSources(
      int tid, int start_request_idx, uint64 version,
      const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources);

  PerThread* GetPerThread();

  int CurrentThreadId() const;

  int NumThreads() const;

  int NumBlockingThreads() const;

  int NumNonBlockingThreads() const;

  void WorkerLoop(int thread_id, bool may_steal_blocking_work);

  // Search tasks from Requets range searching_range_start to
  // searching_range_end. If there is no tasks in the search range and
  // may_steal_blocking_work is true, then search from all requests.
  Task FindTask(
      int searching_range_start, int searching_range_end, int thread_id,
      int sub_thread_pool_id, int max_blocking_inflight,
      bool may_steal_blocking_work,
      const Eigen::MaxSizeVector<ThreadWorkSource*>& thread_work_sources,
      bool* task_from_blocking_queue, ThreadWorkSource** tws);

  void WaitForWork(bool is_blocking, int thread_id,
                   int32 max_blocking_inflight);

  void WaitForWorkInSubThreadPool(bool is_blocking, int sub_thread_pool_id);

 private:
  struct ThreadData {
    ThreadData();
    mutex mu;
    uint64 new_version;
    condition_variable sources_not_empty;
    std::unique_ptr<Thread> thread;
    int current_index;
    std::unique_ptr<Eigen::MaxSizeVector<ThreadWorkSource*>>
        new_thread_work_sources TF_GUARDED_BY(mu);

    uint64 current_version;
    // Should only be accessed by one thread.
    std::unique_ptr<Eigen::MaxSizeVector<ThreadWorkSource*>>
        current_thread_work_sources;

    int sub_thread_pool_id;
  };

  const int num_threads_;
  const int num_blocking_threads_;
  const int num_non_blocking_threads_;
  Eigen::MaxSizeVector<ThreadData> thread_data_;
  internal::RunHandlerEnvironment env_;
  std::atomic<bool> cancelled_;
  string name_;
  Eigen::MaxSizeVector<mutex>* waiters_mu_;
  Eigen::MaxSizeVector<Waiter>* queue_waiters_;

  bool use_sub_thread_pool_;
  std::vector<int> num_threads_in_sub_thread_pool_;

  // Threads in each sub thread pool will search tasks from the given
  // start_request_percentage to end_request_percentage in a round robin
  // fashion.
  std::vector<double> sub_thread_pool_start_request_percentage_;
  std::vector<double> sub_thread_pool_end_request_percentage_;
};

}  // namespace internal

}  // end namespace tensorflow.

#endif  // TENSORFLOW_CORE_FRAMEWORK_RUN_HANDLER_H_
