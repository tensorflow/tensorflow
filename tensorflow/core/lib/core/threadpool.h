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

#ifndef TENSORFLOW_LIB_CORE_THREADPOOL_H_
#define TENSORFLOW_LIB_CORE_THREADPOOL_H_

#include <functional>
#include <memory>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace thread {

class ThreadPool {
 public:
  // Constructs a pool that contains "num_threads" threads with specified
  // "name". env->StartThread() is used to create individual threads with the
  // given ThreadOptions. If "low_latency_hint" is true the thread pool
  // implementation may use it as a hint that lower latency is preferred at the
  // cost of higher CPU usage, e.g. by letting one or more idle threads spin
  // wait. Conversely, if the threadpool is used to schedule high-latency
  // operations like I/O the hint should be set to false.
  //
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options, const string& name,
             int num_threads, bool low_latency_hint);

  // Constructs a pool for low-latency ops that contains "num_threads" threads
  // with specified "name". env->StartThread() is used to create individual
  // threads.
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const string& name, int num_threads);

  ThreadPool(Env* env, const string& name, int num_threads, std::vector<int> &proc_set);

  // Constructs a pool for low-latency ops that contains "num_threads" threads
  // with specified "name". env->StartThread() is used to create individual
  // threads with the given ThreadOptions.
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options, const string& name,
             int num_threads);

  // Waits until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPool();

  // Schedules fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn);

  // ParallelFor shards the "total" units of work assuming each unit of work
  // having roughly "cost_per_unit" cost, in cycles. Each unit of work is
  // indexed 0, 1, ..., total - 1. Each shard contains 1 or more units of work
  // and the total cost of each shard is roughly the same.
  //
  // "cost_per_unit" is an estimate of the number of CPU cycles (or nanoseconds
  // if not CPU-bound) to complete a unit of work. Overestimating creates too
  // many shards and CPU time will be dominated by per-shard overhead, such as
  // Context creation. Underestimating may not fully make use of the specified
  // parallelism.
  void ParallelFor(int64 total, int64 cost_per_unit,
                   std::function<void(int64, int64)> fn);

  // Shards the "total" units of work. For more details, see "ParallelFor".
  //
  // The function is passed a thread_id between 0 and NumThreads() *inclusive*.
  // This is because some work can happen on the caller thread while the threads
  // in the pool are also being used.
  //
  // The caller can allocate NumThreads() + 1 separate buffers for each thread.
  // Each thread can safely write to the buffer given by its id without
  // synchronization. However, the worker fn may be called multiple times
  // sequentially with the same id.
  //
  // At most NumThreads() unique ids will actually be used, and only a few may
  // be used for small workloads. If each buffer is expensive, the buffers
  // should be stored in an array initially filled with null, and a buffer
  // should be allocated by fn the first time that the id is used.
  void ParallelForWithWorkerId(
      int64 total, int64 cost_per_unit,
      const std::function<void(int64, int64, int)>& fn);

  // Returns the number of threads in the pool.
  int NumThreads() const;

  // Returns current thread id between 0 and NumThreads() - 1, if called from a
  // thread in the pool. Returns -1 otherwise.
  int CurrentThreadId() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
  TF_DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_THREADPOOL_H_
