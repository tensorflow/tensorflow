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
  // Construct a pool that contains "num_threads" threads with specified "name".
  // env->StartThread() is used to create individual threads.
  //
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const string& name, int num_threads);

  // Construct a pool that contains "num_threads" threads with specified "name".
  // env->StartThread() is used to create individual threads.
  //
  // REQUIRES: num_threads > 0
  ThreadPool(Env* env, const ThreadOptions& thread_options, const string& name,
             int num_threads);

  // Wait until all scheduled work has finished and then destroy the
  // set of threads.
  ~ThreadPool();

  // Schedule fn() for execution in the pool of threads.
  void Schedule(std::function<void()> fn);

  // ParallelFor shards the "total" unit of work assuming each unit of work
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
