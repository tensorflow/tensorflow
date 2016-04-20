/* Copyright 2015 Google Inc. All Rights Reserved.

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

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
  TF_DISALLOW_COPY_AND_ASSIGN(ThreadPool);
};

}  // namespace thread
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_CORE_THREADPOOL_H_
