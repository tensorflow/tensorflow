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

#include <stdlib.h>
#include <atomic>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/threadpool_impl.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace thread {

// Hook for unit tests.
bool use_nonblocking_pool;

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads) {
  const char* type = getenv("TF_THREAD_POOL");
  if (!use_nonblocking_pool && (type == nullptr || string(type) == "default")) {
    impl_.reset(
        CreateDefaultThreadPool(env, thread_options, name, num_threads));
  } else if (use_nonblocking_pool || string(type) == "nonblocking") {
    static std::atomic_flag warned = ATOMIC_FLAG_INIT;
    if (!warned.test_and_set())
      LOG(INFO) << "Using experimental nonblocking thread pool implementation";
    impl_.reset(
        CreateNonBlockingThreadPool(env, thread_options, name, num_threads));
  } else {
    LOG(FATAL)
        << "Unknown value for TF_THREAD_POOL environment variable: '" << type
        << "': accepted values are 'default' , 'nonblocking' (experimental)";
  }
}

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads) {}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  return impl_->Schedule(std::move(fn));
}

}  // namespace thread
}  // namespace tensorflow
