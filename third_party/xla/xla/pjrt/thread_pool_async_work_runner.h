/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_THREAD_POOL_ASYNC_WORK_RUNNER_H_
#define XLA_PJRT_THREAD_POOL_ASYNC_WORK_RUNNER_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/unbounded_work_queue.h"

namespace xla {

// AsyncWorkRunner backed by a fixed-size tsl::thread::ThreadPool.
class ThreadPoolAsyncWorkRunner : public AsyncWorkRunner {
 public:
  ThreadPoolAsyncWorkRunner(tsl::Env* env, absl::string_view name,
                            int num_threads,
                            const tsl::ThreadOptions& thread_options = {});

  void Execute(Task task) final;

  // Returns the underlying thread pool.
  tsl::thread::ThreadPool* thread_pool() { return &pool_; }

 private:
  tsl::thread::ThreadPool pool_;
};

// AsyncWorkRunner backed by tsl::UnboundedWorkQueue that grows as needed.
class UnboundedAsyncWorkRunner : public AsyncWorkRunner {
 public:
  explicit UnboundedAsyncWorkRunner(
      absl::string_view name, const tsl::ThreadOptions& thread_options = {});

  void Execute(Task task) final;

  // Returns the underlying unbounded work queue.
  tsl::UnboundedWorkQueue* unbounded_work_queue() { return &queue_; }

 private:
  tsl::UnboundedWorkQueue queue_;
};

std::unique_ptr<ThreadPoolAsyncWorkRunner> MakeThreadPoolAsyncWorkRunner(
    tsl::Env* env, absl::string_view name, int num_threads,
    const tsl::ThreadOptions& thread_options = {});

std::unique_ptr<UnboundedAsyncWorkRunner> MakeUnboundedAsyncWorkRunner(
    absl::string_view name, const tsl::ThreadOptions& thread_options = {});

}  // namespace xla

#endif  // XLA_PJRT_THREAD_POOL_ASYNC_WORK_RUNNER_H_
