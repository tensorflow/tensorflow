/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_THREAD_POOL_TASK_RUNNER_H_
#define XLA_BACKENDS_CPU_RUNTIME_THREAD_POOL_TASK_RUNNER_H_

#define EIGEN_USE_THREADS

#include <cstdint>
#include <optional>
#include <utility>

#include "unsupported/Eigen/CXX11/ThreadPool"
#include "xla/backends/cpu/runtime/thunk.h"

namespace xla::cpu {

// An implementation of a `Thunk::TaskRunner` that uses Eigen thread pool for
// launching ThunkExecutor tasks. In XLA in practice it means that we run
// all ThunkExecutor tasks in the intra-op thread pool (owned by PjRt client).
class ThreadPoolTaskRunner : public Thunk::TaskRunner {
 public:
  explicit ThreadPoolTaskRunner(Eigen::ThreadPoolInterface* thread_pool)
      : thread_pool_(thread_pool) {}

  void operator()(Thunk::Task task) final {
    if (thread_pool_ == nullptr) {
      task();
    } else {
      thread_pool_->Schedule(std::move(task));
    }
  }

  std::optional<int64_t> current_worker_id() const final {
    if (thread_pool_ == nullptr) {
      return {0};
    } else {
      int64_t thread_id = thread_pool_->CurrentThreadId();
      return thread_id == -1 ? std::nullopt : std::make_optional(thread_id);
    }
  }

 private:
  Eigen::ThreadPoolInterface* thread_pool_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THREAD_POOL_TASK_RUNNER_H_
