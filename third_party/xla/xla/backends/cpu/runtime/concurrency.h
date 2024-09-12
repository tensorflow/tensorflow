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

#ifndef XLA_BACKENDS_CPU_RUNTIME_CONCURRENCY_H_
#define XLA_BACKENDS_CPU_RUNTIME_CONCURRENCY_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include "tsl/platform/logging.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"

namespace xla::cpu {

// A collection of concurrency primitives for use in the XLA CPU runtime.

// Schedules `n` tasks on the `intra_op_threadpool`, calling `F` for each index
// in the [0, n) range. Returns immediately after scheduling all tasks. It's a
// caller's responsibility to wait for all tasks to finish.
template <typename F,
          std::enable_if_t<std::is_invocable_v<F, int64_t>>* = nullptr>
void ScheduleAll(const Eigen::ThreadPoolDevice* intra_op_threadpool, int64_t n,
                 F&& f) {
  DCHECK(n >= 0) << "n must be non-negative";

  // Short-circuit the case of no tasks.
  if (n == 0) return;

  // Short-circuit the case of a single task.
  if (n == 1) {
    f(0);
    return;
  }

  // Heap-allocated state that manages concurrent execution of `f`.
  struct State {
    State(const Eigen::ThreadPoolDevice* intra_op_threadpool, F&& f)
        : intra_op_threadpool(intra_op_threadpool), f(std::forward<F>(f)) {}

    void Execute(std::shared_ptr<State> self, int64_t start, int64_t end) {
      while (end - start > 1) {
        uint64_t mid = (start + end) / 2;
        intra_op_threadpool->getPool()->Schedule(
            std::bind(&State::Execute, this, self, mid, end));
        end = mid;
      }
      f(start);
    }

    const Eigen::ThreadPoolDevice* intra_op_threadpool;
    F f;
  };

  auto s = std::make_shared<State>(intra_op_threadpool, std::forward<F>(f));
  s->Execute(std::move(s), 0, n);
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_CONCURRENCY_H_
