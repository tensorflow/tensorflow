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

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_TEST_UTILS_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_TEST_UTILS_H_

#include <deque>
#include <functional>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace xla {
namespace ifrt {
namespace proxy {

// TestQueue implements a thread-safe queue that manages values of type T.
template <typename T>
class TestQueue {
 public:
  explicit TestQueue(absl::Duration pop_timeout)
      : pop_timeout_(std::move(pop_timeout)) {}

  // Pushes `t` into the queue.
  void Push(T t) {
    absl::MutexLock l(&mu_);
    queue_.push_back(std::move(t));
  }

  // Pops the first element in the queue if a element is already available or
  // appears within `pop_timeout` (because `Push` is called). Otherwise returns
  // std::nullopt.
  std::optional<T> PopOrTimeout() {
    absl::MutexLock l(&mu_);
    auto cond = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) -> bool {
      return !queue_.empty();
    };
    mu_.AwaitWithTimeout(absl::Condition(&cond), pop_timeout_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    T result = std::move(queue_.front());
    queue_.pop_front();
    return result;
  }

  // Pops the first element in the queue if a element is already available or
  // appears within `pop_timeout`, and fails otherwise.
  T Pop() {
    std::optional<T> result = PopOrTimeout();
    CHECK(result.has_value()) << "Timeout!";
    return std::move(*result);
  }

  // Sets whether the queue is allowed to be destructed while it contains
  // unpopped elements.
  void AllowNonEmptyDestruction(bool allow) {
    absl::MutexLock l(&mu_);
    allow_non_empty_destruction_ = allow;
  }

  // Checks that the queue is either empty, or `AllowNonEmptyDestruction(true)`
  // has been called.
  ~TestQueue() {
    absl::MutexLock l(&mu_);
    if (!allow_non_empty_destruction_) CHECK(queue_.empty()) << " " << this;
  }

 private:
  const absl::Duration pop_timeout_;

  absl::Mutex mu_;
  std::deque<T> queue_ ABSL_GUARDED_BY(mu_);
  bool allow_non_empty_destruction_ ABSL_GUARDED_BY(mu_) = false;
};

// TestHook provides a lightweight mechanism to modify the behavior of
// production code from tests.
// TODO(b/266635130): Extend for more hook types (as of Sep 2023, only allows
// `void(bool*)`) and make more lightweight.
enum class TestHookName {
  kRpcBatcherPausePeriodicFlush,
};

// Allows test code to override the default noop behavior for hook `h`.
void TestHookSet(TestHookName h, std::function<void(bool*)> fn);

// Resets hook `h` to the default noop behavior.
void TestHookClear(TestHookName h);

// Calls hook `h` if it has been overridden by test setup; noop otherwise.
void TestHookCall(TestHookName h, bool* param1);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_TEST_UTILS_H_
