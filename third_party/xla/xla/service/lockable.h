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

#ifndef XLA_SERVICE_LOCKABLE_H_
#define XLA_SERVICE_LOCKABLE_H_

#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/logging.h"

namespace xla {

// An RAII helper for a value of type `T` that requires exclusive access.
template <typename T>
class Lockable {
 public:
  // RAII type that will release the exclusive lock when it is destroyed.
  using Lock = std::unique_ptr<T, std::function<void(T*)>>;

  Lockable() = default;
  explicit Lockable(T value) : value_(std::move(value)) {}

  Lockable(const Lockable&) = delete;
  Lockable& operator=(const Lockable&) = delete;

  ~Lockable() {
    absl::MutexLock lock(&mutex_);
    CHECK_EQ(is_unlocked_, true);  // NOLINT
  }

  Lock Acquire() {
    absl::MutexLock lock(&mutex_);
    mutex_.Await(absl::Condition(&is_unlocked_));
    is_unlocked_ = false;

    return {&value_, [this](T*) {
              absl::MutexLock lock(&mutex_);
              CHECK(!is_unlocked_);  // NOLINT
              is_unlocked_ = true;
            }};
  }

 private:
  T value_;
  absl::Mutex mutex_;
  bool is_unlocked_ ABSL_GUARDED_BY(mutex_) = true;
};

}  // namespace xla

#endif  // XLA_SERVICE_LOCKABLE_H_
