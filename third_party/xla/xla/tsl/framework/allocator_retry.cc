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

#include "xla/tsl/framework/allocator_retry.h"

#include <cstddef>
#include <functional>
#include <optional>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "xla/tsl/framework/metrics.h"
#include "tsl/platform/env.h"
#include "tsl/platform/types.h"

namespace tsl {

namespace {
class ScopedTimeTracker {
 public:
  explicit ScopedTimeTracker(Env* env) : env_(env) {}
  void Enable() {
    if (!start_us_) {  // Only override start_us when not set yet.
      start_us_ = env_->NowMicros();
    }
  }
  ~ScopedTimeTracker() {
    if (start_us_) {
      uint64 end_us = env_->NowMicros();
      metrics::UpdateBfcAllocatorDelayTime(end_us - *start_us_);
    }
  }

 private:
  Env* env_;
  std::optional<uint64> start_us_;
};
}  // namespace

AllocatorRetry::AllocatorRetry() : env_(Env::Default()) {}

AllocatorRetry::~AllocatorRetry() {
  // Lock the mutex to make sure that all memory effects are safely published
  // and available to a thread running the destructor.
  absl::MutexLock l(&mu_);
}

void* AllocatorRetry::AllocateRaw(
    std::function<void*(size_t alignment, size_t num_bytes,
                        bool verbose_failure)>
        alloc_func,
    int max_millis_to_wait, size_t alignment, size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }
  ScopedTimeTracker tracker(env_);
  uint64 deadline_micros = 0;
  bool first = true;
  void* ptr = nullptr;
  while (ptr == nullptr) {
    ptr = alloc_func(alignment, num_bytes, false);
    if (ptr == nullptr) {
      uint64 now = env_->NowMicros();
      if (first) {
        deadline_micros = now + max_millis_to_wait * 1000;
        first = false;
      }
      if (now < deadline_micros) {
        tracker.Enable();
        absl::MutexLock l(&mu_);
        memory_returned_.WaitWithTimeout(
            &mu_, absl::Microseconds(deadline_micros - now));
      } else {
        return alloc_func(alignment, num_bytes, true);
      }
    }
  }
  return ptr;
}

}  // namespace tsl
