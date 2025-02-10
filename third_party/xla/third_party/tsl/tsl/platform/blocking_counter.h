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

#ifndef TENSORFLOW_TSL_PLATFORM_BLOCKING_COUNTER_H_
#define TENSORFLOW_TSL_PLATFORM_BLOCKING_COUNTER_H_

#include <atomic>

#include "xla/tsl/platform/logging.h"
#include "tsl/platform/mutex.h"

namespace tsl {

class BlockingCounter {
 public:
  BlockingCounter(int initial_count)
      : state_(initial_count << 1), notified_(false) {
    CHECK_GE(initial_count, 0);
    DCHECK_EQ((initial_count << 1) >> 1, initial_count);
  }

  ~BlockingCounter() {}

  inline void DecrementCount() {
    unsigned int v = state_.fetch_sub(2, std::memory_order_acq_rel) - 2;
    if (v != 1) {
      DCHECK_NE(((v + 2) & ~1), 0);
      return;  // either count has not dropped to 0, or waiter is not waiting
    }
    mutex_lock l(mu_);
    DCHECK(!notified_);
    notified_ = true;
    cond_var_.notify_all();
  }

  inline void Wait() {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return;
    mutex_lock l(mu_);
    while (!notified_) {
      cond_var_.wait(l);
    }
  }
  // Wait for the specified time, return false iff the count has not dropped to
  // zero before the timeout expired.
  inline bool WaitFor(std::chrono::milliseconds ms) {
    unsigned int v = state_.fetch_or(1, std::memory_order_acq_rel);
    if ((v >> 1) == 0) return true;
    mutex_lock l(mu_);
    while (!notified_) {
      const std::cv_status status = cond_var_.wait_for(l, ms);
      if (status == std::cv_status::timeout) {
        return false;
      }
    }
    return true;
  }

 private:
  mutex mu_;
  condition_variable cond_var_;
  std::atomic<int> state_;  // low bit is waiter flag
  bool notified_;
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_BLOCKING_COUNTER_H_
