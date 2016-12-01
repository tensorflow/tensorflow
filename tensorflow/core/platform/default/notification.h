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

#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_NOTIFICATION_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_NOTIFICATION_H_

#include <assert.h>
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Notification {
 public:
  Notification() : notified_(false) {}
  ~Notification() {}

  void Notify() {
    mutex_lock l(mu_);
    assert(!notified_);
    notified_ = true;
    cv_.notify_all();
  }

  bool HasBeenNotified() {
    mutex_lock l(mu_);
    return notified_;
  }

  void WaitForNotification() {
    mutex_lock l(mu_);
    while (!notified_) {
      cv_.wait(l);
    }
  }

 private:
  friend bool WaitForNotificationWithTimeout(Notification* n,
                                             int64 timeout_in_ms);
  bool WaitForNotificationWithTimeout(int64 timeout_in_ms) {
    mutex_lock l(mu_);
    return cv_.wait_for(l, std::chrono::milliseconds(timeout_in_ms),
                        [this]() { return notified_; });
  }

  mutex mu_;
  condition_variable cv_;
  bool notified_;
};

inline bool WaitForNotificationWithTimeout(Notification* n,
                                           int64 timeout_in_ms) {
  return n->WaitForNotificationWithTimeout(timeout_in_ms);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_NOTIFICATION_H_
