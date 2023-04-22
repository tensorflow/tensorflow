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
#include <atomic>              // NOLINT
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Notification {
 public:
  Notification() : notified_(0) {}
  ~Notification() {
    // In case the notification is being used to synchronize its own deletion,
    // force any prior notifier to leave its critical section before the object
    // is destroyed.
    mutex_lock l(mu_);
  }

  void Notify() {
    mutex_lock l(mu_);
    assert(!HasBeenNotified());
    notified_.store(true, std::memory_order_release);
    cv_.notify_all();
  }

  bool HasBeenNotified() const {
    return notified_.load(std::memory_order_acquire);
  }

  void WaitForNotification() {
    if (!HasBeenNotified()) {
      mutex_lock l(mu_);
      while (!HasBeenNotified()) {
        cv_.wait(l);
      }
    }
  }

 private:
  friend bool WaitForNotificationWithTimeout(Notification* n,
                                             int64 timeout_in_us);
  bool WaitForNotificationWithTimeout(int64 timeout_in_us) {
    bool notified = HasBeenNotified();
    if (!notified) {
      mutex_lock l(mu_);
      do {
        notified = HasBeenNotified();
      } while (!notified &&
               cv_.wait_for(l, std::chrono::microseconds(timeout_in_us)) !=
                   std::cv_status::timeout);
    }
    return notified;
  }

  mutex mu_;                    // protects mutations of notified_
  condition_variable cv_;       // signaled when notified_ becomes non-zero
  std::atomic<bool> notified_;  // mutations under mu_
};

inline bool WaitForNotificationWithTimeout(Notification* n,
                                           int64 timeout_in_us) {
  return n->WaitForNotificationWithTimeout(timeout_in_us);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_DEFAULT_NOTIFICATION_H_
