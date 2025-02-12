/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_NOTIFICATION_H_
#define TENSORFLOW_TSL_PLATFORM_NOTIFICATION_H_

#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT
#include <cstdint>
#include <mutex>  // NOLINT

#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

namespace tsl {

using absl::Notification;

// TODO(ddunleavy): remove this method and replace uses of `tsl::Notification`
// with `absl::Notification`.
inline bool WaitForNotificationWithTimeout(Notification* n,
                                           int64_t timeout_in_us) {
  return n->WaitForNotificationWithTimeout(absl::Microseconds(timeout_in_us));
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_NOTIFICATION_H_
