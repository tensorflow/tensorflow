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

#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_MUTEX_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_MUTEX_H_

// IWYU pragma: private, include "third_party/tensorflow/tsl/platform/mutex.h"
// IWYU pragma: friend third_party/tensorflow/tsl/platform/mutex.h

namespace tsl {

namespace internal {
std::cv_status wait_until_system_clock(
    CVData *cv_data, MuData *mu_data,
    const std::chrono::system_clock::time_point timeout_time);
}  // namespace internal

template <class Rep, class Period>
std::cv_status condition_variable::wait_for(
    mutex_lock &lock, std::chrono::duration<Rep, Period> dur) {
  return wait_until_system_clock(&this->cv_, &lock.mutex()->mu_,
                                 std::chrono::system_clock::now() + dur);
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_MUTEX_H_
