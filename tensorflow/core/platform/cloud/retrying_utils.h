/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_UTILS_H_

#include <functional>
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class RetryingUtils {
 public:
  /// \brief Retries the function in case of failure with exponential backoff.
  ///
  /// The provided callback is retried with an exponential backoff until it
  /// returns OK or a non-retriable error status.
  /// If initial_delay_microseconds is zero, no delays will be made between
  /// retries.
  /// If all retries failed, returns the last error status.
  static Status CallWithRetries(const std::function<Status()>& f,
                                const int64 initial_delay_microseconds);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_PLATFORM_CLOUD_RETRYING_UTILS_H_
