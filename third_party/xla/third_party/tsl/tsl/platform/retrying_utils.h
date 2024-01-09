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
#ifndef TENSORFLOW_TSL_PLATFORM_RETRYING_UTILS_H_
#define TENSORFLOW_TSL_PLATFORM_RETRYING_UTILS_H_

#include <functional>

#include "absl/time/time.h"
#include "tsl/platform/status.h"

namespace tsl {

// Default time before reporting failure: ~100 seconds.
struct RetryConfig {
  RetryConfig(int64_t init_delay_time_us = 100 * 1000,
              int64_t max_delay_time_us = 32 * 1000 * 1000,
              int max_retries = 10) {
    this->init_delay_time_us = init_delay_time_us;
    this->max_delay_time_us = max_delay_time_us;
    this->max_retries = max_retries;
  }

  // In case of failure, every call will be retried max_retries times.
  int max_retries;

  // Initial backoff time
  int64_t init_delay_time_us;

  // Maximum backoff time in microseconds.
  int64_t max_delay_time_us;
};

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
                                const RetryConfig& config);

  /// sleep_usec is a function that sleeps for the given number of microseconds.
  static Status CallWithRetries(const std::function<Status()>& f,
                                const std::function<void(int64_t)>& sleep_usec,
                                const RetryConfig& config);
  /// \brief A retrying wrapper for a function that deletes a resource.
  ///
  /// The function takes care of the scenario when a delete operation
  /// returns a failure but succeeds under the hood: if a retry returns
  /// NOT_FOUND, the whole operation is considered a success.
  static Status DeleteWithRetries(const std::function<Status()>& delete_func,
                                  const RetryConfig& config);
};

// Given the total number of retries attempted, returns a randomized duration of
// time to delay before the next retry.
//
// The average computed backoff increases with the number of retries attempted.
// See implementation for details on the calculations.
absl::Duration ComputeRetryBackoff(
    int current_retry_attempt, absl::Duration min_delay = absl::Milliseconds(1),
    absl::Duration max_delay = absl::Seconds(10));

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_RETRYING_UTILS_H_
