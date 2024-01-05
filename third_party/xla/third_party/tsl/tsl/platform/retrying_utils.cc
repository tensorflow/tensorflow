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
#include "tsl/platform/retrying_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "absl/time/time.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/random.h"

namespace tsl {

namespace {

bool IsRetriable(absl::StatusCode code) {
  switch (code) {
    case absl::StatusCode::kUnavailable:
    case absl::StatusCode::kDeadlineExceeded:
    case absl::StatusCode::kUnknown:
      return true;
    default:
      // OK also falls here.
      return false;
  }
}

double GenerateUniformRandomNumber() {
  return random::New64() * (1.0 / std::numeric_limits<uint64_t>::max());
}

double GenerateUniformRandomNumberBetween(double a, double b) {
  if (a == b) return a;
  DCHECK_LT(a, b);
  return a + GenerateUniformRandomNumber() * (b - a);
}

}  // namespace

Status RetryingUtils::CallWithRetries(const std::function<Status()>& f,
                                      const RetryConfig& config) {
  return CallWithRetries(
      f,
      [](int64_t micros) {
        return Env::Default()->SleepForMicroseconds(micros);
      },
      config);
}

Status RetryingUtils::CallWithRetries(
    const std::function<Status()>& f,
    const std::function<void(int64_t)>& sleep_usec, const RetryConfig& config) {
  int retries = 0;
  while (true) {
    auto status = f();
    if (!IsRetriable(status.code())) {
      return status;
    }
    if (retries >= config.max_retries) {
      // Return AbortedError, so that it doesn't get retried again somewhere
      // at a higher level.
      return Status(
          absl::StatusCode::kAborted,
          strings::StrCat(
              "All ", config.max_retries,
              " retry attempts failed. The last failure: ", status.message()));
    }
    int64_t delay_micros = 0;
    if (config.init_delay_time_us > 0) {
      const int64_t random_micros = random::New64() % 1000000;
      delay_micros = std::min(config.init_delay_time_us << retries,
                              config.max_delay_time_us) +
                     random_micros;
    }
    VLOG(1) << "The operation failed and will be automatically retried in "
            << (delay_micros / 1000000.0) << " seconds (attempt "
            << (retries + 1) << " out of " << config.max_retries
            << "), caused by: " << status.ToString();
    sleep_usec(delay_micros);
    retries++;
  }
}

Status RetryingUtils::DeleteWithRetries(
    const std::function<Status()>& delete_func, const RetryConfig& config) {
  bool is_retried = false;
  return RetryingUtils::CallWithRetries(
      [delete_func, &is_retried]() {
        const Status status = delete_func();
        if (is_retried && status.code() == error::NOT_FOUND) {
          return OkStatus();
        }
        is_retried = true;
        return status;
      },
      config);
}

absl::Duration ComputeRetryBackoff(int current_retry_attempt,
                                   absl::Duration min_delay,
                                   absl::Duration max_delay) {
  DCHECK_GE(current_retry_attempt, 0);

  // This function with the constants below is calculating:
  //
  // (0.4 * min_delay) + (random[0.6,1.0] * min_delay * 1.3^retries)
  //
  // Note that there is an extra truncation that occurs and is documented in
  // comments below.
  constexpr double kBackoffBase = 1.3;
  constexpr double kBackoffRandMult = 0.4;

  // This first term does not vary with current_retry_attempt or a random
  // number. It exists to ensure the final term is >= min_delay.
  const absl::Duration first_term = min_delay * kBackoffRandMult;

  // This is calculating min_delay * 1.3^retries.
  absl::Duration uncapped_second_term =
      min_delay * std::pow(kBackoffBase, current_retry_attempt);

  // Note that first_term + uncapped_second_term can exceed max_delay here
  // because of the final multiply by kBackoffBase.  We fix that problem with
  // the min() below.
  absl::Duration second_term =
      std::min(uncapped_second_term, max_delay - first_term);

  // This supplies the random jitter to ensure that retried don't cause a
  // thundering herd problem.
  second_term *=
      GenerateUniformRandomNumberBetween(1.0 - kBackoffRandMult, 1.0);

  return std::max(first_term + second_term, min_delay);
}

}  // namespace tsl
