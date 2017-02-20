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

#include "tensorflow/core/platform/cloud/retrying_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

namespace {

// In case of failure, every call will be retried kMaxRetries times.
constexpr int kMaxRetries = 5;
// Maximum backoff time in microseconds.
constexpr int64 kMaximumBackoffMicroseconds = 32000000;  // 32 seconds.

bool IsRetriable(Status status) {
  switch (status.code()) {
    case error::UNAVAILABLE:
    case error::DEADLINE_EXCEEDED:
    case error::UNKNOWN:
      return true;
    default:
      // OK also falls here.
      return false;
  }
}

void WaitBeforeRetry(const int64 delay_micros) {
  const int64 random_micros = random::New64() % 1000000;
  Env::Default()->SleepForMicroseconds(std::min(delay_micros + random_micros,
                                                kMaximumBackoffMicroseconds));
}

}  // namespace

Status RetryingUtils::CallWithRetries(const std::function<Status()>& f,
                                      const int64 initial_delay_microseconds) {
  int retries = 0;
  while (true) {
    auto status = f();
    if (!IsRetriable(status) || retries >= kMaxRetries) {
      return status;
    }
    const int64 delay_micros = initial_delay_microseconds << retries;
    if (delay_micros > 0) {
      WaitBeforeRetry(delay_micros);
    }
    retries++;
  }
}

}  // namespace tensorflow
