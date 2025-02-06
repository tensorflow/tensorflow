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

#include "xla/tsl/platform/cloud/time_util.h"

#include <time.h>

#include <cmath>
#include <cstdio>
#include <ctime>
#ifdef _WIN32
#define timegm _mkgmtime
#endif
#include "xla/tsl/platform/errors.h"

namespace tsl {

namespace {
constexpr int64_t kNanosecondsPerSecond = 1000 * 1000 * 1000;

}  // namespace

// Only implements one special case of RFC 3339 which is returned by
// GCS API, e.g 2016-04-29T23:15:24.896Z.
absl::Status ParseRfc3339Time(const string& time, int64_t* mtime_nsec) {
  tm parsed{0};
  float seconds;
  if (sscanf(time.c_str(), "%4d-%2d-%2dT%2d:%2d:%fZ", &(parsed.tm_year),
             &(parsed.tm_mon), &(parsed.tm_mday), &(parsed.tm_hour),
             &(parsed.tm_min), &seconds) != 6) {
    return errors::Internal(
        strings::StrCat("Unrecognized RFC 3339 time format: ", time));
  }
  const int int_seconds = std::floor(seconds);
  parsed.tm_year -= 1900;  // tm_year expects years since 1900.
  parsed.tm_mon -= 1;      // month is zero-based.
  parsed.tm_sec = int_seconds;

  *mtime_nsec = timegm(&parsed) * kNanosecondsPerSecond +
                static_cast<int64_t>(std::floor((seconds - int_seconds) *
                                                kNanosecondsPerSecond));

  return absl::OkStatus();
}

}  // namespace tsl
