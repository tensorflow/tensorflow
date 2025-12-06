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
#ifndef XLA_TSL_PLATFORM_ENV_TIME_H_
#define XLA_TSL_PLATFORM_ENV_TIME_H_

#include <stdint.h>

#include "xla/tsl/platform/types.h"

namespace tsl {

/// \brief An interface used by the tsl implementation to
/// access timer related operations.
class EnvTime {
 public:
  static constexpr uint64_t kMicrosToPicos = 1000ULL * 1000ULL;
  static constexpr uint64_t kMicrosToNanos = 1000ULL;
  static constexpr uint64_t kMillisToMicros = 1000ULL;
  static constexpr uint64_t kMillisToNanos = 1000ULL * 1000ULL;
  static constexpr uint64_t kNanosToPicos = 1000ULL;
  static constexpr uint64_t kSecondsToMillis = 1000ULL;
  static constexpr uint64_t kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64_t kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() = default;
  virtual ~EnvTime() = default;

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  static uint64_t NowNanos();

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  static uint64_t NowMicros() { return NowNanos() / kMicrosToNanos; }

  /// \brief Returns the number of seconds since the Unix epoch.
  static uint64_t NowSeconds() { return NowNanos() / kSecondsToNanos; }

  /// \brief A version of NowNanos() that may be overridden by a subclass.
  virtual uint64_t GetOverridableNowNanos() const { return NowNanos(); }

  /// \brief A version of NowMicros() that may be overridden by a subclass.
  virtual uint64_t GetOverridableNowMicros() const {
    return GetOverridableNowNanos() / kMicrosToNanos;
  }

  /// \brief A version of NowSeconds() that may be overridden by a subclass.
  virtual uint64_t GetOverridableNowSeconds() const {
    return GetOverridableNowNanos() / kSecondsToNanos;
  }
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_ENV_TIME_H_
