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
#ifndef TENSORFLOW_CORE_PLATFORM_ENV_TIME_H_
#define TENSORFLOW_CORE_PLATFORM_ENV_TIME_H_

#include <stdint.h>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

/// \brief An interface used by the tensorflow implementation to
/// access timer related operations.
class EnvTime {
 public:
  static constexpr uint64 kMicrosToPicos = 1000ULL * 1000ULL;
  static constexpr uint64 kMicrosToNanos = 1000ULL;
  static constexpr uint64 kMillisToMicros = 1000ULL;
  static constexpr uint64 kMillisToNanos = 1000ULL * 1000ULL;
  static constexpr uint64 kNanosToPicos = 1000ULL;
  static constexpr uint64 kSecondsToMillis = 1000ULL;
  static constexpr uint64 kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64 kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime() = default;
  virtual ~EnvTime() = default;

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  static uint64 NowNanos();

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  static uint64 NowMicros() { return NowNanos() / kMicrosToNanos; }

  /// \brief Returns the number of seconds since the Unix epoch.
  static uint64 NowSeconds() { return NowNanos() / kSecondsToNanos; }

  /// \brief A version of NowNanos() that may be overridden by a subclass.
  virtual uint64 GetOverridableNowNanos() const { return NowNanos(); }

  /// \brief A version of NowMicros() that may be overridden by a subclass.
  virtual uint64 GetOverridableNowMicros() const {
    return GetOverridableNowNanos() / kMicrosToNanos;
  }

  /// \brief A version of NowSeconds() that may be overridden by a subclass.
  virtual uint64 GetOverridableNowSeconds() const {
    return GetOverridableNowNanos() / kSecondsToNanos;
  }

  /// \brief Returns the absolute elapsed wall-clock time in nano-seconds since
  /// some arbitrary and fixed point. Use it for time measurement or profiling.
  static uint64 MonotonicNanos();

  /// \brief Returns the absolute elapsed wall-clock time in micro-seconds since
  /// some arbitrary and fixed point. Use it for time measurement or profiling.
  static uint64 MonotonicMicros() { return MonotonicNanos() / kMicrosToNanos; }

  /// \brief Returns the absolute elapsed wall-clock time in seconds since some
  /// arbitrary and fixed point. Use it for time measurement or profiling.
  static uint64 MonotonicSeconds() {
    return MonotonicNanos() / kSecondsToNanos;
  }

  /// \brief A version of MonotonicNanos() that may be overridden by a subclass.
  virtual uint64 GetOverridableMonotonicNanos() const {
    return MonotonicNanos();
  }

  /// \brief A version of MonotonicMicros() that may be overridden by a
  /// subclass.
  virtual uint64 GetOverridableMonotonicMicros() const {
    return GetOverridableMonotonicNanos() / kMicrosToNanos;
  }

  /// \brief A version of MonotonicSeconds() that may be overridden by a
  /// subclass.
  virtual uint64 GetOverridableMonotonicSeconds() const {
    return GetOverridableMonotonicNanos() / kSecondsToNanos;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_ENV_TIME_H_
