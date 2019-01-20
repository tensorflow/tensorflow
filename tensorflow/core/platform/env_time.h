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
  static constexpr uint64 kMicrosToNanos = 1000ULL;
  static constexpr uint64 kMillisToMicros = 1000ULL;
  static constexpr uint64 kMillisToNanos = 1000ULL * 1000ULL;
  static constexpr uint64 kSecondsToMillis = 1000ULL;
  static constexpr uint64 kSecondsToMicros = 1000ULL * 1000ULL;
  static constexpr uint64 kSecondsToNanos = 1000ULL * 1000ULL * 1000ULL;

  EnvTime();
  virtual ~EnvTime() = default;

  /// \brief Returns a default impl suitable for the current operating
  /// system.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static EnvTime* Default();

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  virtual uint64 NowNanos() = 0;

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64 NowMicros() { return NowNanos() / kMicrosToNanos; }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64 NowSeconds() { return NowNanos() / kSecondsToNanos; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_ENV_TIME_H_
