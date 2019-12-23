/* Copyright 2019 Google LLC. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_

#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>  // IWYU pragma: keep
#include <ratio>    // NOLINT(build/c++11)

#ifdef __linux__
#include <sys/time.h>
// IWYU pragma: no_include <type_traits>

#include <ctime>
#endif

namespace ruy {

using InternalDefaultClock = std::chrono::steady_clock;

using TimePoint = InternalDefaultClock::time_point;
using Duration = InternalDefaultClock::duration;

template <typename RepresentationType>
Duration DurationFromSeconds(RepresentationType representation) {
  return std::chrono::duration_cast<Duration>(
      std::chrono::duration<RepresentationType>(representation));
}

template <typename RepresentationType>
Duration DurationFromMilliseconds(RepresentationType representation) {
  return std::chrono::duration_cast<Duration>(
      std::chrono::duration<RepresentationType, std::milli>(representation));
}

template <typename RepresentationType>
Duration DurationFromNanoseconds(RepresentationType representation) {
  return std::chrono::duration_cast<Duration>(
      std::chrono::duration<RepresentationType, std::nano>(representation));
}

inline float ToFloatSeconds(const Duration& duration) {
  return std::chrono::duration_cast<std::chrono::duration<float>>(duration)
      .count();
}

inline std::int64_t ToInt64Nanoseconds(const Duration& duration) {
  return std::chrono::duration_cast<
             std::chrono::duration<std::int64_t, std::nano>>(duration)
      .count();
}

inline TimePoint Now() { return InternalDefaultClock::now(); }

inline TimePoint CoarseNow() {
#ifdef __linux__
  timespec t;
  clock_gettime(CLOCK_MONOTONIC_COARSE, &t);
  return TimePoint(
      DurationFromNanoseconds(1000000000LL * t.tv_sec + t.tv_nsec));
#else
  return Now();
#endif
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_
