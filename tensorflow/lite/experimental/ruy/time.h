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

namespace ruy {

using Clock = std::chrono::steady_clock;

using TimePoint = Clock::time_point;
using Duration = Clock::duration;

inline double ToSeconds(Duration d) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
}

inline Duration DurationFromSeconds(double s) {
  return std::chrono::duration_cast<Duration>(std::chrono::duration<double>(s));
}

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_TIME_H_
