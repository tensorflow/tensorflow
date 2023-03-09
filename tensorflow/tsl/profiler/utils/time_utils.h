/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_TIME_UTILS_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_TIME_UTILS_H_

#include <cstdint>

#include "tensorflow/tsl/profiler/utils/math_utils.h"

namespace tsl {
namespace profiler {

// Returns the current CPU wallclock time in nanoseconds.
int64_t GetCurrentTimeNanos();

// Sleeps for the specified duration.
void SleepForNanos(int64_t ns);
inline void SleepForMicros(int64_t us) { SleepForNanos(MicroToNano(us)); }
inline void SleepForMillis(int64_t ms) { SleepForNanos(MilliToNano(ms)); }
inline void SleepForSeconds(int64_t s) { SleepForNanos(UniToNano(s)); }

// Spins to simulate doing some work instead of sleeping, because sleep
// precision is poor. For testing only.
void SpinForNanos(int64_t ns);
inline void SpinForMicros(int64_t us) { SpinForNanos(us * 1000); }

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_TIME_UTILS_H_
