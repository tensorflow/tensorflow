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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TIME_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TIME_UTILS_H_

#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

// Converts among different time units.
// NOTE: We use uint64 for picoseconds and nanoseconds, which are used in
// storage, and double for other units that are used in the UI.
inline double PicosToNanos(uint64 ps) { return ps / 1E3; }
inline double PicosToMicros(uint64 ps) { return ps / 1E6; }
inline double PicosToMillis(uint64 ps) { return ps / 1E9; }
inline double PicosToSeconds(uint64 ps) { return ps / 1E12; }
inline uint64 NanosToPicos(uint64 ns) { return ns * 1000; }
inline double NanosToMicros(uint64 ns) { return ns / 1E3; }
inline double MicrosToNanos(double us) { return us * 1E3; }
inline double MicrosToMillis(double us) { return us / 1E3; }
inline uint64 MillisToPicos(double ms) { return ms * 1E9; }
inline uint64 MillisToNanos(double ms) { return ms * 1E6; }
inline double MillisToSeconds(double ms) { return ms / 1E3; }
inline uint64 SecondsToNanos(double s) { return s * 1E9; }

// Returns the current CPU wallclock time in nanoseconds.
int64 GetCurrentTimeNanos();

// Sleeps for the specified duration.
void SleepForNanos(int64 ns);
inline void SleepForMicros(int64 us) { SleepForNanos(us * 1000); }
inline void SleepForMillis(int64 ms) { SleepForNanos(ms * 1000000); }
inline void SleepForSeconds(int64 s) { SleepForNanos(s * 1000000000); }

// Spins to simulate doing some work instead of sleeping, because sleep
// precision is poor. For testing only.
void SpinForNanos(int64 ns);
inline void SpinForMicros(int64 us) { SpinForNanos(us * 1000); }

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TIME_UTILS_H_
