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

#include "xla/tsl/platform/env_time.h"

#include <time.h>
#include <windows.h>

#include <chrono>

using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::system_clock;

namespace tsl {

namespace {
typedef VOID(WINAPI* FnGetSystemTimePreciseAsFileTime)(LPFILETIME);
}

uint64 EnvTime::NowNanos() {
  static FnGetSystemTimePreciseAsFileTime precise_time_function =
      []() -> FnGetSystemTimePreciseAsFileTime {
    HMODULE module = GetModuleHandleW(L"kernel32.dll");
    if (module != NULL) {
      return (FnGetSystemTimePreciseAsFileTime)GetProcAddress(
          module, "GetSystemTimePreciseAsFileTime");
    } else {
      return NULL;
    }
  }();

  if (precise_time_function != NULL) {
    // GetSystemTimePreciseAsFileTime function is only available in latest
    // versions of Windows, so we need to check for its existence here.
    // All std::chrono clocks on Windows proved to return values that may
    // repeat, which is not good enough for some uses.
    constexpr int64_t kUnixEpochStartTicks = 116444736000000000LL;

    // This interface needs to return system time and not just any time
    // because it is often used as an argument to TimedWait() on condition
    // variable.
    FILETIME system_time;
    precise_time_function(&system_time);

    LARGE_INTEGER li;
    li.LowPart = system_time.dwLowDateTime;
    li.HighPart = system_time.dwHighDateTime;
    // Subtract unix epoch start
    li.QuadPart -= kUnixEpochStartTicks;

    constexpr int64_t kFtToNanoSec = 100;
    li.QuadPart *= kFtToNanoSec;
    return li.QuadPart;
  }
  return duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
      .count();
}

}  // namespace tsl
