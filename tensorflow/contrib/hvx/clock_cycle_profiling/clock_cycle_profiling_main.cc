/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/profile_utils/clock_cycle_profiler.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"

int main(int argc, char** argv) {
  static constexpr int LOOP_COUNT = 1000000;

#if defined(__ANDROID_API__)
#if defined(__aarch64__)
  LOG(INFO) << "android arm 64 bit";
#endif
#if defined(__ARM_ARCH_7A__)
  LOG(INFO) << "android arm 32 bit";
#endif
  LOG(INFO) << "Android API = " << __ANDROID_API__;
  if (__ANDROID_API__ < 21) {
    LOG(INFO) << "Cpu utils requires API level 21 or above.";
    return 0;
  }
#endif

  tensorflow::profile_utils::CpuUtils::EnableClockCycleProfiling(true);

  tensorflow::ClockCycleProfiler prof_global;
  tensorflow::ClockCycleProfiler prof_internal;

  prof_global.Start();
  for (int i = 0; i < LOOP_COUNT; ++i) {
    prof_internal.Start();
    prof_internal.Stop();
  }
  prof_global.Stop();

  prof_global.DumpStatistics("prof_global");
  prof_internal.DumpStatistics("prof_internal");

  return 0;
}
