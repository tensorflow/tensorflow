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

#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

namespace tensorflow {
namespace profile_utils {

namespace {

const class StaticVariableInitializer {
 public:
  StaticVariableInitializer() { CpuUtils::Initialize(); }
} STATIC_VARIABLE_INITIALIZER;

}  // anonymous namespace for initializer

/* static */ constexpr int64 CpuUtils::INVALID_FREQUENCY;

/* static */ int64 CpuUtils::GetCpuFrequency() {
  static const int64 cpu_frequency = GetCpuFrequencyImpl();
  return cpu_frequency;
}

/* static */ int CpuUtils::GetClockPerMicroSec() {
  static const int clock_per_micro_sec =
      static_cast<int>(GetCpuFrequency() / (1000LL * 1000LL));
  return clock_per_micro_sec;
}

/* static */ double CpuUtils::GetMicroSecPerClock() {
  static const double micro_sec_per_clock =
      (1000.0 * 1000.0) / static_cast<double>(GetCpuFrequency());
  return micro_sec_per_clock;
}

/* static */ void CpuUtils::Initialize() {
  CpuUtils::GetCpuFrequency();
  CpuUtils::GetClockPerMicroSec();
  CpuUtils::GetMicroSecPerClock();
  GetCpuUtilsHelper().Initialize();
}

/* static */ void CpuUtils::ResetClockCycle() {
  GetCpuUtilsHelper().ResetClockCycle();
}

/* static */ void CpuUtils::EnableClockCycleProfiling(const bool enable) {
  GetCpuUtilsHelper().EnableClockCycleProfiling(enable);
}

/* static */ int64 CpuUtils::GetCpuFrequencyImpl() {
// TODO(satok): do not switch by macro here
#if defined(__ANDROID__)
  // TODO:(satok): Support Android
  return INVALID_FREQUENCY;
#elif defined(__linux__)
  double bogomips;
  FILE* fp = popen("grep '^bogomips' /proc/cpuinfo | head -1", "r");
  const int retval_of_bogomips = fscanf(fp, "bogomips : %lf", &bogomips);
  pclose(fp);
  const double freq_ghz = bogomips / 1000.0 / 2.0;
  if (retval_of_bogomips != 1 || freq_ghz < 0.01) {
    LOG(WARNING) << "Failed to get CPU frequency: " << freq_ghz << " Hz";
    return INVALID_FREQUENCY;
  }
  return static_cast<int64>(freq_ghz * 1000.0 * 1000.0 * 1000.0);
#elif defined(__APPLE__)
  int64 freq_hz;
  FILE* fp =
      popen("sysctl hw | grep hw.cpufrequency_max: | cut -d' ' -f 2", "r");
  fscanf(fp, "%lld", &freq_hz);
  pclose(fp);
  if (freq_hz < 1e6) {
    LOG(WARNING) << "Failed to get CPU frequency: " << freq_hz << " Hz";
    return INVALID_FREQUENCY;
  }
  return freq_hz;
#else
  // TODO(satok): Support other OS if needed
  // Return INVALID_FREQUENCY on unsupported OS
  return INVALID_FREQUENCY;
#endif
}

/* static */ ICpuUtilsHelper& CpuUtils::GetCpuUtilsHelper() {
#if defined(__ANDROID__) && defined(__ARM_ARCH_7A__) && (__ANDROID_API__ >= 21)
  static AndroidArmV7ACpuUtilsHelper cpu_utils_helper;
#else
  // TODO(satok): Change CpuUtilsHelper by cpu architecture
  static DefaultCpuUtilsHelper cpu_utils_helper;
#endif
  return cpu_utils_helper;
}

}  // namespace profile_utils
}  // namespace tensorflow
