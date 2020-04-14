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

#include <fstream>
#include <limits>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "absl/base/call_once.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

namespace tensorflow {
namespace profile_utils {

/* static */ constexpr int64 CpuUtils::INVALID_FREQUENCY;

static ICpuUtilsHelper* cpu_utils_helper_instance_ = nullptr;

#if (defined(__powerpc__) ||                                             \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) || \
    (defined(__s390x__))
/* static */ uint64 CpuUtils::GetCycleCounterFrequency() {
  static const uint64 cpu_frequency = GetCycleCounterFrequencyImpl();
  return cpu_frequency;
}
#else
/* static */ int64 CpuUtils::GetCycleCounterFrequency() {
  static const int64 cpu_frequency = GetCycleCounterFrequencyImpl();
  return cpu_frequency;
}
#endif

/* static */ double CpuUtils::GetMicroSecPerClock() {
  static const double micro_sec_per_clock =
      (1000.0 * 1000.0) / static_cast<double>(GetCycleCounterFrequency());
  return micro_sec_per_clock;
}

/* static */ void CpuUtils::ResetClockCycle() {
  GetCpuUtilsHelperSingletonInstance().ResetClockCycle();
}

/* static */ void CpuUtils::EnableClockCycleProfiling(const bool enable) {
  GetCpuUtilsHelperSingletonInstance().EnableClockCycleProfiling(enable);
}

/* static */ std::chrono::duration<double> CpuUtils::ConvertClockCycleToTime(
    const int64 clock_cycle) {
  return std::chrono::duration<double>(static_cast<double>(clock_cycle) /
                                       GetCycleCounterFrequency());
}

/* static */ int64 CpuUtils::GetCycleCounterFrequencyImpl() {
// TODO(satok): do not switch by macro here
#if defined(__ANDROID__)
  return GetCpuUtilsHelperSingletonInstance().CalculateCpuFrequency();
#elif defined(__linux__)
  // Read the contents of /proc/cpuinfo.
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo) {
    LOG(WARNING) << "Failed to open /proc/cpuinfo";
    return INVALID_FREQUENCY;
  }
  string line;
  while (std::getline(cpuinfo, line)) {
    double cpu_freq = 0.0;
    int retval = 0;
    double freq_factor = 2.0;
#if (defined(__powerpc__) || \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
    retval = sscanf(line.c_str(), "clock              : %lfMHz", &cpu_freq);
    freq_factor = 1.0;
#else
    retval = sscanf(line.c_str(), "bogomips : %lf", &cpu_freq);
#endif
    if (retval > 0) {
      const double freq_ghz = cpu_freq / 1000.0 / freq_factor;
      if (retval != 1 || freq_ghz < 0.01) {
        LOG(WARNING) << "Failed to get CPU frequency: " << freq_ghz << " GHz";
        return INVALID_FREQUENCY;
      }
      const int64 freq_n =
          static_cast<int64>(freq_ghz * 1000.0 * 1000.0 * 1000.0);
      LOG(INFO) << "CPU Frequency: " << freq_n << " Hz";
      return freq_n;
    }
  }
  LOG(WARNING)
      << "Failed to find bogomips or clock in /proc/cpuinfo; cannot determine "
         "CPU frequency";
  return INVALID_FREQUENCY;
#elif defined(__APPLE__)
  int64 freq_hz;
  FILE* fp =
      popen("sysctl hw | grep hw.cpufrequency_max: | cut -d' ' -f 2", "r");
  if (fp == nullptr) {
    return INVALID_FREQUENCY;
  }
  if (fscanf(fp, "%lld", &freq_hz) != 1) {
    return INVALID_FREQUENCY;
  }
  pclose(fp);
  if (freq_hz < 1e6) {
    LOG(WARNING) << "Failed to get CPU frequency: " << freq_hz << " Hz";
    return INVALID_FREQUENCY;
  }
  return freq_hz;
#elif defined(_WIN32)
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return freq.QuadPart;
#else
  // TODO(satok): Support other OS if needed
  // Return INVALID_FREQUENCY on unsupported OS
  return INVALID_FREQUENCY;
#endif
}

/* static */ ICpuUtilsHelper& CpuUtils::GetCpuUtilsHelperSingletonInstance() {
  static absl::once_flag flag;
  absl::call_once(flag, []() {
    if (cpu_utils_helper_instance_ != nullptr) {
      LOG(FATAL) << "cpu_utils_helper_instance_ is already instantiated.";
    }
#if defined(__ANDROID__) && (__ANDROID_API__ >= 21) && \
    (defined(__ARM_ARCH_7A__) || defined(__aarch64__))
    cpu_utils_helper_instance_ = new AndroidArmV7ACpuUtilsHelper();
#else
      cpu_utils_helper_instance_ = new DefaultCpuUtilsHelper();
#endif
  });
  return *cpu_utils_helper_instance_;
}

}  // namespace profile_utils
}  // namespace tensorflow
