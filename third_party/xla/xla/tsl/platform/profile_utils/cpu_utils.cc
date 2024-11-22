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

#include "xla/tsl/platform/profile_utils/cpu_utils.h"

#include <fstream>
#include <limits>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include "absl/base/call_once.h"
#include "xla/tsl/platform/profile_utils/android_armv7a_cpu_utils_helper.h"
#include "tsl/platform/logging.h"

namespace tsl {
namespace profile_utils {

/* static */ constexpr int64_t CpuUtils::INVALID_FREQUENCY;

static ICpuUtilsHelper* cpu_utils_helper_instance_ = nullptr;

#if (defined(__powerpc__) ||                                             \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) || \
    (defined(__s390x__))
/* static */ uint64 CpuUtils::GetCycleCounterFrequency() {
  static const uint64 cpu_frequency = GetCycleCounterFrequencyImpl();
  return cpu_frequency;
}
#else
/* static */ int64_t CpuUtils::GetCycleCounterFrequency() {
  static const int64_t cpu_frequency = GetCycleCounterFrequencyImpl();
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

/* static */ void CpuUtils::EnableClockCycleProfiling() {
  GetCpuUtilsHelperSingletonInstance().EnableClockCycleProfiling();
}

/* static */ void CpuUtils::DisableClockCycleProfiling() {
  GetCpuUtilsHelperSingletonInstance().DisableClockCycleProfiling();
}

/* static */ std::chrono::duration<double> CpuUtils::ConvertClockCycleToTime(
    const int64_t clock_cycle) {
  return std::chrono::duration<double>(static_cast<double>(clock_cycle) /
                                       GetCycleCounterFrequency());
}

/* static */ int64_t CpuUtils::GetCycleCounterFrequencyImpl() {
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
#elif defined(__s390x__)
    retval = sscanf(line.c_str(), "bogomips per cpu: %lf", &cpu_freq);
#elif defined(__aarch64__)
    retval = sscanf(line.c_str(), "BogoMIPS : %lf", &cpu_freq);
#else
    retval = sscanf(line.c_str(), "bogomips : %lf", &cpu_freq);
#endif
    if (retval > 0) {
      const double freq_ghz = cpu_freq / 1000.0 / freq_factor;
      if (retval != 1 || freq_ghz < 0.01) {
        LOG(WARNING) << "Failed to get CPU frequency: " << freq_ghz << " GHz";
        return INVALID_FREQUENCY;
      }
      const int64_t freq_n =
          static_cast<int64_t>(freq_ghz * 1000.0 * 1000.0 * 1000.0);
      VLOG(1) << "CPU Frequency: " << freq_n << " Hz";
      return freq_n;
    }
  }
  LOG(WARNING)
      << "Failed to find bogomips or clock in /proc/cpuinfo; cannot determine "
         "CPU frequency";
  return INVALID_FREQUENCY;
#elif defined(__APPLE__)
  int64_t freq_hz = 0;
  size_t freq_hz_size = sizeof(freq_hz);
  int retval =
      sysctlbyname("hw.cpufrequency_max", &freq_hz, &freq_hz_size, NULL, 0);
  if (retval != 0 || freq_hz < 1e6) {
    // Apple M1/M2 do not have hw.cpufrequency.* values, but instead rely on
    // a base clock rate hw.tbfrequency and multiplier kern.clockrate.hz.
    int64_t tbfrequency = 0;
    size_t tbfrequency_size = sizeof(tbfrequency);
    retval = sysctlbyname("hw.tbfrequency", &tbfrequency, &tbfrequency_size,
                          NULL, 0);
    if (retval == 0) {
      clockinfo clock_info;
      size_t clock_info_size = sizeof(clock_info);
      retval = sysctlbyname("kern.clockrate", &clock_info, &clock_info_size,
                            NULL, 0);
      if (retval == 0) {
        freq_hz = clock_info.hz * tbfrequency;
      }
    }

    if (retval != 0 || freq_hz < 1e6) {
      LOG(WARNING) << "Failed to get CPU frequency: " << freq_hz << " Hz";
      return INVALID_FREQUENCY;
    }
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
}  // namespace tsl
