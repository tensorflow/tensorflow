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

#include "tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

#if defined(__ANDROID__) && defined(__ARM_ARCH_7A__) && (__ANDROID_API__ >= 21)

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace profile_utils {

/* static */ constexpr int AndroidArmV7ACpuUtilsHelper::INVALID_FD;
/* static */ constexpr int64 AndroidArmV7ACpuUtilsHelper::INVALID_CPU_FREQUENCY;

void AndroidArmV7ACpuUtilsHelper::ResetClockCycle() {
  if (!is_initialized_) {
    return;
  }
  ioctl(fd_, PERF_EVENT_IOC_RESET, 0);
}

uint64 AndroidArmV7ACpuUtilsHelper::GetCurrentClockCycle() {
  if (!is_initialized_) {
    return 1;  // DUMMY
  }
  long long count;
  read(fd_, &count, sizeof(long long));
  return static_cast<uint64>(count);
}

void AndroidArmV7ACpuUtilsHelper::EnableClockCycleProfiling(const bool enable) {
  if (!is_initialized_) {
    // Initialize here to avoid unnecessary initialization
    InitializeInternal();
  }
  if (enable) {
    const int64 cpu0_scaling_min = ReadCpuFrequencyFile(0, "scaling_min");
    const int64 cpu0_scaling_max = ReadCpuFrequencyFile(0, "scaling_max");
    if (cpu0_scaling_max != cpu0_scaling_min) {
      LOG(WARNING) << "You enabled clock cycle profile but frequency may "
                   << "be scaled. (max = " << cpu0_scaling_max << ", min "
                   << cpu0_scaling_min << ")";
    }
    ResetClockCycle();
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
  } else {
    ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
  }
}

int64 AndroidArmV7ACpuUtilsHelper::CalculateCpuFrequency() {
  return ReadCpuFrequencyFile(0, "scaling_cur");
}

void AndroidArmV7ACpuUtilsHelper::InitializeInternal() {
  perf_event_attr pe;

  memset(&pe, 0, sizeof(perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(perf_event_attr);
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;

  fd_ = OpenPerfEvent(&pe, 0, -1, -1, 0);
  if (fd_ == INVALID_FD) {
    LOG(WARNING) << "Error opening perf event";
    is_initialized_ = false;
  } else {
    is_initialized_ = true;
  }
}

int AndroidArmV7ACpuUtilsHelper::OpenPerfEvent(perf_event_attr *const hw_event,
                                               const pid_t pid, const int cpu,
                                               const int group_fd,
                                               const unsigned long flags) {
  const int ret =
      syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return ret;
}

int64 AndroidArmV7ACpuUtilsHelper::ReadCpuFrequencyFile(
    const int cpu_id, const char *const type) {
  const string file_path = strings::Printf(
      "/sys/devices/system/cpu/cpu%d/cpufreq/%s_freq", cpu_id, type);
  FILE *fp = fopen(file_path.c_str(), "r");
  if (fp == nullptr) {
    return INVALID_CPU_FREQUENCY;
  }
  int64 freq_in_khz = INVALID_CPU_FREQUENCY;
  const int retval = fscanf(fp, "%lld", &freq_in_khz);
  if (retval < 0) {
    LOG(WARNING) << "Failed to \"" << file_path << "\"";
    return INVALID_CPU_FREQUENCY;
  }
  pclose(fp);
  return freq_in_khz * 1000;  // The file contains cpu frequency in khz
}

}  // namespace profile_utils
}  // namespace tensorflow

// defined(__ANDROID__) && defined(__ARM_ARCH_7A__) && (__ANDROID_API__ >= 21)
#else

// Dummy implementations to avoid link error.

namespace tensorflow {
namespace profile_utils {

void AndroidArmV7ACpuUtilsHelper::ResetClockCycle() {}
uint64 AndroidArmV7ACpuUtilsHelper::GetCurrentClockCycle() { return 1; }
void AndroidArmV7ACpuUtilsHelper::EnableClockCycleProfiling(bool) {}
int AndroidArmV7ACpuUtilsHelper::OpenPerfEvent(perf_event_attr *const,
                                               const pid_t, const int,
                                               const int, const unsigned long) {
  return 0;
}
int64 AndroidArmV7ACpuUtilsHelper::CalculateCpuFrequency() { return 0; }

}  // namespace profile_utils
}  // namespace tensorflow

// defined(__ANDROID__) && defined(__ARM_ARCH_7A__) && (__ANDROID_API__ >= 21)
#endif
