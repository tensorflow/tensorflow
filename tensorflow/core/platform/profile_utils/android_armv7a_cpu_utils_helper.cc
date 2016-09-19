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

#if defined(__ANDROID__) && defined(__ARM_ARCH_7A__) && (__ANDROID_API__ >= 21)
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

namespace tensorflow {
namespace profile_utils {

/* static */ constexpr int AndroidArmV7ACpuUtilsHelper::INVALID_FD;

void AndroidArmV7ACpuUtilsHelper::Initialize() {
  struct perf_event_attr pe;

  memset(&pe, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(struct perf_event_attr);
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
    return;
  }
  if (enable) {
    ioctl(fd_, PERF_EVENT_IOC_ENABLE, 0);
  } else {
    ioctl(fd_, PERF_EVENT_IOC_DISABLE, 0);
  }
}

int AndroidArmV7ACpuUtilsHelper::OpenPerfEvent(
    struct perf_event_attr *const hw_event, const pid_t pid, const int cpu,
    const int group_fd, const unsigned long flags) {
  const int ret =
      syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  return ret;
}

}  // namespace profile_utils
}  // namespace tensorflow

// defined(__ANDROID__) && defined(__ARM_ARCH_7A__) && (__ANDROID_API__ >= 21)
#else

// Dummy implementations to avoid link error.
#include "tensorflow/core/platform/profile_utils/android_armv7a_cpu_utils_helper.h"

namespace tensorflow {
namespace profile_utils {

void AndroidArmV7ACpuUtilsHelper::Initialize() {}
void AndroidArmV7ACpuUtilsHelper::ResetClockCycle() {}
uint64 AndroidArmV7ACpuUtilsHelper::GetCurrentClockCycle() { return 1; }
void AndroidArmV7ACpuUtilsHelper::EnableClockCycleProfiling(bool) {}
int AndroidArmV7ACpuUtilsHelper::OpenPerfEvent(struct perf_event_attr *const,
                                               const pid_t, const int,
                                               const int, const unsigned long) {
  return 0;
}

}  // namespace profile_utils
}  // namespace tensorflow

// defined(__ANDROID__) && defined(__ARM_ARCH_7A__) && (__ANDROID_API__ >= 21)
#endif
