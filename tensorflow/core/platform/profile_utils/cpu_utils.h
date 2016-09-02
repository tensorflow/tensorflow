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
// This class is designed to get accurate profile for programs.

#ifndef TENSORFLOW_PLATFORM_PROFILEUTILS_CPU_UTILS_H__
#define TENSORFLOW_PLATFORM_PROFILEUTILS_CPU_UTILS_H__

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/profile_utils/i_cpu_utils_helper.h"
#include "tensorflow/core/platform/types.h"

#if defined(ARMV6) || defined(__ARM_ARCH_7A__)
#include <sys/time.h>
#endif

namespace tensorflow {

namespace profile_utils {

class CpuUtils {
 public:
  // Constant for invalid frequency.
  // This value is returned when the furequency is not obtained somehow.
  static constexpr int64 INVALID_FREQUENCY = -1;
  static constexpr uint64 DUMMY_CYCLE_CLOCK = 1;

  // Return current clock cycle. This function is designed to
  // minimize the overhead to get clock and maximize the accuracy of
  // time for profile.
  // This returns unsigned int because there is no guarantee that rdtsc
  // is less than 2 ^ 61.
  static inline uint64 GetCurrentClockCycle() {
// ----------------------------------------------------------------
#if defined(__x86_64__) || defined(__amd64__)
    uint64_t high, low;
    __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
    return (high << 32) | low;
// ----------------------------------------------------------------
#elif defined(__aarch64__)
    // System timer of ARMv8 runs at a different frequency than the CPU's.
    // The frequency is fixed, typically in the range 1-50MHz.  It can because
    // read at CNTFRQ special register.  We assume the OS has set up
    // the virtual timer properly.
    uint64_t virtual_timer_value;
    asm volatile("mrs %0, cntvct_el0" : "=r"(virtual_timer_value));
    return virtual_timer_value;
// ----------------------------------------------------------------
// V6 is the earliest arm that has a standard cyclecount
#elif defined(ARMV6) || defined(__ARM_ARCH_7A__)
    uint32_t pmccntr;
    uint32_t pmuseren;
    uint32_t pmcntenset;
    // Read the user mode perf monitor counter access permissions.
    asm volatile("mrc p15, 0, %0, c9, c14, 0" : "=r"(pmuseren));
    if (pmuseren & 1) {  // Allows reading perfmon counters for user mode code.
      asm volatile("mrc p15, 0, %0, c9, c12, 1" : "=r"(pmcntenset));
      if (pmcntenset & 0x80000000ul) {  // Is it counting?
        asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(pmccntr));
        // The counter is set up to count every 64th cyclecount
        return static_cast<uint64>(pmccntr) * 64;  // Should optimize to << 64
      }
    }
    // Returning dummy clock when can't access to the counter
    return DUMMY_CYCLE_CLOCK;
#else
    // TODO(satok): Support generic way to emulate clock count.
    // TODO(satok): Support other architectures if wanted.
    // Returning dummy clock when can't access to the counter
    return DUMMY_CYCLE_CLOCK;
#endif
  }

  // Return cpu frequency. As this method caches the cpu frequency internally,
  // there is no overhead except function call to call this method.
  static int64 GetCpuFrequency();

  // Return cached cpu count per each micro second.
  // As this method caches the cpu frequency internally,
  // there is no overhead except function call to call this method.
  static int GetClockPerMicroSec();

  // Return micro secound per each clock
  // As this method caches the cpu frequency internally,
  // there is no overhead except function call to call this method.
  static double GetMicroSecPerClock();

  // Initialize CpuUtils
  // This method is called from the static initializer declared in cpu_utils.cc
  // This initializes state and cached static variables declared in functions.
  static void Initialize();

  // Reset clock cycle
  // Resetting clock cycle is recommended to prevent
  // clock cycle counters from overflowing on some platforms.
  static void ResetClockCycle();

  // Enable clock cycle profile
  // You can enable / disable profile if it's supported by the platform
  static void EnableClockCycleProfile(bool enable);

 private:
  class DefaultCpuUtilsHelper : public ICpuUtilsHelper {
   public:
    DefaultCpuUtilsHelper() = default;
    void Initialize() final {}
    void ResetClockCycle() final {}
    uint64 GetCurrentClockCycle() final { return DUMMY_CYCLE_CLOCK; }
    void EnableClockCycleProfile(bool /* enable */) final {}

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(DefaultCpuUtilsHelper);
  };

  // Return cpu frequency.
  // CAVEAT: as this method calls system call and parse the mssage,
  // this call may be slow. This is why this class caches the value by
  // StaticVariableInitializer.
  static int64 GetCpuFrequencyImpl();

  static ICpuUtilsHelper& GetCpuUtilsHelper();

  TF_DISALLOW_COPY_AND_ASSIGN(CpuUtils);
};

}  // namespace profile_utils

}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_PROFILEUTILS_CPU_UTILS_H__
