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

#ifndef TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_
#define TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_

#include <chrono>
#include <memory>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/profile_utils/i_cpu_utils_helper.h"
#include "tensorflow/core/platform/types.h"

#if defined(ARMV6) || defined(__ARM_ARCH_7A__)
#include <sys/time.h>
#endif

#if defined(_WIN32)
#include <intrin.h>
#endif

namespace tensorflow {

namespace profile_utils {

// CpuUtils is a profiling tool with static functions
// designed to be called from multiple classes.
// A dedicated class which inherits ICpuUtilsHelper is
// stored as a function-local static variable which inherits
// GetCpuUtilsHelperSingletonInstance that caches CPU information,
// because loading CPU information may take a long time.
// Users must call EnableClockCycleProfiling before using CpuUtils.
class CpuUtils {
 public:
  // Constant for invalid frequency.
  // This value is returned when the frequency is not obtained somehow.
  static constexpr int64 INVALID_FREQUENCY = -1;
  static constexpr uint64 DUMMY_CYCLE_CLOCK = 1;

  // Return current clock cycle. This function is designed to
  // minimize the overhead to get clock and maximize the accuracy of
  // time for profile.
  // This returns unsigned int because there is no guarantee that rdtsc
  // is less than 2 ^ 61.
  static inline uint64 GetCurrentClockCycle() {
#if defined(__ANDROID__)
    return GetCpuUtilsHelperSingletonInstance().GetCurrentClockCycle();
// ----------------------------------------------------------------
#elif defined(_WIN32)
    return __rdtsc();
// ----------------------------------------------------------------
#elif defined(__x86_64__) || defined(__amd64__)
    uint64_t high, low;
    __asm__ volatile("rdtsc" : "=a"(low), "=d"(high));
    return (high << 32) | low;
// ----------------------------------------------------------------
#elif defined(__aarch64__) && defined(TARGET_OS_IOS)
    // On iOS, we are not able to access the cntvct_el0 register.
    // As a temporary build fix, we will just return the dummy cycle clock.
    return DUMMY_CYCLE_CLOCK
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

// Return cycle counter frequency.
// As this method caches the cpu frequency internally,
// the first call will incur overhead, but not subsequent calls.
#if (defined(__powerpc__) ||                                             \
     defined(__ppc__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) || \
    (defined(__s390x__))
  static uint64 GetCycleCounterFrequency();
#else
  static int64 GetCycleCounterFrequency();
#endif

  // Return micro second per each clock
  // As this method caches the cpu frequency internally,
  // the first call will incur overhead, but not subsequent calls.
  static double GetMicroSecPerClock();

  // Reset clock cycle
  // Resetting clock cycle is recommended to prevent
  // clock cycle counters from overflowing on some platforms.
  static void ResetClockCycle();

  // Enable clock cycle profile
  // You can enable / disable profile if it's supported by the platform
  static void EnableClockCycleProfiling(bool enable);

  // Return chrono::duration per each clock
  static std::chrono::duration<double> ConvertClockCycleToTime(
      const int64 clock_cycle);

 private:
  class DefaultCpuUtilsHelper : public ICpuUtilsHelper {
   public:
    DefaultCpuUtilsHelper() = default;
    void ResetClockCycle() final {}
    uint64 GetCurrentClockCycle() final { return DUMMY_CYCLE_CLOCK; }
    void EnableClockCycleProfiling(bool /* enable */) final {}
    int64 CalculateCpuFrequency() final { return INVALID_FREQUENCY; }

   private:
    TF_DISALLOW_COPY_AND_ASSIGN(DefaultCpuUtilsHelper);
  };

  // Return cpu frequency.
  // CAVEAT: as this method calls system call and parse the mssage,
  // this call may be slow. This is why this class caches the value by
  // StaticVariableInitializer.
  static int64 GetCycleCounterFrequencyImpl();

  // Return a singleton of ICpuUtilsHelper
  // ICpuUtilsHelper is declared as a function-local static variable
  // for the following two reasons:
  // 1. Avoid passing instances to all classes which want
  // to use profiling tools in CpuUtils
  // 2. Minimize the overhead of acquiring ICpuUtilsHelper
  static ICpuUtilsHelper& GetCpuUtilsHelperSingletonInstance();

  TF_DISALLOW_COPY_AND_ASSIGN(CpuUtils);
};

}  // namespace profile_utils

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_
