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

#ifndef XLA_TSL_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_
#define XLA_TSL_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_

#include <chrono>
#include <memory>

#include "xla/tsl/platform/profile_utils/i_cpu_utils_helper.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

#if defined(ARMV6) || defined(__ARM_ARCH_7A__)
#include <sys/time.h>
#endif

#if defined(_WIN32)
#include <intrin.h>
#endif

namespace tsl {

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
  static constexpr int64_t INVALID_FREQUENCY = -1;
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
#elif defined(__powerpc64__) || defined(__ppc64__)
    uint64 __t;
    __asm__ __volatile__("mfspr %0,268" : "=r"(__t));
    return __t;

#elif defined(__powerpc__) || defined(__ppc__)
    uint64 upper, lower, tmp;
    __asm__ volatile(
        "0:                     \n"
        "\tmftbu   %0           \n"
        "\tmftb    %1           \n"
        "\tmftbu   %2           \n"
        "\tcmpw    %2,%0        \n"
        "\tbne     0b           \n"
        : "=r"(upper), "=r"(lower), "=r"(tmp));
    return ((static_cast<uint64>(upper) << 32) | lower);
#elif defined(__s390x__)
    // TOD Clock of s390x runs at a different frequency than the CPU's.
    // The stepping is 244 picoseconds (~4Ghz).
    uint64 t;
    __asm__ __volatile__("stckf %0" : "=Q"(t));
    return t;
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
  static int64_t GetCycleCounterFrequency();
#endif

  // Return micro second per each clock
  // As this method caches the cpu frequency internally,
  // the first call will incur overhead, but not subsequent calls.
  static double GetMicroSecPerClock();

  // Reset clock cycle
  // Resetting clock cycle is recommended to prevent
  // clock cycle counters from overflowing on some platforms.
  static void ResetClockCycle();

  // Enable/Disable clock cycle profile
  // You can enable / disable profile if it's supported by the platform
  static void EnableClockCycleProfiling();
  static void DisableClockCycleProfiling();

  // Return chrono::duration per each clock
  static std::chrono::duration<double> ConvertClockCycleToTime(
      const int64_t clock_cycle);

 private:
  class DefaultCpuUtilsHelper : public ICpuUtilsHelper {
   public:
    DefaultCpuUtilsHelper() = default;
    void ResetClockCycle() final {}
    uint64 GetCurrentClockCycle() final { return DUMMY_CYCLE_CLOCK; }
    void EnableClockCycleProfiling() final {}
    void DisableClockCycleProfiling() final {}
    int64_t CalculateCpuFrequency() final { return INVALID_FREQUENCY; }

   private:
    DefaultCpuUtilsHelper(const DefaultCpuUtilsHelper&) = delete;
    void operator=(const DefaultCpuUtilsHelper&) = delete;
  };

  // Return cpu frequency.
  // CAVEAT: as this method calls system call and parse the message,
  // this call may be slow. This is why this class caches the value by
  // StaticVariableInitializer.
  static int64_t GetCycleCounterFrequencyImpl();

  // Return a singleton of ICpuUtilsHelper
  // ICpuUtilsHelper is declared as a function-local static variable
  // for the following two reasons:
  // 1. Avoid passing instances to all classes which want
  // to use profiling tools in CpuUtils
  // 2. Minimize the overhead of acquiring ICpuUtilsHelper
  static ICpuUtilsHelper& GetCpuUtilsHelperSingletonInstance();

  CpuUtils(const CpuUtils&) = delete;
  void operator=(const CpuUtils&) = delete;
};

}  // namespace profile_utils

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_PROFILE_UTILS_CPU_UTILS_H_
