/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Reference implementation of timer functions.  Platforms are not required to
// implement these timer methods, but they are required to enable profiling.

// On platforms that have a POSIX stack or C library, it can be written using
// methods from <sys/time.h> or clock() from <time.h>.

// To add an equivalent function for your own platform, create your own
// implementation file, and place it in a subfolder with named after the OS
// you're targeting. For example, see the Cortex M bare metal version in
// tensorflow/lite/micro/bluepill/micro_timer.cc or the mbed one on
// tensorflow/lite/micro/mbed/micro_timer.cc.

#include "tensorflow/lite/micro/micro_time.h"

#include "tensorflow/lite/micro/debug_log.h"

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT

namespace tflite {
namespace {

// Select CTIMER 1 as benchmarking timer on Sparkfun Edge. This timer must not
// be used elsewhere.
constexpr int kTimerNum = 1;

// Clock set to operate at 12MHz.
constexpr int kClocksPerSecond = 12e6;

// Enables 96MHz burst mode on Sparkfun Edge. Enable in timer since most
// benchmarks and profilers want maximum performance for debugging.
void BurstModeEnable() {
  am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);

  // Set the default cache configuration
  am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
  am_hal_cachectrl_enable();

  am_hal_burst_avail_e eBurstModeAvailable;
  am_hal_burst_mode_e eBurstMode;

  // Check that the Burst Feature is available.
  int status = am_hal_burst_mode_initialize(&eBurstModeAvailable);
  if (status != AM_HAL_STATUS_SUCCESS ||
      eBurstModeAvailable != AM_HAL_BURST_AVAIL) {
    DebugLog("Failed to initialize burst mode.");
    return;
  }

  status = am_hal_burst_mode_enable(&eBurstMode);

  if (status != AM_HAL_STATUS_SUCCESS || eBurstMode != AM_HAL_BURST_MODE) {
    DebugLog("Failed to Enable Burst Mode operation\n");
  }
}

}  // namespace

int32_t ticks_per_second() { return kClocksPerSecond; }

// Calling this method enables a timer that runs for eternity. The user is
// responsible for avoiding trampling on this timer's config, otherwise timing
// measurements may no longer be valid.
int32_t GetCurrentTimeTicks() {
  // TODO(b/150808076): Split out initialization, intialize in interpreter.
  static bool is_initialized = false;
  if (!is_initialized) {
    BurstModeEnable();
    am_hal_ctimer_config_t timer_config;
    // Operate as a 32-bit timer.
    timer_config.ui32Link = 1;
    // Set timer A to continuous mode at 12MHz.
    timer_config.ui32TimerAConfig =
        AM_HAL_CTIMER_FN_CONTINUOUS | AM_HAL_CTIMER_HFRC_12MHZ;

    am_hal_ctimer_stop(kTimerNum, AM_HAL_CTIMER_BOTH);
    am_hal_ctimer_clear(kTimerNum, AM_HAL_CTIMER_BOTH);
    am_hal_ctimer_config(kTimerNum, &timer_config);
    am_hal_ctimer_start(kTimerNum, AM_HAL_CTIMER_TIMERA);
    is_initialized = true;
  }
  return CTIMERn(kTimerNum)->TMR0;
}

}  // namespace tflite
