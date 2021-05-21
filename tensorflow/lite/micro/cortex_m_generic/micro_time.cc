/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/micro_time.h"

// DWT (Data Watchpoint and Trace) registers, only exists on ARM Cortex with a
// DWT unit.
#define KIN1_DWT_CONTROL (*((volatile uint32_t*)0xE0001000))
/*!< DWT Control register */

// DWT Control register.
#define KIN1_DWT_CYCCNTENA_BIT (1UL << 0)

// CYCCNTENA bit in DWT_CONTROL register.
#define KIN1_DWT_CYCCNT (*((volatile uint32_t*)0xE0001004))

// DWT Cycle Counter register.
#define KIN1_DEMCR (*((volatile uint32_t*)0xE000EDFC))

// DEMCR: Debug Exception and Monitor Control Register.
#define KIN1_TRCENA_BIT (1UL << 24)

#define KIN1_LAR (*((volatile uint32_t*)0xE0001FB0))

#define KIN1_DWT_CONTROL (*((volatile uint32_t*)0xE0001000))

// Unlock access to DWT (ITM, etc.)registers.
#define KIN1_UnlockAccessToDWT() KIN1_LAR = 0xC5ACCE55;

// TRCENA: Enable trace and debug block DEMCR (Debug Exception and Monitor
// Control Register.
#define KIN1_InitCycleCounter() KIN1_DEMCR |= KIN1_TRCENA_BIT

#define KIN1_ResetCycleCounter() KIN1_DWT_CYCCNT = 0
#define KIN1_EnableCycleCounter() KIN1_DWT_CONTROL |= KIN1_DWT_CYCCNTENA_BIT
#define KIN1_DisableCycleCounter() KIN1_DWT_CONTROL &= ~KIN1_DWT_CYCCNTENA_BIT
#define KIN1_GetCycleCounter() KIN1_DWT_CYCCNT

namespace tflite {

int32_t ticks_per_second() { return 0; }

int32_t GetCurrentTimeTicks() {
  static bool is_initialized = false;
  if (!is_initialized) {
    KIN1_UnlockAccessToDWT();
    KIN1_InitCycleCounter();
    KIN1_ResetCycleCounter();
    KIN1_EnableCycleCounter();
    is_initialized = true;
  }
  return KIN1_GetCycleCounter();
}

}  // namespace tflite
