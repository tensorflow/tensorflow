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

#ifdef ETHOS_U
#include CMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE
#include "ethosu_driver.h"
#endif
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace tflite {

#ifdef ETHOS_U
struct ethosu_driver* ethosu0_driver = &ethosu_drv;

void ethosuIrqHandler0() { ethosu_irq_handler_v2(ethosu0_driver); }
#endif

extern "C" {
void uart_init(void);
}

void InitializeTarget() {
  uart_init();

#ifdef ETHOS_U
  constexpr int ethosu_base_address = 0x48102000;
  constexpr int ethosu_irq = 56;

  // Initialize Ethos-U NPU driver.
  if (ethosu_init_v4(ethosu0_driver,
                     reinterpret_cast<void*>(ethosu_base_address), 0, 0, 1,
                     1)) {
    MicroPrintf("Failed to initialize Ethos-U driver %p", ethosu0_driver);
  }
  NVIC_SetVector(static_cast<IRQn_Type>(ethosu_irq),
                 (uint32_t)&ethosuIrqHandler0);
  NVIC_EnableIRQ(static_cast<IRQn_Type>(ethosu_irq));
#endif
}

}  // namespace tflite
