/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/debug_log.h"

#include <stm32f4xx_hal.h>
#include <stm32f4xx_hal_uart.h>

#include <cstdio>

extern UART_HandleTypeDef DEBUG_UART_HANDLE;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __GNUC__
int __io_putchar(int ch) {
  HAL_UART_Transmit(&DEBUG_UART_HANDLE, (uint8_t*)&ch, 1, HAL_MAX_DELAY);

  return ch;
}
#else
int fputc(int ch, FILE* f) {
  HAL_UART_Transmit(&DEBUG_UART_HANDLE, (uint8_t*)&ch, 1, HAL_MAX_DELAY);

  return ch;
}
#endif /* __GNUC__ */

void DebugLog(const char* s) { fprintf(stderr, "%s", s); }

#ifdef __cplusplus
}
#endif
