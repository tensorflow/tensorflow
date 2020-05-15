/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstring>
#include <cstdint>
#include <cstdio>

// Print to debug console by default. One can define next to extend destinations set:
// EMSDP_LOG_TO_MEMORY 
//   : fill .debug_log memory region (data section) with passed chars. 
// EMSDP_LOG_TO_HOST 
//   : Use MetaWare HostLink to print output log. Requires Synopsys MetaWare debugger  
// EMSDP_LOG_TO_UART 
//   : use default debug UART (out to FTDI channel 0). The same USB Port is used for JTAG.
#define EMSDP_LOG_TO_UART

// Memory size for symbols dump in EMSDP_LOG_TO_MEMORY destination
#define EMSDP_LOG_TO_MEMORY_SIZE (2 * 1024)

// EMSDP Debug UART related defines (registers and bits)
#define EMSDP_DBG_UART_BASE (0xF0004000U)
#define DW_UART_CPR_FIFO_STAT (1 << 10)
#define DW_UART_USR_TFNF (0x02)
#define DW_UART_LSR_TXD_EMPTY (0x20)

// EMSDP UART registers map (only necessairy fields)
typedef volatile struct dw_uart_reg {
  uint32_t DATA; /* data in/out and DLL */
  uint32_t RES1[4];
  uint32_t LSR; /* Line Status Register */
  uint32_t RES2[25];
  uint32_t USR; /* UART status register */
  uint32_t RES3[29];
  uint32_t CPR; /* Component parameter register */
} DW_UART_REG;



// For simplicity we assume U-boot has already initialized debug console during 
// application loading (or on reset). Hence, we use only status and data registers 
// to organize blocking loop for printing symbols. No input and no IRQ handling. 
// See embarc_osp repository for full EMSDP uart driver.
// (https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_osp)
void DbgUartSendStr(const char* s) {
  DW_UART_REG* uart_reg_ptr = (DW_UART_REG*)(EMSDP_DBG_UART_BASE);
  const char* src = s;
  while (*src) {
    // Check uart status to send char
    bool uart_is_ready = false;
    if (uart_reg_ptr->CPR & DW_UART_CPR_FIFO_STAT)
      uart_is_ready = ((uart_reg_ptr->USR & DW_UART_USR_TFNF) != 0);
    else
      uart_is_ready = ((uart_reg_ptr->LSR & DW_UART_LSR_TXD_EMPTY) != 0);

    // Send char if uart is ready. 
    if (uart_is_ready)
      uart_reg_ptr->DATA = *src++;
  }
}

// Simple dump of symbols to a pre-allocated memory region.
// When total log exceeds memory region size, cursor is moved to its begining.
// The memory region can be viewed afterward with debugger.
// It can be viewed/read with debugger afterward.
void LogToMem(const char* s) {
  static int cursor = 0;
#pragma Bss(".debug_log")
  volatile static char debug_log_mem[EMSDP_LOG_TO_MEMORY_SIZE];
#pragma Bss()

  const char* src = s;
  while (*src) {
    debug_log_mem[cursor] = *src++;
    cursor = (cursor < EMSDP_LOG_TO_MEMORY_SIZE) ? cursor + 1 : 0;
  }
  debug_log_mem[cursor] = '^';
}


extern "C" void DebugLog(const char* s) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS

#if defined EMSDP_LOG_TO_UART
  DbgUartSendStr(s);
#endif

#if defined EMSDP_LOG_TO_MEMORY
#warning "EMSDP_LOG_TO_MEMORY is defined. View .debug_log memory region for stdout"
  LogToMem(s);
#endif

#if defined EMSDP_LOG_TO_HOST
#warning "EMSDP_LOG_TO_HOST is defined. Ensure hostlib is linked."
  fprintf(stderr, "%s", s);
#endif

#endif // TF_LITE_STRIP_ERROR_STRINGS
}


