/*
 * Copyright (C) EEMBC(R). All Rights Reserved
 *
 * All EEMBC Benchmark Software are products of EEMBC and are provided under the
 * terms of the EEMBC Benchmark License Agreements. The EEMBC Benchmark Software
 * are proprietary intellectual properties of EEMBC and its Members and is
 * protected under all applicable laws, including all applicable copyright laws.
 *
 * If you received this EEMBC Benchmark Software without having a currently
 * effective EEMBC Benchmark License Agreement, you must discontinue use.
 */

#ifndef __TH_LIB_H
#define __TH_LIB_H

#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include "tensorflow/lite/micro/benchmarks/eembc/monitor/th_api/th_libc.h"
#include "tensorflow/lite/micro/benchmarks/eembc/monitor/ee_main.h"

// For printf_ redefine
#include "tensorflow/lite/micro/tools/make/downloads/TensaiSDK/3rd_party/tiny_printf/printf.h"
// For UART baud definitions
#include "tensorflow/lite/micro/tools/make/downloads/TensaiSDK/soc/ecm3532/m3/csp/inc/eta_csp_uart.h"

// It is crucial to follow EEMBC message syntax for key messages
#define EE_MSG_TIMESTAMP "m-lap-us-%lu\r\n"

#ifndef EE_CFG_ENERGY_MODE
#define EE_CFG_ENERGY_MODE 1
#endif

#if EE_CFG_ENERGY_MODE==1
#   define EE_MSG_TIMESTAMP_MODE "m-timestamp-mode-energy\r\n"
#else
#   define EE_MSG_TIMESTAMP_MODE "m-timestamp-mode-performance\r\n"
#endif

/** 
 * This string is used in the "name%" command. When the host UI starts a test,
 * it calles the "name%" command, and the result is captured in the log file.
 * It can be quite useful to have the device's name in the log file for future
 * reference or debug.
 */
#define TH_VENDOR_NAME_STRING "unspecified"

void th_monitor_initialize(void);
void th_timestamp_initialize(void);
void th_timestamp(void);
void th_serialport_initialize(void);
/* PORT:ECM3532
void th_printf(const char * fmt, ...);
*/
#define th_printf printf_
void th_command_ready(volatile char *);
/* PORT:ECM3532
// Yay, we have getchar
*/
void th_check_serial(void);

#endif // __TH_LIB_H
