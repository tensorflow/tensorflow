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

// MCPS measurement macros for CEVA optimized kernels

#ifndef MCPS_MACROS_
#define MCPS_MACROS_

#ifndef WIN32
#include <ceva-time.h>
#endif

#ifdef MCPS_MEASUREMENT

#ifdef STACK_MEASUREMENT
#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
void CEVA_BX_Stack_Marking(const int32_t _count);
int32_t CEVA_BX_Stack_Measurement(const int32_t count);
#if defined(__cplusplus)
}
#endif /* __cplusplus */
#endif

#define MCPS_CALL_RET_VALUE 4

#ifdef STACK_MEASUREMENT
#define MCPS_VARIBLES             \
  clock_t c1, c2;                 \
  int ClockCEVA, Constant_cycles; \
  int StackSize;                  \
  FILE* f_mcps_report;
#else
#define MCPS_VARIBLES             \
  clock_t c1, c2;                 \
  int ClockCEVA, Constant_cycles; \
  FILE* f_mcps_report;
#endif
#define MCPS_OPEN_FILE f_mcps_report = fopen("mcps_report.txt", "at");

#define MCPS_CLOSE_FILE fclose(f_mcps_report);

#ifdef STACK_MEASUREMENT
#define MCPS_START_CLOCK        \
  CEVA_BX_Stack_Marking(0x800); \
  reset_clock();                \
  start_clock();                \
  c1 = clock();                 \
  c2 = clock();                 \
  Constant_cycles = c2 - c1;    \
  c1 = clock();

#define MCPS_STOP_AND_LOG(...)                                 \
  c2 = clock();                                                \
  ClockCEVA = c2 - c1 - Constant_cycles - MCPS_CALL_RET_VALUE; \
  StackSize = CEVA_BX_Stack_Measurement(0x800) * 4;            \
  fprintf(f_mcps_report, __VA_ARGS__);                         \
  fprintf(f_mcps_report, ":cycles:%d:Stack:%d\r\n", ClockCEVA, StackSize);

#else  // STACK_MEASUREMENT
#define MCPS_START_CLOCK     \
  reset_clock();             \
  start_clock();             \
  c1 = clock();              \
  c2 = clock();              \
  Constant_cycles = c2 - c1; \
  c1 = clock();

#define MCPS_STOP_AND_LOG(...)                                 \
  c2 = clock();                                                \
  ClockCEVA = c2 - c1 - Constant_cycles - MCPS_CALL_RET_VALUE; \
  fprintf(f_mcps_report, __VA_ARGS__);                         \
  fprintf(f_mcps_report, ":cycles:%d\r\n", ClockCEVA);
#endif  // STACK_MEASUREMENT

#define MCPS_STOP_AND_PRINT(...)                               \
  c2 = clock();                                                \
  ClockCEVA = c2 - c1 - Constant_cycles - MCPS_CALL_RET_VALUE; \
  fprintf(stdout, __VA_ARGS__);                                \
  fprintf(stdout, ":cycles=%d\n", ClockCEVA);

#define MCPS_START_ONE \
  MCPS_VARIBLES;       \
  MCPS_OPEN_FILE;      \
  MCPS_START_CLOCK;
#define MCPS_STOP_ONE(...)        \
  MCPS_STOP_AND_LOG(__VA_ARGS__); \
  MCPS_CLOSE_FILE;

#else
#define MCPS_VARIBLES
#define MCPS_OPEN_FILE
#define MCPS_START_CLOCK
#define MCPS_STOP_AND_LOG(...)
#define MCPS_STOP_AND_PRINT(...)
#define MCPS_CLOSE_FILE

#define MCPS_START_ONE
#define MCPS_STOP_ONE(...)
#endif

#endif
