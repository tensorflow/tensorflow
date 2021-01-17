/*************************************************************************************\
* Copyright (C) CEVA(R) Inc. All rights reserved *
* *
* This information embodies materials and concepts, which are proprietary and *
* confidential to CEVA Inc., and is made available solely pursuant to the terms
*
* of a written license agreement, or NDA, or another written agreement, as *
* applicable ("CEVA Agreement"), with CEVA Inc. or any of its subsidiaries
("CEVA").  *
* *
* This information can be used only with the written permission from CEVA, in *
* accordance with the terms and conditions stipulated in the CEVA Agreement,
under    *
* which the information has been supplied and solely as expressly permitted for
*
* the purpose specified in the CEVA Agreement. *
* *
* This information is made available exclusively to licensees or parties that
have    *
* received express written authorization from CEVA to download or receive the *
* information and have agreed to the terms and conditions of the CEVA Agreement.
*
* *
* IF YOU HAVE NOT RECEIVED SUCH EXPRESS AUTHORIZATION AND AGREED TO THE CEVA *
* AGREEMENT, YOU MAY NOT DOWNLOAD, INSTALL OR USE THIS INFORMATION. *
* *
* The information contained in this document is subject to change without notice
*
* and does not represent a commitment on any part of CEVA. Unless specifically *
* agreed otherwise in the CEVA Agreement, CEVA make no warranty of any kind with
*
* regard to this material, including, but not limited to implied warranties of *
* merchantability and fitness for a particular purpose whether arising out of
law,    *
* custom, conduct or otherwise. *
* *
* While the information contained herein is assumed to be accurate, CEVA assumes
no   *
* responsibility for any errors or omissions contained herein, and assumes no *
* liability for special, direct, indirect or consequential damage, losses,
costs,     *
* charges, claims, demands, fees or expenses, of any nature or kind, which are *
* incurred in connection with the furnishing, performance or use of this
material.    *
* *
* This document contains proprietary information, which is protected by U.S. and
*
* international copyright laws. All rights reserved. No part of this document
may     *
* be reproduced, photocopied, or translated into another language without the
prior   *
* written consent of CEVA. *
\*************************************************************************************/
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
  FILE *f_mcps_report;
#else
#define MCPS_VARIBLES             \
  clock_t c1, c2;                 \
  int ClockCEVA, Constant_cycles; \
  FILE *f_mcps_report;
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
