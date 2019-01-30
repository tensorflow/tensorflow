/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

/* This is file contains the entry point to the application and is called after
   startup.
   The GPIOs, Uart and timer are intialized and Tensorflow is invoked with the
   call to main().
   Tensorflow will print out if the tests have passed or failed and the
   execution time is also
   printed. */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "eta_bsp.h"
#include "eta_chip.h"
#include "eta_csp.h"
#include "eta_csp_buck.h"
#include "eta_csp_gpio.h"
#include "eta_csp_io.h"
#include "eta_csp_pwr.h"
#include "eta_csp_rtc.h"
#include "eta_csp_socctrl.h"
#include "eta_csp_sys_clock.h"
#include "eta_csp_timer.h"
#include "eta_csp_uart.h"

tUart g_sUart0 = {eUartNum0, eUartBaud115200};
tUart g_sUart1 = {eUartNum1, eUartBaud115200};

int init_main(int);
void EtaPrintExecutionTime(uint64_t);

//*****************************************************************************
//
// The entry point for the application.
//
//*****************************************************************************
extern int main(int argc, char** argv);


int _main(void) {
  uint64_t time_ms;

  EtaCspInit();      // initialize csp registers
  EtaCspGpioInit();  // initialize gpios
  EtaCspUartInit(&g_sUart1, eUartNum0, eUartBaud115200,
                 eUartFlowControlHardware);  // initialize Uart
  EtaCspBuckInit(ETA_BSP_VDD_IO_SETTING, eBuckAo600Mv, eBuckM3Frequency60Mhz,
                 eBuckMemVoltage900Mv);  // set M3 freq
  EtaCspTimerInitMs();                   // start timer
  main(0, NULL);  // Call to Tensorflow; this will print if test was successful.
  time_ms = EtaCspTimerCountGetMs();  // read time
  EtaPrintExecutionTime(time_ms);     // print execution time
}

void EtaPrintExecutionTime(uint64_t time_ms) {
  uint8_t c;
  int k1;
  char time_string[] = "00000";

  EtaCspIoPrintf("Execution time (msec) = ");
  if (time_ms < 100000)  // Convert time to a string
  {
    for (k1 = 0; k1 < 5; k1++) {
      c = time_ms % 10;
      time_ms = time_ms / 10;
      time_string[k1] = (char)(0x30 + c);
    }
    for (k1 = 4; k1 >= 0; k1--) {  // print out 1 char at a time
      EtaCspUartPutc(&g_sUart1, time_string[k1]);
    }
  } else {
    EtaCspIoPrintf("Execution time exceeds 100 sec\n");
  }
  EtaCspIoPrintf("\n\n");
}
