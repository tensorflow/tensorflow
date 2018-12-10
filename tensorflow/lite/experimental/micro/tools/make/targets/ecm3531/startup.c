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

#include <stdint.h>
#include "eta_chip.h"
#include "memio.h"

#ifndef NULL
#define NULL (0)
#endif

//*****************************************************************************
//
// Macro for hardware access, both direct and via the bit-band region.
//
//*****************************************************************************

int _main(int argc, char *argv[]);
void set_vtor(void);
void *startup_get_my_pc(void);

//*****************************************************************************
// Forward DECLS for interrupt service routines (ISR)
//*****************************************************************************
extern void ResetISR(void) __attribute__((weak, alias("default_ResetISR")));
extern void NmiSR(void) __attribute__((weak, alias("default_NmiSR")));
extern void FaultISR(void) __attribute__((weak, alias("default_FaultISR")));

extern void DebugMonitor_ISR(void)
    __attribute__((weak, alias("default_DebugMonitor_ISR")));
extern void SVCall_ISR(void) __attribute__((weak, alias("default_SVCall_ISR")));
extern void PENDSV_ISR(void) __attribute__((weak, alias("default_PENDSV_ISR")));

extern void SYSTICK_ISR(void)
    __attribute__((weak, alias("default_SYSTICK_ISR")));

extern void GPIO0_ISR(void) __attribute__((weak, alias("default_GPIO0_ISR")));
extern void GPIO1_ISR(void) __attribute__((weak, alias("default_GPIO1_ISR")));
extern void TIMER0_ISR(void) __attribute__((weak, alias("default_TIMER0_ISR")));
extern void TIMER1_ISR(void) __attribute__((weak, alias("default_TIMER1_ISR")));
extern void UART0_ISR(void) __attribute__((weak, alias("default_UART0_ISR")));
extern void UART1_ISR(void) __attribute__((weak, alias("default_UART1_ISR")));
extern void SPI0_ISR(void) __attribute__((weak, alias("default_SPI0_ISR")));
extern void SPI1_ISR(void) __attribute__((weak, alias("default_SPI1_ISR")));
extern void I2C0_ISR(void) __attribute__((weak, alias("default_I2C0_ISR")));
extern void I2C1_ISR(void) __attribute__((weak, alias("default_I2C1_ISR")));
extern void RTC0_ISR(void) __attribute__((weak, alias("default_RTC0_ISR")));
extern void RTC1_ISR(void) __attribute__((weak, alias("default_RTC1_ISR")));
extern void DSP_ISR(void) __attribute__((weak, alias("default_DSP_ISR")));
extern void ADC_ISR(void) __attribute__((weak, alias("default_ADC_ISR")));
extern void SW0_ISR(void) __attribute__((weak, alias("default_SW0_ISR")));
extern void SW1_ISR(void) __attribute__((weak, alias("default_SW1_ISR")));
extern void PWM_ISR(void) __attribute__((weak, alias("default_PWM_ISR")));
extern void WDT_ISR(void) __attribute__((weak, alias("default_WDT_ISR")));
extern void RTC_TMR_ISR(void)
    __attribute__((weak, alias("default_RTC_TMR_ISR")));

extern void SW2_ISR(void) __attribute__((weak, alias("default_SW1_ISR")));
extern void SW3_ISR(void) __attribute__((weak, alias("default_SW1_ISR")));
extern void SW4_ISR(void) __attribute__((weak, alias("default_SW1_ISR")));
extern void SW5_ISR(void) __attribute__((weak, alias("default_SW1_ISR")));
extern void SW6_ISR(void) __attribute__((weak, alias("default_SW1_ISR")));

extern void IntDefaultHandler(void) __attribute__((weak));

//*****************************************************************************
//
// Reserve space for the system stack.
//
//*****************************************************************************
extern uint32_t _stack_top;
//__attribute__ ((section(".mainStack"), used))
// static uint32_t pui32Stack[2048];
#define STARTUP_STACK_TOP (&_stack_top)

//*****************************************************************************
// VECTOR TABLE
//*****************************************************************************
__attribute__((section(".vectors"), used)) void (*const gVectors[])(void) = {
    //(void (*)(void))((uint32_t)pui32Stack + sizeof(pui32Stack)), // Stack
    //pointer
    (void *)STARTUP_STACK_TOP,
    ResetISR,           // Reset handler
    NmiSR,              // The NMI handler
    FaultISR,           // The hard fault handler
    IntDefaultHandler,  // 4 The MPU fault handler
    IntDefaultHandler,  // 5 The bus fault handler
    IntDefaultHandler,  // 6 The usage fault handler
    0,                  // 7 Reserved
    0,                  // 8 Reserved
    0,                  // 9 Reserved
    0,                  // 10 Reserved
    SVCall_ISR,         // 11 SVCall handler
    DebugMonitor_ISR,   // 12 Debug monitor handler
    0,                  // 13 Reserved
    PENDSV_ISR,         // 14 The PendSV handler
    SYSTICK_ISR,        // 15 The SysTick handler

    // external interrupt service routines (ISR)
    GPIO0_ISR,    // 16 GPIO Port A            [ 0]
    GPIO1_ISR,    // 17 GPIO Port B            [ 1]
    TIMER0_ISR,   // 18 Timer 0                [ 2]
    TIMER1_ISR,   // 19 Timer 1                [ 3]
    UART0_ISR,    // 20 UART 0                 [ 4]
    UART1_ISR,    // 21 UART 1                 [ 5]
    SPI0_ISR,     // 22 SPI0                   [ 6]
    SPI1_ISR,     // 23 SPI1                   [ 7]
    I2C0_ISR,     // 24 I2C 0                  [ 8]
    I2C1_ISR,     // 25 I2C 1                  [ 9]
    RTC0_ISR,     // 26 RTC 0                  [10]
    RTC1_ISR,     // 27 RTC 1                  [11]
    DSP_ISR,      // 28 DSP MAILBOX            [12]
    ADC_ISR,      // 29 ADC                    [13]
    PWM_ISR,      // 32 PWM                    [14]
    WDT_ISR,      // 33 WDT                    [15]
    RTC_TMR_ISR,  // 34 RTC                    [16]

    SW0_ISR,  // 30 Software Interrupt 0   [17]
    SW1_ISR,  // 31 Software Interrupt 1   [18]
    SW2_ISR,  // 35 Software Interrupt 2   [19]
    SW3_ISR,  // 36 Software Interrupt 3   [20]
    SW4_ISR,  // 37 Software Interrupt 4   [21]
    SW5_ISR,  // 38 Software Interrupt 5   [22]
    SW6_ISR,  // 39 Software Interrupt 6   [23]

};

//*****************************************************************************
//
// The following are constructs created by the linker, indicating where the
// the "data" and "bss" segments reside in memory.  The initializers for the
// for the "data" segment resides immediately following the "text" segment.
//
//*****************************************************************************
extern uint32_t _etext;
extern uint32_t _eftext;
extern uint32_t _data;
extern uint32_t _edata;
extern uint32_t _bss;
extern uint32_t _ebss;

//
// And here are the weak interrupt handlers.
//
void default_NmiSR(void) {
  __asm("    movs     r0, #2");
  while (1) {
  }
}

void default_FaultISR(void) {
  __asm("    movs     r0, #3");
  MEMIO32(0x1001FFF0) = 0xbad0beef;  // near the top of 128KB of SRAM
  MEMIO32(0x1001FFF4) = 0xbad1beef;  // near the top of 128KB of SRAM
  while (1) {
    __asm("    BKPT      #1");
  }
}

void IntDefaultHandler(void) {
  __asm("    movs     r0, #20");
  while (1) {
    __asm("    BKPT      #1");
  }
}

void default_SVCall_ISR(void) {
  __asm("    movs     r0, #11");
  while (1) {
    __asm("    BKPT      #11");
  }
}

void default_DebugMonitor_ISR(void) {
  __asm("    movs     r0, #12");
  while (1) {
    __asm("    BKPT      #12");
  }
}

void default_PENDSV_ISR(void) {
  __asm("    movs     r0, #14");
  while (1) {
    __asm("    BKPT      #14");
  }
}

void default_SYSTICK_ISR(void) {
  __asm("    movs     r0, #15");
  while (1) {
    __asm("    BKPT      #15");
  }
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void default_SPI0_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_SPI1_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_I2C0_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_I2C1_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_UART0_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_UART1_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_GPIO0_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_GPIO1_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_ADC_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_DSP_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_TIMER0_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_TIMER1_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_RTC0_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_RTC1_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_PWM_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_WDT_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_RTC_TMR_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_SW0_ISR(void) {
  __asm("    movs     r0, #16");
  while (1) {
    __asm("    BKPT      #16");
  }
}

void default_SW1_ISR(void) {
  __asm("    movs     r0, #17");
  while (1) {
    __asm("    BKPT      #17");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Reset ISR
////////////////////////////////////////////////////////////////////////////////
void default_ResetISR(void) {
  int rc;
  bool bRunningInFlash;

  set_vtor();

  bRunningInFlash =
      ((((uint32_t)startup_get_my_pc()) & 0xFF000000) == 0x01000000);

  if ((!REG_RTC_AO_CSR.BF.WARM_START_MODE) || bRunningInFlash) {
    //
    //  Copy any .ro bytes to .data so that initialized global variables
    //  are actually properly initialized.
    //
    __asm(
        "    ldr      r0, =_eftext\n"
        "    ldr      r1, =_data\n"
        "    ldr      r2, =_edata\n"
        "ro_copy_loop:\n"
        "    ldr      r3, [r0], #4\n"
        "    str      r3, [r1], #4\n"
        "    cmp      r1, r2\n"
        "    ble      ro_copy_loop\n");

    //
    // Zero fill the .bss section.
    //
    __asm(
        "    ldr      r0, =_bss\n"
        "    ldr      r1, =_ebss\n"
        "    mov      r2, #0\n"
        "bss_zero_loop:\n"
        "    cmp      r0, r1\n"
        "    it       lt\n"
        "    strlt    r2, [r0], #4\n"
        "    blt      bss_zero_loop\n");
  }

  //
  // call the main routine barefoot, i.e. without the normal CRTC0 entry
  // point.
  //
  rc = _main(0, NULL);

  //
  //  If main ever returns, trap it here and wake up the debugger if it is
  //  connected.
  //
  while (1)  // for FPGA/real chip use
  {
    __asm("    BKPT      #1");
  }
}

////////////////////////////////////////////////////////////////////////////////
// get my PC
////////////////////////////////////////////////////////////////////////////////
void *startup_get_my_pc(void) {
  void *pc;
  asm("mov %0, pc" : "=r"(pc));
  return pc;
}

////////////////////////////////////////////////////////////////////////////////
// get my SP
////////////////////////////////////////////////////////////////////////////////
void *startup_get_my_sp(void) {
  void *sp;
  asm("mov %0, sp" : "=r"(sp));
  return sp;
}

////////////////////////////////////////////////////////////////////////////////
// Set VTOR based on PC
////////////////////////////////////////////////////////////////////////////////
void set_vtor(void) {
  __asm(
      "    ldr      r0, =0xe000ed08\n"
      "    ldr      r1, =0xFF000000\n"
      "    mov      r2, lr\n"
      "    and      r1, r2\n"
      "    str      r1, [r0]\n");

  return;
}
