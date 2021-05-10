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

/* This file is a modification of the Tensorflow Micro Lite file _main.c */

#include <stdint.h>

#include "am_bsp.h"
#include "am_mcu_apollo.h"  // Defines AM_CMSIS_REGS
#include "am_util.h"

#define ARM_MATH_CM4
#include <arm_math.h>

//*****************************************************************************
// Parameters
//
// Total number of bytes transferred = 320*50*2 = 32000
//*****************************************************************************

#define FRAME_SIZE 320  // Capture one 320-sample (20-ms) frame at a time
#define NUM_FRAMES 50   // Number of frames in 1 second

//*****************************************************************************
// GLOBALS
//*****************************************************************************

volatile int16_t g_numFramesCaptured = 0;
volatile bool g_bPDMDataReady = false;
int16_t
    captured_data[FRAME_SIZE * NUM_FRAMES];  // Location of 1-second data buffer
extern uint8_t g_silence_score;
extern uint8_t g_unknown_score;
extern uint8_t g_yes_score;
extern uint8_t g_no_score;
q7_t g_scores[4] = {0};

//*****************************************************************************
// The entry point for the application.
//*****************************************************************************
extern int main(int argc, char** argv);

void DebugLog(const char* s) { am_util_stdio_printf("%s", s); }
void DebugLogInt32(int32_t i) { am_util_stdio_printf("%d", i); }
void DebugLogUInt32(uint32_t i) { am_util_stdio_printf("%d", i); }
void DebugLogHex(uint32_t i) { am_util_stdio_printf("0x%8x", i); }
void DebugLogFloat(float i) { am_util_stdio_printf("%f", i); }

//*****************************************************************************
// PDM configuration information.
//*****************************************************************************
void* PDMHandle;

am_hal_pdm_config_t g_sPdmConfig = {
    .eClkDivider = AM_HAL_PDM_MCLKDIV_1,
    .eLeftGain = AM_HAL_PDM_GAIN_P225DB,
    .eRightGain = AM_HAL_PDM_GAIN_P225DB,
    .ui32DecimationRate =
        48,  // OSR = 1500/16 = 96 = 2*SINCRATE --> SINC_RATE = 48
    .bHighPassEnable = 0,
    .ui32HighPassCutoff = 0xB,
    .ePDMClkSpeed = AM_HAL_PDM_CLK_1_5MHZ,
    .bInvertI2SBCLK = 0,
    .ePDMClkSource = AM_HAL_PDM_INTERNAL_CLK,
    .bPDMSampleDelay = 0,
    .bDataPacking = 1,
    .ePCMChannels = AM_HAL_PDM_CHANNEL_RIGHT,
    .bLRSwap = 0,
};

//*****************************************************************************
// BUTTON0 pin configuration settings.
//*****************************************************************************
const am_hal_gpio_pincfg_t g_deepsleep_button0 = {
    .uFuncSel = 3,
    .eIntDir = AM_HAL_GPIO_PIN_INTDIR_LO2HI,
    .eGPInput = AM_HAL_GPIO_PIN_INPUT_ENABLE,
};

//*****************************************************************************
// PDM initialization.
//*****************************************************************************
void pdm_init(void) {
  //
  // Initialize, power-up, and configure the PDM.
  //
  am_hal_pdm_initialize(0, &PDMHandle);
  am_hal_pdm_power_control(PDMHandle, AM_HAL_PDM_POWER_ON, false);
  am_hal_pdm_configure(PDMHandle, &g_sPdmConfig);
  am_hal_pdm_enable(PDMHandle);

  //
  // Configure the necessary pins.
  //
  am_hal_gpio_pincfg_t sPinCfg = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // ARPIT 181019
  // sPinCfg.uFuncSel = AM_HAL_PIN_10_PDMCLK;
  // am_hal_gpio_pinconfig(10, sPinCfg);
  sPinCfg.uFuncSel = AM_HAL_PIN_12_PDMCLK;
  am_hal_gpio_pinconfig(12, sPinCfg);

  sPinCfg.uFuncSel = AM_HAL_PIN_11_PDMDATA;
  am_hal_gpio_pinconfig(11, sPinCfg);

  // am_hal_gpio_state_write(14, AM_HAL_GPIO_OUTPUT_CLEAR);
  // am_hal_gpio_pinconfig(14, g_AM_HAL_GPIO_OUTPUT);

  //
  // Configure and enable PDM interrupts (set up to trigger on DMA
  // completion).
  //
  am_hal_pdm_interrupt_enable(PDMHandle,
                              (AM_HAL_PDM_INT_DERR | AM_HAL_PDM_INT_DCMP |
                               AM_HAL_PDM_INT_UNDFL | AM_HAL_PDM_INT_OVF));

#if AM_CMSIS_REGS
  NVIC_EnableIRQ(PDM_IRQn);
#else
  am_hal_interrupt_enable(AM_HAL_INTERRUPT_PDM);
#endif
}

//*****************************************************************************
//
// Start a transaction to get some number of bytes from the PDM interface.
//
//*****************************************************************************
void pdm_data_get(void) {
  //
  // Configure DMA and target address.
  //
  am_hal_pdm_transfer_t sTransfer;
  sTransfer.ui32TargetAddr =
      (uint32_t)(&captured_data[FRAME_SIZE * g_numFramesCaptured]);
  sTransfer.ui32TotalCount = 2 * FRAME_SIZE;  // Each sample is 2 bytes

  //
  // Start the data transfer.
  //
  am_hal_pdm_dma_start(PDMHandle, &sTransfer);
}

//*****************************************************************************
//
// PDM interrupt handler.
//
//*****************************************************************************
void am_pdm0_isr(void) {
  uint32_t ui32Status;
  //
  // Read the interrupt status.
  //
  am_hal_pdm_interrupt_status_get(PDMHandle, &ui32Status, true);
  am_hal_pdm_interrupt_clear(PDMHandle, ui32Status);

  //
  // Once our DMA transaction completes, send a flag to the main routine
  //
  if (ui32Status & AM_HAL_PDM_INT_DCMP) g_bPDMDataReady = true;
}

//*****************************************************************************
// GPIO ISR
// Will enable the PDM, set number of frames transferred to 0, and turn on LED
//*****************************************************************************
void am_gpio_isr(void) {
  //
  // Delay for debounce.
  //
  am_util_delay_ms(200);

  //
  // Clear the GPIO Interrupt (write to clear).
  //
  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));

  // Start audio transfer
  am_hal_pdm_fifo_flush(PDMHandle);
  pdm_data_get();
  am_hal_pdm_enable(PDMHandle);

  //
  // Turn on LED 0
  //
  am_devices_led_on(am_bsp_psLEDs, 0);
}

int _main(void) {
  am_util_id_t sIdDevice;
  uint32_t ui32StrBuf;

  //
  // Set the clock frequency.
  //
  am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);

  //
  // Set the default cache configuration
  //
  am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
  am_hal_cachectrl_enable();

  //
  // Configure the board for low power operation.
  //
  am_bsp_low_power_init();

#if defined(AM_BSP_NUM_BUTTONS) && defined(AM_BSP_NUM_LEDS)
  //
  // Configure the button pin.
  //
  am_hal_gpio_pinconfig(AM_BSP_GPIO_BUTTON0, g_deepsleep_button0);

  //
  // Clear the GPIO Interrupt (write to clear).
  //
  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));

  //
  // Enable the GPIO/button interrupt.
  //
  am_hal_gpio_interrupt_enable(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));

  //
  // Configure the LEDs.
  //
  am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);

  //
  // Turn the LEDs off
  //
  for (int ix = 0; ix < AM_BSP_NUM_LEDS; ix++) {
    am_devices_led_off(am_bsp_psLEDs, ix);
  }

//    am_devices_led_on(am_bsp_psLEDs, 1);
#endif  // defined(AM_BSP_NUM_BUTTONS)  &&  defined(AM_BSP_NUM_LEDS)

#if AM_CMSIS_REGS
  NVIC_EnableIRQ(GPIO_IRQn);
#else   // AM_CMSIS_REGS
  am_hal_interrupt_enable(AM_HAL_INTERRUPT_GPIO);
#endif  // AM_CMSIS_REGS

  //
  // Enable interrupts to the core.
  //
  am_hal_interrupt_master_enable();

  // Turn on PDM
  pdm_init();

  //
  // Initialize the printf interface for UART output
  //
  am_bsp_uart_printf_enable();

  //
  // Print the banner.
  //
  am_util_stdio_terminal_clear();
  am_util_stdio_printf("Starting streaming test\n\n");

  // Score variables
  q7_t max_score = 0;
  uint32_t max_score_index = 0;

  while (1) {
    am_hal_interrupt_master_disable();

    if (g_bPDMDataReady) {
      g_bPDMDataReady = false;
      g_numFramesCaptured++;

      if (g_numFramesCaptured < NUM_FRAMES) {
        pdm_data_get();  // Start converting the next set of PCM samples.
      }

      else {
        g_numFramesCaptured = 0;
        // am_hal_pdm_disable(PDMHandle);
        am_devices_led_off(am_bsp_psLEDs, 0);

        main(0, NULL);

        g_scores[0] = (q7_t)g_silence_score - 128;
        g_scores[1] = (q7_t)g_unknown_score - 128;
        g_scores[2] = (q7_t)g_yes_score - 128;
        g_scores[3] = (q7_t)g_no_score - 128;

        am_devices_led_off(
            am_bsp_psLEDs,
            max_score_index + 1);  // Turn off LED for previous max score
        arm_max_q7(g_scores, 4, &max_score, &max_score_index);
        am_devices_led_on(
            am_bsp_psLEDs,
            max_score_index + 1);  // Turn on LED for new max score
      }
    }

    //
    // Go to Deep Sleep.
    //
    am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);

    am_hal_interrupt_master_enable();
  }

  // main(0, NULL);
}
