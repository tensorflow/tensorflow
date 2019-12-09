/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Apollo3 EVB specific features compile options:
// USE AM_BSP_NUM_LEDS : LED initialization and management per EVB target (# of
// LEDs defined in EVB BSP) USE_TIME_STAMP : Enable timers and time stamping for
// debug and performance profiling (customize per application) USE_DEBUG_GPIO :
// Enable GPIO flag polling for debug and performance profiling (customize per
// application) USE_MAYA : Enable specific pin configuration and features for
// AP3B "quarter" sized board

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"

#include <limits>

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

namespace {

// These are the raw buffers that are filled by the PDM during DMA
constexpr int kPdmNumSlots = 1;
constexpr int kPdmSamplesPerSlot = 256;
constexpr int kPdmSampleBufferSize = (kPdmNumSlots * kPdmSamplesPerSlot);
uint32_t g_ui32PDMSampleBuffer0[kPdmSampleBufferSize];
uint32_t g_ui32PDMSampleBuffer1[kPdmSampleBufferSize];
uint32_t g_PowerOff = 0;

// Controls the double buffering between the two DMA buffers.
int g_dma_destination_index = 0;
// PDM Device Handle.
static void* g_pdm_handle;
// PDM DMA error flag.
volatile bool g_pdm_dma_error;
// So the interrupt can use the passed-in error handler to report issues.
tflite::ErrorReporter* g_pdm_dma_error_reporter = nullptr;

// Holds a longer history of audio samples in a ring buffer.
constexpr int kAudioCaptureBufferSize = 16000;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize] = {};
int g_audio_capture_buffer_start = 0;
int64_t g_total_samples_captured = 0;
int32_t g_latest_audio_timestamp = 0;

// Copy of audio samples returned to the caller.
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
bool g_is_audio_initialized = false;

//*****************************************************************************
//
// Globals
//
//*****************************************************************************
#if USE_TIME_STAMP
// Select the CTIMER number to use for timing.
// The entire 32-bit timer is used.
#define SELFTEST_TIMERNUM 0

// Timer configuration.
static am_hal_ctimer_config_t g_sContTimer = {
    // Create 32-bit timer
    1,

    // Set up TimerA.
    (AM_HAL_CTIMER_FN_CONTINUOUS | AM_HAL_CTIMER_HFRC_12MHZ),

    // Set up Timer0B.
    0};

#endif  // USE_TIME_STAMP

// ARPIT TODO : Implement low power configuration
void custom_am_bsp_low_power_init(void) {
#if USE_MAYA
  // Make sure SWO/ITM/TPIU is disabled.
  // SBL may not get it completely shut down.
  am_bsp_itm_printf_disable();
#else
  // Initialize the printf interface for AP3B ITM/SWO output.
  am_bsp_itm_printf_enable();
#endif

  // Initialize for low power in the power control block
  // am_hal_pwrctrl_low_power_init();

  // Run the RTC off the LFRC.
  // am_hal_rtc_osc_select(AM_HAL_RTC_OSC_LFRC);

  // Stop the XTAL.
  // am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_XTAL_STOP, 0);

  // Disable the RTC.
  // am_hal_rtc_osc_disable();

#ifdef AM_BSP_NUM_LEDS
  //
  // Initialize the LEDs.
  // On the apollo3_evb, when the GPIO outputs are disabled (the default at
  // power up), the FET gates are floating and
  // partially illuminating the LEDs.
  //
  uint32_t ux, ui32GPIONumber;
  for (ux = 0; ux < AM_BSP_NUM_LEDS; ux++) {
    ui32GPIONumber = am_bsp_psLEDs[ux].ui32GPIONumber;

    //
    // Configure the pin as a push-pull GPIO output
    // (aka AM_DEVICES_LED_POL_DIRECT_DRIVE_M).
    //
    am_hal_gpio_pinconfig(ui32GPIONumber, g_AM_HAL_GPIO_OUTPUT);

    //
    // Turn off the LED.
    //
    am_hal_gpio_state_write(ui32GPIONumber,
                            AM_HAL_GPIO_OUTPUT_TRISTATE_DISABLE);
    am_hal_gpio_state_write(ui32GPIONumber, AM_HAL_GPIO_OUTPUT_CLEAR);
  }
#endif  // AM_BSP_NUM_LEDS

}  // am_bsp_low_power_init()

// Make sure the CPU is running as fast as possible.
void enable_burst_mode(tflite::ErrorReporter* error_reporter) {
  am_hal_burst_avail_e eBurstModeAvailable;
  am_hal_burst_mode_e eBurstMode;

  // Check that the Burst Feature is available.
  if (AM_HAL_STATUS_SUCCESS ==
      am_hal_burst_mode_initialize(&eBurstModeAvailable)) {
    if (AM_HAL_BURST_AVAIL == eBurstModeAvailable) {
      error_reporter->Report("Apollo3 Burst Mode is Available\n");
    } else {
      error_reporter->Report("Apollo3 Burst Mode is Not Available\n");
    }
  } else {
    error_reporter->Report("Failed to Initialize for Burst Mode operation\n");
  }

  // Put the MCU into "Burst" mode.
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_enable(&eBurstMode)) {
    if (AM_HAL_BURST_MODE == eBurstMode) {
      error_reporter->Report("Apollo3 operating in Burst Mode (96MHz)\n");
    }
  } else {
    error_reporter->Report("Failed to Enable Burst Mode operation\n");
  }
}

}  // namespace

//*****************************************************************************
// PDM configuration information.
//*****************************************************************************
am_hal_pdm_config_t g_sPdmConfig = {
    .eClkDivider = AM_HAL_PDM_MCLKDIV_1,
    .eLeftGain = AM_HAL_PDM_GAIN_P165DB,
    .eRightGain = AM_HAL_PDM_GAIN_P165DB,
    .ui32DecimationRate =
        48,  // OSR = 1500/16 = 96 = 2*SINCRATE --> SINC_RATE = 48
    .bHighPassEnable = 1,
    .ui32HighPassCutoff = 0x2,
    .ePDMClkSpeed = AM_HAL_PDM_CLK_1_5MHZ,
    .bInvertI2SBCLK = 0,
    .ePDMClkSource = AM_HAL_PDM_INTERNAL_CLK,
    .bPDMSampleDelay = 0,
    .bDataPacking = 0,
    .ePCMChannels = AM_HAL_PDM_CHANNEL_LEFT,
    .ui32GainChangeDelay = 1,
    .bI2SEnable = 0,
    .bSoftMute = 0,
    .bLRSwap = 0,
};

//*****************************************************************************
// PDM initialization.
//*****************************************************************************
extern "C" void pdm_init(void) {
  //
  // Initialize, power-up, and configure the PDM.
  //
  am_hal_pdm_initialize(0, &g_pdm_handle);
  am_hal_pdm_power_control(g_pdm_handle, AM_HAL_PDM_POWER_ON, false);
  am_hal_pdm_configure(g_pdm_handle, &g_sPdmConfig);

  //
  // Configure the necessary pins.
  //
  am_hal_gpio_pincfg_t sPinCfg = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  //
  // AP3B EVB w/ PDM MIC in slot3
  //
  sPinCfg.uFuncSel = AM_HAL_PIN_12_PDMCLK;
  am_hal_gpio_pinconfig(12, sPinCfg);

  sPinCfg.uFuncSel = AM_HAL_PIN_11_PDMDATA;
  am_hal_gpio_pinconfig(11, sPinCfg);

  //
  // Configure and enable PDM interrupts (set up to trigger on DMA
  // completion).
  //
  am_hal_pdm_interrupt_enable(g_pdm_handle,
                              (AM_HAL_PDM_INT_DERR | AM_HAL_PDM_INT_DCMP |
                               AM_HAL_PDM_INT_UNDFL | AM_HAL_PDM_INT_OVF));

  NVIC_EnableIRQ(PDM_IRQn);

  // Enable PDM
  am_hal_pdm_enable(g_pdm_handle);
}

// Start the DMA fetch of PDM samples.
void pdm_start_dma(tflite::ErrorReporter* error_reporter) {
  // Configure DMA and target address.
  am_hal_pdm_transfer_t sTransfer;

  if (g_dma_destination_index == 0) {
    sTransfer.ui32TargetAddr = (uint32_t)g_ui32PDMSampleBuffer0;
  } else {
    sTransfer.ui32TargetAddr = (uint32_t)g_ui32PDMSampleBuffer1;
  }

  sTransfer.ui32TotalCount = 4 * kPdmSampleBufferSize;
  // PDM DMA count is in Bytes

  // Start the data transfer.
  if (AM_HAL_STATUS_SUCCESS != am_hal_pdm_dma_start(g_pdm_handle, &sTransfer)) {
    error_reporter->Report("Error - configuring PDM DMA failed.");
  }

  // Reset the PDM DMA flags.
  g_pdm_dma_error = false;
  g_pdm_dma_error_reporter = error_reporter;
}

#if USE_MAYA
extern "C" void power_down_sequence(void) {
  am_hal_gpio_read_type_e eReadType;
  eReadType = AM_HAL_GPIO_INPUT_READ;

  // Reconfigure PDM Pins for low power
  // Drive PDMCLK low so Mics go in standby mode of ~ 10 to 20uA each
  am_hal_gpio_pinconfig(12, g_AM_HAL_GPIO_OUTPUT);
  am_hal_gpio_state_write(12, AM_HAL_GPIO_OUTPUT_SET);

  // Disable PDMDATA pin so no input buffer leakage current from floating pin
  am_hal_gpio_pinconfig(11, g_AM_HAL_GPIO_DISABLE);

  // Disable PDM
  am_hal_pdm_disable(g_pdm_handle);
  am_hal_pdm_power_control(g_pdm_handle, AM_HAL_PDM_POWER_OFF, false);
  am_hal_interrupt_master_disable();

  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));
  am_hal_gpio_interrupt_disable(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));
  am_util_delay_ms(200);  // Debounce Delay
  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));
  am_hal_gpio_interrupt_enable(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON0));

  for (int ix = 0; ix < AM_BSP_NUM_LEDS; ix++) {
    am_devices_led_off(am_bsp_psLEDs, ix);
  }

  am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
  // Apollo3 will be < 3uA in deep sleep

  am_hal_reset_control(AM_HAL_RESET_CONTROL_SWPOR, 0);
  // Use Reset to perform clean power-on from sleep
}

//*****************************************************************************
//
// GPIO ISR
//
//*****************************************************************************
extern "C" void am_gpio_isr(void) {
  uint64_t ui64Status;
  // Read and clear the GPIO interrupt status then service the interrupts.
  am_hal_gpio_interrupt_status_get(false, &ui64Status);
  am_hal_gpio_interrupt_clear(ui64Status);
  am_hal_gpio_interrupt_service(ui64Status);
}

extern "C" void power_button_handler(void) { g_PowerOff = 1; }

#endif  // USE_MAYA

// Interrupt handler for the PDM.
extern "C" void am_pdm0_isr(void) {
  uint32_t ui32IntMask;

  // Read the interrupt status.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_pdm_interrupt_status_get(g_pdm_handle, &ui32IntMask, false)) {
    g_pdm_dma_error_reporter->Report("Error reading PDM0 interrupt status.");
  }

  // Clear the PDM interrupt.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_pdm_interrupt_clear(g_pdm_handle, ui32IntMask)) {
    g_pdm_dma_error_reporter->Report("Error clearing PDM interrupt status.");
  }

#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
  am_hal_gpio_state_write(31, AM_HAL_GPIO_OUTPUT_SET);  // Slot1 AN pin
#endif

  // If we got a DMA complete, set the flag.
  if (ui32IntMask & AM_HAL_PDM_INT_OVF) {
    am_util_stdio_printf("\n%s\n", "\nPDM ISR OVF.");
  }
  if (ui32IntMask & AM_HAL_PDM_INT_UNDFL) {
    am_util_stdio_printf("\n%s\n", "\nPDM ISR UNDLF.");
  }
  if (ui32IntMask & AM_HAL_PDM_INT_DCMP) {
    uint32_t* source_buffer;
    if (g_dma_destination_index == 0) {
      source_buffer = g_ui32PDMSampleBuffer0;
      g_dma_destination_index = 1;
    } else {
      source_buffer = g_ui32PDMSampleBuffer1;
      g_dma_destination_index = 0;
    }
    pdm_start_dma(g_pdm_dma_error_reporter);

    uint32_t slotCount = 0;
    for (uint32_t indi = 0; indi < kPdmSampleBufferSize; indi++) {
      g_audio_capture_buffer[g_audio_capture_buffer_start] =
          source_buffer[indi];
      g_audio_capture_buffer_start =
          (g_audio_capture_buffer_start + 1) % kAudioCaptureBufferSize;
      slotCount++;
    }

    g_total_samples_captured += slotCount;
    g_latest_audio_timestamp =
        (g_total_samples_captured / (kAudioSampleFrequency / 1000));
  }

  // If we got a DMA error, set the flag.
  if (ui32IntMask & AM_HAL_PDM_INT_DERR) {
    g_pdm_dma_error = true;
  }

#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
  am_hal_gpio_state_write(31, AM_HAL_GPIO_OUTPUT_CLEAR);  // Slot1 AN pin
#endif
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  // Set the clock frequency.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0)) {
    error_reporter->Report("Error - configuring the system clock failed.");
    return kTfLiteError;
  }

  // Individually select elements of am_bsp_low_power_init
  custom_am_bsp_low_power_init();

  // Set the default cache configuration and enable it.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_cachectrl_config(&am_hal_cachectrl_defaults)) {
    error_reporter->Report("Error - configuring the system cache failed.");
    return kTfLiteError;
  }
  if (AM_HAL_STATUS_SUCCESS != am_hal_cachectrl_enable()) {
    error_reporter->Report("Error - enabling the system cache failed.");
    return kTfLiteError;
  }

  // Configure Flash wait states.
  CACHECTRL->FLASHCFG_b.RD_WAIT = 1;      // Default is 3
  CACHECTRL->FLASHCFG_b.SEDELAY = 6;      // Default is 7
  CACHECTRL->FLASHCFG_b.LPM_RD_WAIT = 5;  // Default is 8

  // Enable cache sleep states.
  uint32_t ui32LPMMode = CACHECTRL_FLASHCFG_LPMMODE_STANDBY;
  if (am_hal_cachectrl_control(AM_HAL_CACHECTRL_CONTROL_LPMMODE_SET,
                               &ui32LPMMode)) {
    error_reporter->Report("Error - enabling cache sleep state failed.");
  }

  // Enable Instruction & Data pre-fetching.
  MCUCTRL->SRAMMODE_b.DPREFETCH = 1;
  MCUCTRL->SRAMMODE_b.DPREFETCH_CACHE = 1;
  MCUCTRL->SRAMMODE_b.IPREFETCH = 1;
  MCUCTRL->SRAMMODE_b.IPREFETCH_CACHE = 1;

  // Enable the floating point module, and configure the core for lazy stacking.
  am_hal_sysctrl_fpu_enable();
  am_hal_sysctrl_fpu_stacking_enable(true);
  error_reporter->Report("FPU Enabled.");

  // Configure the LEDs.
  am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
  // Turn the LEDs off
  for (int ix = 0; ix < AM_BSP_NUM_LEDS; ix++) {
    am_devices_led_off(am_bsp_psLEDs, ix);
  }

#if USE_MAYA
  // Configure Power Button
  am_hal_gpio_pinconfig(AM_BSP_GPIO_BUTTON_POWER, g_AM_BSP_GPIO_BUTTON_POWER);

  // Clear and Enable the GPIO Interrupt (write to clear).
  am_hal_gpio_interrupt_clear(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON_POWER));
  am_hal_gpio_interrupt_register(AM_BSP_GPIO_BUTTON_POWER,
                                 power_button_handler);
  am_hal_gpio_interrupt_enable(AM_HAL_GPIO_BIT(AM_BSP_GPIO_BUTTON_POWER));

  // Enable GPIO interrupts to the NVIC.
  NVIC_EnableIRQ(GPIO_IRQn);
#endif  // USE_MAYA

#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
  // Configure the GPIOs for flag polling.
  am_hal_gpio_pinconfig(31, g_AM_HAL_GPIO_OUTPUT);  // Slot1 AN pin
  am_hal_gpio_pinconfig(39, g_AM_HAL_GPIO_OUTPUT);  // Slot1 RST pin
  am_hal_gpio_pinconfig(44, g_AM_HAL_GPIO_OUTPUT);  // Slot1 CS pin
  am_hal_gpio_pinconfig(48, g_AM_HAL_GPIO_OUTPUT);  // Slot1 PWM pin

  am_hal_gpio_pinconfig(32, g_AM_HAL_GPIO_OUTPUT);  // Slot2 AN pin
  am_hal_gpio_pinconfig(46, g_AM_HAL_GPIO_OUTPUT);  // Slot2 RST pin
  am_hal_gpio_pinconfig(42, g_AM_HAL_GPIO_OUTPUT);  // Slot2 CS pin
  am_hal_gpio_pinconfig(47, g_AM_HAL_GPIO_OUTPUT);  // Slot2 PWM pin
#endif

  // Ensure the CPU is running as fast as possible.
  // enable_burst_mode(error_reporter);

#if USE_TIME_STAMP
  //
  // Set up and start the timer.
  //
  am_hal_ctimer_stop(SELFTEST_TIMERNUM, AM_HAL_CTIMER_BOTH);
  am_hal_ctimer_clear(SELFTEST_TIMERNUM, AM_HAL_CTIMER_BOTH);
  am_hal_ctimer_config(SELFTEST_TIMERNUM, &g_sContTimer);
  am_hal_ctimer_start(SELFTEST_TIMERNUM, AM_HAL_CTIMER_TIMERA);
#endif  // USE_TIME_STAMP

  // Configure, turn on PDM
  pdm_init();
  am_hal_interrupt_master_enable();
  am_hal_pdm_fifo_flush(g_pdm_handle);
  // Trigger the PDM DMA for the first time manually.
  pdm_start_dma(g_pdm_dma_error_reporter);

  error_reporter->Report("\nPDM DMA Threshold = %d", PDMn(0)->FIFOTHR);

  // Turn on LED 0 to indicate PDM initialized
  am_devices_led_on(am_bsp_psLEDs, 0);

  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
#if USE_MAYA
  if (g_PowerOff) {
    power_down_sequence();
  }
#endif  // USE_MAYA

  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }

#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
  am_hal_gpio_state_write(39, AM_HAL_GPIO_OUTPUT_SET);  // Slot1 RST pin
#endif

  // This should only be called when the main thread notices that the latest
  // audio sample data timestamp has changed, so that there's new data in the
  // capture ring buffer. The ring buffer will eventually wrap around and
  // overwrite the data, but the assumption is that the main thread is checking
  // often enough and the buffer is large enough that this call will be made
  // before that happens.
  const int start_offset =
      (start_ms < 0) ? 0 : start_ms * (kAudioSampleFrequency / 1000);
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < duration_sample_count; ++i) {
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;

#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
  am_hal_gpio_state_write(39, AM_HAL_GPIO_OUTPUT_CLEAR);  // Slot1 RST pin
#endif

  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
