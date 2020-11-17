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

#if defined(ARDUINO)
#if defined(ARDUINO_SFE_EDGE)
#define USE_ANALOG_MIC_SFE_BSPS
#else
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO_SFE_EDGE)
#endif  // defined(ARDUINO)

#ifndef ARDUINO_EXCLUDE_CODE

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"

#include <limits>

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

#if defined(TFLU_APOLLO3_BOARD_sparkfun_edge)
#define USE_ANALOG_MIC_SFE_BSPS
#else
#define USE_PDM_MIC
#endif // defined(TFLU_APOLLO3_BOARD_sparkfun_edge)


/**************************************************************
PDM Mic Implementation
**************************************************************/
#if defined(USE_PDM_MIC)

namespace {

// These are the raw buffers that are filled by the PDM during DMA
constexpr int kPdmNumSlots = 1;
constexpr int kPdmSamplesPerSlot = 256;
constexpr int kPdmSampleBufferSize = (kPdmNumSlots * kPdmSamplesPerSlot);
uint32_t g_ui32PDMSampleBuffer0[kPdmSampleBufferSize];
uint32_t g_ui32PDMSampleBuffer1[kPdmSampleBufferSize];

// Controls the double buffering between the two DMA buffers.
int g_dma_destination_index = 0;
static void* g_pdm_handle;                                  // PDM Device Handle.
volatile bool g_pdm_dma_error;                              // PDM DMA error flag.
tflite::ErrorReporter* g_pdm_dma_error_reporter = nullptr;  // So the interrupt can use the passed-in
                                                            // error handler to report issues.

// Holds a longer history of audio samples in a ring buffer.
constexpr int kAudioCaptureBufferSize = 15*1024;            // must be multiple of 1024 for alignment
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

static am_hal_ctimer_config_t g_sContTimer = {                // Timer configuration.
    1,                                                        //  Create 32-bit timer
    (AM_HAL_CTIMER_FN_CONTINUOUS | AM_HAL_CTIMER_HFRC_12MHZ), //  Set up TimerA.
    0                                                         //  Set up Timer0B.
  };
#endif  // USE_TIME_STAMP

// Make sure the CPU is running as fast as possible.
void enable_burst_mode(tflite::ErrorReporter* error_reporter) {
  am_hal_burst_avail_e eBurstModeAvailable;
  am_hal_burst_mode_e eBurstMode;

  // Check that the Burst Feature is available.
  if (AM_HAL_STATUS_SUCCESS ==
      am_hal_burst_mode_initialize(&eBurstModeAvailable)) {
    if (AM_HAL_BURST_AVAIL == eBurstModeAvailable) {
      TF_LITE_REPORT_ERROR(error_reporter, "Apollo3 Burst Mode is Available\n");
    } else {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Apollo3 Burst Mode is Not Available\n");
    }
  } else {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to Initialize for Burst Mode operation\n");
  }

  // Put the MCU into "Burst" mode.
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_enable(&eBurstMode)) {
    if (AM_HAL_BURST_MODE == eBurstMode) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Apollo3 operating in Burst Mode (96MHz)\n");
    }
  } else {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to Enable Burst Mode operation\n");
  }
}

}  // namespace

//*****************************************************************************
// PDM configuration information.
//*****************************************************************************
am_hal_pdm_config_t g_sPdmConfig = {
    .eClkDivider = AM_HAL_PDM_MCLKDIV_1,
    .eLeftGain = AM_HAL_PDM_GAIN_0DB,
    .eRightGain = AM_HAL_PDM_GAIN_0DB,
    .ui32DecimationRate =
        47,  // OSR = 1500/16 = 96 = 2*SINCRATE --> SINC_RATE = 47
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

#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
  // Configure the necessary pins.
  // AP3B EVB w/ PDM MIC in slot3
  am_hal_gpio_pincfg_t sPinCfg = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  sPinCfg.uFuncSel = AM_HAL_PIN_12_PDMCLK;
  am_hal_gpio_pinconfig(12, sPinCfg);
  sPinCfg.uFuncSel = AM_HAL_PIN_11_PDMDATA;
  am_hal_gpio_pinconfig(11, sPinCfg);

  // power
  am_hal_gpio_state_write(14, AM_HAL_GPIO_OUTPUT_CLEAR);
  am_hal_gpio_pinconfig(14, g_AM_HAL_GPIO_OUTPUT);
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)

  // Configure and enable PDM interrupts (set up to trigger on DMA completion).
  am_hal_pdm_interrupt_enable(g_pdm_handle,
                              (AM_HAL_PDM_INT_DERR | AM_HAL_PDM_INT_DCMP |
                               AM_HAL_PDM_INT_UNDFL | AM_HAL_PDM_INT_OVF));

  // Enable PDM
  am_hal_pdm_enable(g_pdm_handle);
  NVIC_EnableIRQ(PDM_IRQn);
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
    TF_LITE_REPORT_ERROR(error_reporter, "Error - configuring PDM DMA failed.");
  }

  // Reset the PDM DMA flags.
  g_pdm_dma_error = false;
}

// Interrupt handler for the PDM.
extern "C" void am_pdm0_isr(void) {
  uint32_t ui32IntMask;

  // Read the interrupt status.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_pdm_interrupt_status_get(g_pdm_handle, &ui32IntMask, false)) {
    TF_LITE_REPORT_ERROR(g_pdm_dma_error_reporter,
                         "Error reading PDM0 interrupt status.");
  }

  // Clear the PDM interrupt.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_pdm_interrupt_clear(g_pdm_handle, ui32IntMask)) {
    TF_LITE_REPORT_ERROR(g_pdm_dma_error_reporter,
                         "Error clearing PDM interrupt status.");
  }

#if USE_DEBUG_GPIO
  // DEBUG : GPIO flag polling.
#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
  am_hal_gpio_state_write(31, AM_HAL_GPIO_OUTPUT_SET);  // Slot1 AN pin
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)
#endif // USE_DEBUG_GPIO

  // If we got a DMA complete, set the flag.
  if (ui32IntMask & AM_HAL_PDM_INT_OVF) {
    TF_LITE_REPORT_ERROR(g_pdm_dma_error_reporter, "\n\nPDM ISR OVF.\n");
  }
  if (ui32IntMask & AM_HAL_PDM_INT_UNDFL) {
    TF_LITE_REPORT_ERROR(g_pdm_dma_error_reporter, "\n\nPDM ISR UNDLF.\n");
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
#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
  // DEBUG : GPIO flag polling.
  am_hal_gpio_state_write(31, AM_HAL_GPIO_OUTPUT_CLEAR);  // Slot1 AN pin
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)
#endif // USE_DEBUG_GPIO
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  // Set the clock frequency.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - configuring the system clock failed.");
    return kTfLiteError;
  }

  // Set the default cache configuration and enable it.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_cachectrl_config(&am_hal_cachectrl_defaults)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - configuring the system cache failed.");
    return kTfLiteError;
  }
  if (AM_HAL_STATUS_SUCCESS != am_hal_cachectrl_enable()) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - enabling the system cache failed.");
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
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - enabling cache sleep state failed.");
  }

  // Enable Instruction & Data pre-fetching.
  MCUCTRL->SRAMMODE_b.DPREFETCH = 1;
  MCUCTRL->SRAMMODE_b.DPREFETCH_CACHE = 1;
  MCUCTRL->SRAMMODE_b.IPREFETCH = 1;
  MCUCTRL->SRAMMODE_b.IPREFETCH_CACHE = 1;

  // Enable the floating point module, and configure the core for lazy stacking.
  am_hal_sysctrl_fpu_enable();
  am_hal_sysctrl_fpu_stacking_enable(true);
  TF_LITE_REPORT_ERROR(error_reporter, "FPU Enabled.");

  // Ensure the CPU is running as fast as possible.
  enable_burst_mode(error_reporter);

#if USE_TIME_STAMP
  // Set up and start the timer.
  am_hal_ctimer_stop(SELFTEST_TIMERNUM, AM_HAL_CTIMER_BOTH);
  am_hal_ctimer_clear(SELFTEST_TIMERNUM, AM_HAL_CTIMER_BOTH);
  am_hal_ctimer_config(SELFTEST_TIMERNUM, &g_sContTimer);
  am_hal_ctimer_start(SELFTEST_TIMERNUM, AM_HAL_CTIMER_TIMERA);
#endif  // USE_TIME_STAMP

  // Configure, turn on PDM
  g_pdm_dma_error_reporter = error_reporter;
  pdm_init();
  am_hal_interrupt_master_enable();
  am_hal_pdm_fifo_flush(g_pdm_handle);
  // Trigger the PDM DMA for the first time manually.
  pdm_start_dma(error_reporter);

  TF_LITE_REPORT_ERROR(error_reporter, "\nPDM DMA Threshold = %d",
                       PDMn(0)->FIFOTHR);

  // Turn on LED 0 to indicate PDM initialized
  am_devices_led_on(am_bsp_psLEDs, 0);

  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }

#if USE_DEBUG_GPIO
#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
  am_hal_gpio_state_write(39, AM_HAL_GPIO_OUTPUT_SET);  // Slot1 RST pin
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)
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
#if defined(TFLU_APOLLO3_BOARD_apollo3evb)
  am_hal_gpio_state_write(39, AM_HAL_GPIO_OUTPUT_CLEAR);  // Slot1 RST pin
#endif // defined(TFLU_APOLLO3_BOARD_apollo3evb)
#endif

  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }

#endif // defined(USE_PDM_MIC)


/**************************************************************
SparkFun Edge Analog Mic Implementation
**************************************************************/
#if defined(USE_ANALOG_MIC_SFE_BSPS)
namespace {

// These are the raw buffers that are filled by the ADC during DMA
constexpr int kAdcNumSlots = 2;
constexpr int kAdcSamplesPerSlot = 1024;
constexpr int kAdcSampleBufferSize = (kAdcNumSlots * kAdcSamplesPerSlot);
uint32_t g_ui32ADCSampleBuffer0[kAdcSampleBufferSize];
uint32_t g_ui32ADCSampleBuffer1[kAdcSampleBufferSize];
// Controls the double buffering between the two DMA buffers.
int g_dma_destination_index = 0;
// ADC Device Handle.
static void* g_adc_handle;
// ADC DMA error flag.
volatile bool g_adc_dma_error;
// So the interrupt can use the passed-in error handler to report issues.
tflite::ErrorReporter* g_adc_dma_error_reporter = nullptr;

// Holds a longer history of audio samples in a ring buffer.
constexpr int kAudioCaptureBufferSize = 16000;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize] = {};
int g_audio_capture_buffer_start = 0;
int64_t g_total_samples_captured = 0;
int32_t g_latest_audio_timestamp = 0;

// Copy of audio samples returned to the caller.
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
bool g_is_audio_initialized = false;

// Start the DMA fetch of ADC samples.
void adc_start_dma(tflite::ErrorReporter* error_reporter) {
  am_hal_adc_dma_config_t ADCDMAConfig;

  // Configure the ADC to use DMA for the sample transfer.
  ADCDMAConfig.bDynamicPriority = true;
  ADCDMAConfig.ePriority = AM_HAL_ADC_PRIOR_SERVICE_IMMED;
  ADCDMAConfig.bDMAEnable = true;
  ADCDMAConfig.ui32SampleCount = kAdcSampleBufferSize;
  if (g_dma_destination_index == 0) {
    ADCDMAConfig.ui32TargetAddress = (uint32_t)g_ui32ADCSampleBuffer0;
  } else {
    ADCDMAConfig.ui32TargetAddress = (uint32_t)g_ui32ADCSampleBuffer1;
  }
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_adc_configure_dma(g_adc_handle, &ADCDMAConfig)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error - configuring ADC DMA failed.");
  }

  // Reset the ADC DMA flags.
  g_adc_dma_error = false;
  g_adc_dma_error_reporter = error_reporter;
}

// Configure the ADC.
void adc_config0(tflite::ErrorReporter* error_reporter) {
  am_hal_adc_config_t ADCConfig;
  am_hal_adc_slot_config_t ADCSlotConfig;

  // Initialize the ADC and get the handle.
  if (AM_HAL_STATUS_SUCCESS != am_hal_adc_initialize(0, &g_adc_handle)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - reservation of the ADC0 instance failed.");
  }

  // Power on the ADC.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_adc_power_control(g_adc_handle, AM_HAL_SYSCTRL_WAKE, false)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error - ADC0 power on failed.");
  }

  // Set up the ADC configuration parameters. These settings are reasonable
  // for accurate measurements at a low sample rate.
  ADCConfig.eClock = AM_HAL_ADC_CLKSEL_HFRC_DIV2;
  ADCConfig.ePolarity = AM_HAL_ADC_TRIGPOL_RISING;
  ADCConfig.eTrigger = AM_HAL_ADC_TRIGSEL_SOFTWARE;
  ADCConfig.eReference =
      AM_HAL_ADC_REFSEL_INT_2P0;  // AM_HAL_ADC_REFSEL_INT_1P5;
  ADCConfig.eClockMode = AM_HAL_ADC_CLKMODE_LOW_LATENCY;
  ADCConfig.ePowerMode = AM_HAL_ADC_LPMODE0;
  ADCConfig.eRepeat = AM_HAL_ADC_REPEATING_SCAN;
  if (AM_HAL_STATUS_SUCCESS != am_hal_adc_configure(g_adc_handle, &ADCConfig)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error - configuring ADC0 failed.");
  }

  // Set up an ADC slot (2)
  ADCSlotConfig.eMeasToAvg = AM_HAL_ADC_SLOT_AVG_1;
  ADCSlotConfig.ePrecisionMode = AM_HAL_ADC_SLOT_14BIT;
  ADCSlotConfig.eChannel = AM_HAL_ADC_SLOT_CHSEL_SE2;
  ADCSlotConfig.bWindowCompare = false;
  ADCSlotConfig.bEnabled = true;
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_adc_configure_slot(g_adc_handle, 2, &ADCSlotConfig)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - configuring ADC Slot 2 failed.");
  }

  // Set up an ADC slot (1)
  ADCSlotConfig.eMeasToAvg = AM_HAL_ADC_SLOT_AVG_1;
  ADCSlotConfig.ePrecisionMode = AM_HAL_ADC_SLOT_14BIT;
  ADCSlotConfig.eChannel = AM_HAL_ADC_SLOT_CHSEL_SE1;
  ADCSlotConfig.bWindowCompare = false;
  ADCSlotConfig.bEnabled = true;
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_adc_configure_slot(g_adc_handle, 1, &ADCSlotConfig)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - configuring ADC Slot 1 failed.");
  }

  // Configure the ADC to use DMA for the sample transfer.
  adc_start_dma(error_reporter);

  // For this example, the samples will be coming in slowly. This means we
  // can afford to wake up for every conversion.
  am_hal_adc_interrupt_enable(g_adc_handle,
                              AM_HAL_ADC_INT_DERR | AM_HAL_ADC_INT_DCMP);

  // Enable the ADC.
  if (AM_HAL_STATUS_SUCCESS != am_hal_adc_enable(g_adc_handle)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error - enabling ADC0 failed.");
  }
}

// Initialize the ADC repetitive sample timer A3.
void init_timerA3_for_ADC() {
  // Start a timer to trigger the ADC periodically (1 second).
  am_hal_ctimer_config_single(3, AM_HAL_CTIMER_TIMERA,
                              AM_HAL_CTIMER_HFRC_12MHZ |
                                  AM_HAL_CTIMER_FN_REPEAT |
                                  AM_HAL_CTIMER_INT_ENABLE);

  am_hal_ctimer_int_enable(AM_HAL_CTIMER_INT_TIMERA3);

  // 750 = 12,000,000 (clock rate) / 16,000 (desired sample rate).
  am_hal_ctimer_period_set(3, AM_HAL_CTIMER_TIMERA, 750, 374);

  // Enable the timer A3 to trigger the ADC directly
  am_hal_ctimer_adc_trigger_enable();

  // Start the timer.
  am_hal_ctimer_start(3, AM_HAL_CTIMER_TIMERA);
}

// Make sure the CPU is running as fast as possible.
void enable_burst_mode(tflite::ErrorReporter* error_reporter) {
  am_hal_burst_avail_e eBurstModeAvailable;
  am_hal_burst_mode_e eBurstMode;

  // Check that the Burst Feature is available.
  if (AM_HAL_STATUS_SUCCESS ==
      am_hal_burst_mode_initialize(&eBurstModeAvailable)) {
    if (AM_HAL_BURST_AVAIL == eBurstModeAvailable) {
      TF_LITE_REPORT_ERROR(error_reporter, "Apollo3 Burst Mode is Available\n");
    } else {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Apollo3 Burst Mode is Not Available\n");
    }
  } else {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to Initialize for Burst Mode operation\n");
  }

  // Put the MCU into "Burst" mode.
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_enable(&eBurstMode)) {
    if (AM_HAL_BURST_MODE == eBurstMode) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Apollo3 operating in Burst Mode (96MHz)\n");
    }
  } else {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to Enable Burst Mode operation\n");
  }
}

}  // namespace

// Interrupt handler for the ADC.
extern "C" void am_adc_isr(void) {
  uint32_t ui32IntMask;

  // Read the interrupt status.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_adc_interrupt_status(g_adc_handle, &ui32IntMask, false)) {
    TF_LITE_REPORT_ERROR(g_adc_dma_error_reporter,
                         "Error reading ADC0 interrupt status.");
  }

  // Clear the ADC interrupt.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_adc_interrupt_clear(g_adc_handle, ui32IntMask)) {
    TF_LITE_REPORT_ERROR(g_adc_dma_error_reporter,
                         "Error clearing ADC0 interrupt status.");
  }

  // If we got a DMA complete, set the flag.
  if (ui32IntMask & AM_HAL_ADC_INT_DCMP) {
    uint32_t* source_buffer;
    if (g_dma_destination_index == 0) {
      source_buffer = g_ui32ADCSampleBuffer0;
      g_dma_destination_index = 1;
    } else {
      source_buffer = g_ui32ADCSampleBuffer1;
      g_dma_destination_index = 0;
    }
    adc_start_dma(g_adc_dma_error_reporter);

    // For slot 1:
    uint32_t slotCount = 0;
    for (uint32_t indi = 0; indi < kAdcSampleBufferSize; indi++) {
      am_hal_adc_sample_t temp;

      temp.ui32Slot = AM_HAL_ADC_FIFO_SLOT(source_buffer[indi]);
      temp.ui32Sample = AM_HAL_ADC_FIFO_SAMPLE(source_buffer[indi]);

      if (temp.ui32Slot == 1) {
        g_audio_capture_buffer[g_audio_capture_buffer_start] = temp.ui32Sample;
        g_audio_capture_buffer_start =
            (g_audio_capture_buffer_start + 1) % kAudioCaptureBufferSize;
        slotCount++;
      }
    }

    g_total_samples_captured += slotCount;
    g_latest_audio_timestamp =
        (g_total_samples_captured / (kAudioSampleFrequency / 1000));
  }

  // If we got a DMA error, set the flag.
  if (ui32IntMask & AM_HAL_ADC_INT_DERR) {
    g_adc_dma_error = true;
  }
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  // Set the clock frequency.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - configuring the system clock failed.");
    return kTfLiteError;
  }

  // Set the default cache configuration and enable it.
  if (AM_HAL_STATUS_SUCCESS !=
      am_hal_cachectrl_config(&am_hal_cachectrl_defaults)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - configuring the system cache failed.");
    return kTfLiteError;
  }
  if (AM_HAL_STATUS_SUCCESS != am_hal_cachectrl_enable()) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Error - enabling the system cache failed.");
    return kTfLiteError;
  }

  // Ensure the CPU is running as fast as possible.
  enable_burst_mode(error_reporter);

  // Start the CTIMER A3 for timer-based ADC measurements.
  init_timerA3_for_ADC();

  // Enable interrupts.
  NVIC_EnableIRQ(ADC_IRQn);
  am_hal_interrupt_master_enable();

  // Edge Board Pin Definitions
  constexpr int kSfEdgePinMic0 = 11;
  const am_hal_gpio_pincfg_t g_sf_edge_pin_mic0 = {
      .uFuncSel = AM_HAL_PIN_11_ADCSE2,
  };
  constexpr int kSfEdgePinMic1 = 29;
  const am_hal_gpio_pincfg_t g_sf_edge_pin_mic1 = {
      .uFuncSel = AM_HAL_PIN_29_ADCSE1,
  };

  // Set pins to act as our ADC input
  am_hal_gpio_pinconfig(kSfEdgePinMic0, g_sf_edge_pin_mic0);
  am_hal_gpio_pinconfig(kSfEdgePinMic1, g_sf_edge_pin_mic1);

  // Configure the ADC
  adc_config0(error_reporter);

  // Trigger the ADC sampling for the first time manually.
  if (AM_HAL_STATUS_SUCCESS != am_hal_adc_sw_trigger(g_adc_handle)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Error - triggering the ADC0 failed.");
    return kTfLiteError;
  }

  // Enable the LED outputs.
  am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_RED, g_AM_HAL_GPIO_OUTPUT_12);
  am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_YELLOW, g_AM_HAL_GPIO_OUTPUT_12);

  am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
  am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);

  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }

  // This is the 'zero' level of the microphone when no audio is present, and
  // should be recalibrated if the hardware configuration ever changes. It was
  // generated experimentally by averaging some samples captured on a board.
  const int16_t kAdcSampleDC = 6003;

  // Temporary gain emulation to deal with too-quiet audio on prototype boards.
  const int16_t kAdcSampleGain = 10;

  // This should only be called when the main thread notices that the latest
  // audio sample data timestamp has changed, so that there's new data in the
  // capture ring buffer. The ring buffer will eventually wrap around and
  // overwrite the data, but the assumption is that the main thread is checking
  // often enough and the buffer is large enough that this call will be made
  // before that happens.
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < duration_sample_count; ++i) {
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    const int32_t capture_value = g_audio_capture_buffer[capture_index];
    int32_t output_value = capture_value - kAdcSampleDC;
    output_value *= kAdcSampleGain;
    if (output_value < std::numeric_limits<int16_t>::min()) {
      output_value = std::numeric_limits<int16_t>::min();
    }
    if (output_value > std::numeric_limits<int16_t>::max()) {
      output_value = std::numeric_limits<int16_t>::max();
    }
    g_audio_output_buffer[i] = output_value;
  }

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }

#endif // USE_ANALOG_MIC_SFE_BSPS

#endif  // ARDUINO_EXCLUDE_CODE
