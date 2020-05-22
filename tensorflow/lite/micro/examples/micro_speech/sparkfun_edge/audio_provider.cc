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

#if defined(ARDUINO) && !defined(ARDUINO_SFE_EDGE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_SFE_EDGE)

#ifndef ARDUINO_EXCLUDE_CODE

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"

#include <limits>

// These are headers from Ambiq's Apollo3 SDK.
#include "am_bsp.h"         // NOLINT
#include "am_mcu_apollo.h"  // NOLINT
#include "am_util.h"        // NOLINT
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

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

#endif  // ARDUINO_EXCLUDE_CODE
