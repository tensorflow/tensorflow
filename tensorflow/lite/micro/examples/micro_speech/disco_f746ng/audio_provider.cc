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

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"

#include "AUDIO_DISCO_F746NG.h"
#include "SDRAM_DISCO_F746NG.h"
#include "mbed.h"  // NOLINT
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

namespace {

bool g_is_audio_initialized = false;
constexpr int kAudioCaptureBufferSize = kAudioSampleFrequency * 0.5;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;

// For a full example of how to access audio on the STM32F746NG board, see
// https://os.mbed.com/teams/ST/code/DISCO-F746NG_AUDIO_demo/
AUDIO_DISCO_F746NG g_audio_device;
SDRAM_DISCO_F746NG g_sdram_device;

typedef enum {
  BUFFER_OFFSET_NONE = 0,
  BUFFER_OFFSET_HALF = 1,
  BUFFER_OFFSET_FULL = 2,
} BUFFER_StateTypeDef;

#define AUDIO_BLOCK_SIZE ((uint32_t)2048)
#define AUDIO_BUFFER_IN SDRAM_DEVICE_ADDR /* In SDRAM */
#define AUDIO_BUFFER_OUT \
  (SDRAM_DEVICE_ADDR + (AUDIO_BLOCK_SIZE * 2)) /* In SDRAM */
__IO uint32_t g_audio_rec_buffer_state = BUFFER_OFFSET_NONE;

uint8_t SetSysClock_PLL_HSE_200MHz() {
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;

  // Enable power clock
  __PWR_CLK_ENABLE();

  // Enable HSE oscillator and activate PLL with HSE as source
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON; /* External xtal on OSC_IN/OSC_OUT */

  // Warning: this configuration is for a 25 MHz xtal clock only
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;   // VCO input clock = 1 MHz (25 MHz / 25)
  RCC_OscInitStruct.PLL.PLLN = 400;  // VCO output clock = 400 MHz (1 MHz * 400)
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;  // PLLCLK = 200 MHz (400 MHz / 2)
  RCC_OscInitStruct.PLL.PLLQ = 8;  // USB clock = 50 MHz (400 MHz / 8)

  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    return 0;  // FAIL
  }

  // Activate the OverDrive to reach the 216 MHz Frequency
  if (HAL_PWREx_EnableOverDrive() != HAL_OK) {
    return 0;  // FAIL
  }

  // Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2
  // clocks dividers
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK |
                                 RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;  // 200 MHz
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;         // 200 MHz
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;          //  50 MHz
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;          // 100 MHz

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK) {
    return 0;  // FAIL
  }
  HAL_RCC_MCOConfig(RCC_MCO1, RCC_MCO1SOURCE_HSE, RCC_MCODIV_4);
  return 1;  // OK
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  SetSysClock_PLL_HSE_200MHz();

  // Initialize SDRAM buffers.
  memset((uint16_t*)AUDIO_BUFFER_IN, 0, AUDIO_BLOCK_SIZE * 2);
  memset((uint16_t*)AUDIO_BUFFER_OUT, 0, AUDIO_BLOCK_SIZE * 2);
  g_audio_rec_buffer_state = BUFFER_OFFSET_NONE;

  // Start Recording.
  g_audio_device.IN_Record((uint16_t*)AUDIO_BUFFER_IN, AUDIO_BLOCK_SIZE);

  // Also play results out to headphone jack.
  g_audio_device.OUT_SetAudioFrameSlot(CODEC_AUDIOFRAME_SLOT_02);
  g_audio_device.OUT_Play((uint16_t*)AUDIO_BUFFER_OUT, AUDIO_BLOCK_SIZE * 2);

  return kTfLiteOk;
}

void CaptureSamples(const int16_t* sample_data) {
  const int sample_size = AUDIO_BLOCK_SIZE / (sizeof(int16_t) * 2);
  const int32_t time_in_ms =
      g_latest_audio_timestamp + (sample_size / (kAudioSampleFrequency / 1000));

  const int32_t start_sample_offset =
      g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
  for (int i = 0; i < sample_size; ++i) {
    const int capture_index =
        (start_sample_offset + i) % kAudioCaptureBufferSize;
    g_audio_capture_buffer[capture_index] =
        (sample_data[(i * 2) + 0] / 2) + (sample_data[(i * 2) + 1] / 2);
  }
  // This is how we let the outside world know that new audio data has arrived.
  g_latest_audio_timestamp = time_in_ms;
}

}  // namespace

// These callbacks need to be linkable symbols, because they override weak
// default versions.
void BSP_AUDIO_IN_TransferComplete_CallBack(void) {
  g_audio_rec_buffer_state = BUFFER_OFFSET_FULL;
  /* Copy recorded 1st half block */
  memcpy((uint16_t*)(AUDIO_BUFFER_OUT), (uint16_t*)(AUDIO_BUFFER_IN),
         AUDIO_BLOCK_SIZE);
  CaptureSamples(reinterpret_cast<int16_t*>(AUDIO_BUFFER_IN));
  return;
}

// Another weak symbol override.
void BSP_AUDIO_IN_HalfTransfer_CallBack(void) {
  g_audio_rec_buffer_state = BUFFER_OFFSET_HALF;
  /* Copy recorded 2nd half block */
  memcpy((uint16_t*)(AUDIO_BUFFER_OUT + (AUDIO_BLOCK_SIZE)),
         (uint16_t*)(AUDIO_BUFFER_IN + (AUDIO_BLOCK_SIZE)), AUDIO_BLOCK_SIZE);
  CaptureSamples(
      reinterpret_cast<int16_t*>(AUDIO_BUFFER_IN + AUDIO_BLOCK_SIZE));
  return;
}

// Main entry point for getting audio data.
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
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
