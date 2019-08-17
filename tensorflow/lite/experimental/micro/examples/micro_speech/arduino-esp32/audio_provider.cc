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

#include "tensorflow/lite/experimental/micro/examples/micro_speech/audio_provider.h"

#include <Arduino.h>        // NOLINT
#include <driver/i2s.h>
//#include "freertos/queue.h"
#include <soc/i2s_reg.h> // I2S_TIMING_REG

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h"

namespace {
static const i2s_port_t i2s_num = I2S_NUM_0; // i2s port number

// These are the raw buffers that are filled by the ADC during DMA
constexpr int kAdcNumSlots = 8;
constexpr int kAdcSamplesPerSlot = 128;

TaskHandle_t xCaptureTaskHandle = NULL;

bool g_is_audio_initialized = false;
constexpr int kAudioCaptureBufferSize = kAudioSampleFrequency * 0.5;
int32_t* g_audio_capture_buffer = (int32_t*)heap_caps_malloc(kAudioCaptureBufferSize * sizeof(int32_t), MALLOC_CAP_DMA | MALLOC_CAP_32BIT);
// A buffer that holds our output
int16_t* g_audio_output_buffer = (int16_t*)calloc(kMaxAudioSampleSize, sizeof(int16_t));
// Mark as volatile so we can check in a while loop to see if
// any samples have arrived yet.
volatile int32_t g_latest_audio_timestamp = 0;
}  // anonymous namespace

void CaptureSamples(void * pvParameters) {
  for( ;; ) {
    // Determine the index, in the history of all samples, of the last sample
    const int32_t start_sample_offset =
        g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
    // Determine the index of this sample in our ring buffer
    const int capture_index = start_sample_offset % kAudioCaptureBufferSize;
    // Read the data to the correct place in our buffer
    size_t num_bytes_read;
    i2s_read(i2s_num,
             (char *)(g_audio_capture_buffer + capture_index),
             (kAudioCaptureBufferSize - capture_index) * sizeof(int32_t),
             &num_bytes_read,
             (TickType_t) 0);
    // This is how many bytes of new data we have
    int number_of_samples = num_bytes_read / sizeof(uint32_t);
    // Calculate what timestamp the last audio sample represents
    int32_t time_in_ms =
        g_latest_audio_timestamp +
        (number_of_samples / (kAudioSampleFrequency / 1000));
    // This is how we let the outside world know that new audio data has arrived.
    g_latest_audio_timestamp = time_in_ms;

    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  if (g_audio_capture_buffer == NULL || g_audio_output_buffer == NULL) {
    error_reporter->Report("Error - Failed to allocate buffer memory\n");
    return kTfLiteError;
  }

  const i2s_config_t i2s_config = {
      .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = kAudioSampleFrequency,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_RIGHT,
      .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,     // Interrupt level 1
      .dma_buf_count = kAdcNumSlots,
      .dma_buf_len = kAdcSamplesPerSlot,
      .use_apll = false
  };

  // The pin config as per the setup
  const i2s_pin_config_t pin_config = {
      .bck_io_num = 26,   // BCKL
      .ws_io_num = 25,    // LRCL
      .data_out_num = I2S_PIN_NO_CHANGE, // not used (only for audio output)
      .data_in_num = 22   // DOUT
  };

  esp_err_t err = i2s_driver_install(i2s_num, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    error_reporter->Report("Error - Failed installing driver: %d\n", err);
    return kTfLiteError;
  }

  // set i2s to philips standard
  REG_SET_BIT(  I2S_TIMING_REG(i2s_num), /* I2S_RX_SD_IN_DELAY */ BIT(9));   // delay SD signal 2 cycles
  REG_SET_BIT( I2S_CONF_REG(i2s_num), I2S_RX_MSB_SHIFT);   // philips mode

  err = i2s_set_pin(i2s_num, &pin_config);
  if (err != ESP_OK) {
    error_reporter->Report("Error - Failed setting pin: %d\n", err);
    return kTfLiteError;
  }

  // start task that will get the samples from i2s
  BaseType_t task_created = xTaskCreate(
                                        CaptureSamples,
                                        "CaptureSamples",
                                        2048,
                                        NULL,
                                        tskIDLE_PRIORITY+1,
                                        &xCaptureTaskHandle
                                       );
  configASSERT( xCaptureTaskHandle );
  if (task_created != pdPASS) {
    error_reporter->Report("Error - Failed creating capture task\n");
    return kTfLiteError;
  }

  // Block until we have our first audio sample
  while (!g_latest_audio_timestamp) {
  }

  //Serial.printf("Audio initialized\n");

  return kTfLiteOk;
}

TfLiteStatus EndAudioRecording(tflite::ErrorReporter* error_reporter) {
  i2s_driver_uninstall(i2s_num); //stop & destroy i2s driver
  //free(g_audio_capture_buffer);
  //free(g_audio_output_buffer);
  g_latest_audio_timestamp = 0;
  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  // Set everything up to start receiving audio
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    i2s_zero_dma_buffer(i2s_num);
    g_is_audio_initialized = true;
  }
  // This next part should only be called when the main thread notices that the
  // latest audio sample data timestamp has changed, so that there's new data
  // in the capture ring buffer. The ring buffer will eventually wrap around and
  // overwrite the data, but the assumption is that the main thread is checking
  // often enough and the buffer is large enough that this call will be made
  // before that happens.

  // Determine the index, in the history of all samples, of the first
  // sample we want
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  // Determine how many samples we want in total
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);
  static int mic_dc_offset = 0;
  int32_t mic_dc_offset_samples = 0;
  for (int i = 0; i < duration_sample_count; ++i) {
    // For each sample, transform its index in the history of all samples into
    // its index in g_audio_capture_buffer
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    // Write the sample to the output buffer
    int16_t sample = (g_audio_capture_buffer[capture_index]>>14) - mic_dc_offset;
    g_audio_output_buffer[i] = sample;
    mic_dc_offset_samples += sample;
    #ifdef DEBUG_SAMPLES
      Serial.println(sample);
    #endif
  }
  // calculate microphone dc offset
  mic_dc_offset_samples /= duration_sample_count;
  mic_dc_offset += mic_dc_offset_samples / 8;  // running average
  // Set pointers to provide access to the audio
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;

  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
