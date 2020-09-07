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

#include "hx_drv_tflm.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

namespace {
// Feedback silence buffer when beginning start_ms <= 0
int16_t g_silence[kMaxAudioSampleSize] = {0};
// Latest time-stamp
int32_t g_latest_audio_timestamp = 0;
// config about audio data size and address
hx_drv_mic_data_config_t mic_config;
// Flag for check if audio is initialize or not
bool g_is_audio_initialized = false;
}  // namespace

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    if (hx_drv_mic_initial() != HX_DRV_LIB_PASS) return kTfLiteError;

    hx_drv_mic_on();
    g_is_audio_initialized = true;
  }

  if (start_ms > 0) {
    hx_drv_mic_capture(&mic_config);
  } else {
    mic_config.data_size = kMaxAudioSampleSize;
    mic_config.data_address = (uint32_t)g_silence;
  }

  *audio_samples_size = mic_config.data_size;
  *audio_samples = (int16_t*)mic_config.data_address;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  hx_drv_mic_timestamp_get(&g_latest_audio_timestamp);
  return g_latest_audio_timestamp;
}
