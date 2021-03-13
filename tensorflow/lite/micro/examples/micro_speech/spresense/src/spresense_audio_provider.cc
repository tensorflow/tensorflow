/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// The SPRESENSE_CONFIG_H is defined on compiler option.
// It contains "nuttx/config.h" from Spresense SDK to see the configurated
// parameters.
#include SPRESENSE_CONFIG_H
#include "spresense_audio_provider.h"

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

// Below definition is for dump audio captured data for debugging.
// #define CAPTURE_DATA

#ifdef CAPTURE_DATA
#include <stdio.h>
#include <string.h>

static int16_t tmp_data[16000];
static int data_cnt;
static bool is_printed = false;
#endif

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (spresense_audio_getsamples(start_ms, duration_ms, kAudioSampleFrequency,
                                 audio_samples_size, audio_samples) < 0) {
    return kTfLiteError;
  } else {
#ifdef CAPTURE_DATA
    if (start_ms >= 10000) {
      if (data_cnt == 0) printf("=========== Start Recording ==============\n");
      if (data_cnt < 16000) {
        int sz = (16000 - data_cnt) > *audio_samples_size ? *audio_samples_size
                                                          : (16000 - data_cnt);
        memcpy(&tmp_data[data_cnt], *audio_samples, sz * 2);
        data_cnt += sz;
      }
      if (!is_printed && data_cnt >= 16000) {
        printf("============ Stop Recording =============\n");
        for (int i = 0; i < 16000; i++) {
          printf("%d\n", tmp_data[i]);
        }
        is_printed = true;
      }
    }
#endif
    return kTfLiteOk;
  }
}

int32_t LatestAudioTimestamp() { return spresense_audio_lasttimestamp(); }
