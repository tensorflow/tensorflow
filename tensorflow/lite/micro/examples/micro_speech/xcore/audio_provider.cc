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
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

#include <stdio.h>
#include <string.h>

extern "C" {
#include "fifo.h"
#include "microspeech_xcore_support.h"
#include "mic_array_conf.h"
#include "dsp_qformat.h"
}
namespace {
volatile int32_t g_latest_audio_timestamp = 0;

/* Hold 80ms of samples */
#define FRAMES_TO_HOLD  5
#define SAMPLES_PER_FRAME (1<<MIC_ARRAY_MAX_FRAME_SIZE_LOG2)

int16_t g_audio_output_buffer[kMaxAudioSampleSize];
bool g_is_audio_initialized = false;
int16_t g_captured_audio_buffer[FRAMES_TO_HOLD * SAMPLES_PER_FRAME];

microspeech_device_t* device = NULL;
fifo_t* this_fifo = NULL;
int32_t* sample_buf = NULL;
int write_cnt = 0;
}  // namespace

// Initialization for receiving audio data
TfLiteStatus InitAudioRecording(tflite::ErrorReporter *error_reporter) {
  microspeech_device_t* tmp_ptr = NULL;
  if ((tmp_ptr = get_microspeech_device()) == NULL ) {
    return kTfLiteError;
  }
  device = tmp_ptr;
  sample_buf = (int32_t*)device->sample_buffer;
  this_fifo = (fifo_t*)device->sample_fifo;

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

  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  const int duration_sample_count = duration_ms * (kAudioSampleFrequency / 1000);

  for (int i = 0; i<duration_sample_count; i++) {
    const int capture_index = (start_offset + i) % (FRAMES_TO_HOLD * SAMPLES_PER_FRAME);
    g_audio_output_buffer[i] = g_captured_audio_buffer[capture_index];
  }

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  return g_latest_audio_timestamp;
}

extern "C" {
void increment_timestamp(int32_t increment) {
  int32_t sample;
  int buf_num= 0;

  if (g_is_audio_initialized) {
    fifo_get(*this_fifo, &buf_num);

    for (int i = 0; i < SAMPLES_PER_FRAME; i++) {
      sample = *(sample_buf + i + (buf_num * SAMPLES_PER_FRAME));
      g_captured_audio_buffer[(SAMPLES_PER_FRAME * write_cnt) + i] = (int16_t) (sample>>16);
    }

    write_cnt++;
    if (write_cnt >= FRAMES_TO_HOLD) {
        write_cnt = 0;
    }

    g_latest_audio_timestamp += increment;
  }
}
}
