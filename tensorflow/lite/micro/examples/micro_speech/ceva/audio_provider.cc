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

#include <stdio.h>
#include <stdlib.h>

#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

namespace {

int32_t g_latest_audio_timestamp = 0;

constexpr int kNoOfSamples = 512;
bool g_is_audio_initialized = false;
constexpr int kAudioCaptureBufferSize = kAudioSampleFrequency * 0.5;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
int16_t audio[513];

}  // namespace

// Main entry point for getting audio data.
TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
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

char filename[50] = "input.wav";
FILE* infile;
void init_audio() {
  uint8_t mem[3];
  printf("Using filename %s\n", filename);

  infile = fopen(filename, "rb");
  if (!infile) {
    printf("Can't open file\n");
    exit(1);
  }

  // skip wav header
  for (int i = 0; i < 44; i++) {
    fread(mem, 1, 1, infile);
  }
}

void read_samples() {
  int i = 0;
  uint8_t mem[3];
  bool done;
  for (int i = 0; i < kNoOfSamples; i++) {
    if (fread((char*)mem, 1, 2, infile) == 2) {
      audio[i] = (int16_t)mem[0] + (((int16_t)mem[1]) << 8);
    } else {
      done = true;
      fclose(infile);
      infile = fopen(filename, "rb");
      printf("EOF reached\n");

      break;
    }
  }
}

void CaptureSamples(const int16_t* sample_data) {
  const int sample_size = kNoOfSamples;
  const int32_t time_in_ms =
      g_latest_audio_timestamp + (sample_size / (kAudioSampleFrequency / 1000));

  const int32_t start_sample_offset =
      g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);

  for (int i = 0; i < sample_size; ++i) {
    const int capture_index =
        (start_sample_offset + i) % kAudioCaptureBufferSize;
    g_audio_capture_buffer[capture_index] = sample_data[i];
  }
  // This is how we let the outside world know that new audio data has arrived.
  g_latest_audio_timestamp = time_in_ms;
}

void GetAudio() {
  read_samples();
  CaptureSamples(audio);
}