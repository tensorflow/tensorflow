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
#include <stdio.h>

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

int main(int argc, char** argv) {
  struct FrontendConfig frontend_config;
  FrontendFillConfigWithDefaults(&frontend_config);

  char* filename = argv[1];
  int sample_rate = 16000;

  struct FrontendState frontend_state;
  if (!FrontendPopulateState(&frontend_config, &frontend_state, sample_rate)) {
    fprintf(stderr, "Failed to populate frontend state\n");
    FrontendFreeStateContents(&frontend_state);
    return 1;
  }

  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open %s for read\n", filename);
    return 1;
  }
  fseek(fp, 0L, SEEK_END);
  size_t audio_file_size = ftell(fp) / sizeof(int16_t);
  fseek(fp, 0L, SEEK_SET);
  int16_t* audio_data = malloc(audio_file_size * sizeof(int16_t));
  int16_t* original_audio_data = audio_data;
  if (audio_file_size !=
      fread(audio_data, sizeof(int16_t), audio_file_size, fp)) {
    fprintf(stderr, "Failed to read in all audio data\n");
    return 1;
  }

  while (audio_file_size > 0) {
    size_t num_samples_read;
    struct FrontendOutput output = FrontendProcessSamples(
        &frontend_state, audio_data, audio_file_size, &num_samples_read);
    audio_data += num_samples_read;
    audio_file_size -= num_samples_read;

    if (output.values != NULL) {
      int i;
      for (i = 0; i < output.size; ++i) {
        printf("%d ", output.values[i]);
      }
      printf("\n");
    }
  }

  FrontendFreeStateContents(&frontend_state);
  free(original_audio_data);
  return 0;
}
