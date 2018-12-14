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
#include "tensorflow/lite/experimental/microfrontend/lib/noise_reduction_util.h"

#include <stdio.h>

void NoiseReductionFillConfigWithDefaults(struct NoiseReductionConfig* config) {
  config->smoothing_bits = 10;
  config->even_smoothing = 0.025;
  config->odd_smoothing = 0.06;
  config->min_signal_remaining = 0.05;
}

int NoiseReductionPopulateState(const struct NoiseReductionConfig* config,
                                struct NoiseReductionState* state,
                                int num_channels) {
  state->smoothing_bits = config->smoothing_bits;
  state->odd_smoothing = config->odd_smoothing * (1 << kNoiseReductionBits);
  state->even_smoothing = config->even_smoothing * (1 << kNoiseReductionBits);
  state->min_signal_remaining =
      config->min_signal_remaining * (1 << kNoiseReductionBits);
  state->num_channels = num_channels;
  state->estimate = calloc(state->num_channels, sizeof(*state->estimate));
  if (state->estimate == NULL) {
    fprintf(stderr, "Failed to alloc estimate buffer\n");
    return 0;
  }
  return 1;
}

void NoiseReductionFreeStateContents(struct NoiseReductionState* state) {
  free(state->estimate);
}
