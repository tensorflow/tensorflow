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
#include "tensorflow/lite/experimental/microfrontend/lib/noise_reduction.h"

#include <string.h>

void NoiseReductionApply(struct NoiseReductionState* state, uint32_t* signal) {
  int i;
  for (i = 0; i < state->num_channels; ++i) {
    const uint32_t smoothing =
        ((i & 1) == 0) ? state->even_smoothing : state->odd_smoothing;
    const uint32_t one_minus_smoothing = (1 << kNoiseReductionBits) - smoothing;

    // Update the estimate of the noise.
    const uint32_t signal_scaled_up = signal[i] << state->smoothing_bits;
    uint32_t estimate =
        (((uint64_t) signal_scaled_up * smoothing) +
         ((uint64_t) state->estimate[i] * one_minus_smoothing)) >>
        kNoiseReductionBits;
    state->estimate[i] = estimate;

    // Make sure that we can't get a negative value for the signal - estimate.
    if (estimate > signal_scaled_up) {
      estimate = signal_scaled_up;
    }

    const uint32_t floor =
        ((uint64_t) signal[i] * state->min_signal_remaining) >>
        kNoiseReductionBits;
    const uint32_t subtracted = (signal_scaled_up - estimate) >>
        state->smoothing_bits;
    const uint32_t output = subtracted > floor ? subtracted : floor;
    signal[i] = output;
  }
}

void NoiseReductionReset(struct NoiseReductionState* state) {
  memset(state->estimate, 0, sizeof(*state->estimate) * state->num_channels);
}
