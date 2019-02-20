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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend_util.h"

#include <string.h>

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/bits.h"

void FrontendFillConfigWithDefaults(struct FrontendConfig* config) {
  WindowFillConfigWithDefaults(&config->window);
  FilterbankFillConfigWithDefaults(&config->filterbank);
  NoiseReductionFillConfigWithDefaults(&config->noise_reduction);
  PcanGainControlFillConfigWithDefaults(&config->pcan_gain_control);
  LogScaleFillConfigWithDefaults(&config->log_scale);
}

int FrontendPopulateState(tflite::ErrorReporter* error_reporter,
                          const struct FrontendConfig* config,
                          struct FrontendState* state, int sample_rate) {
  memset(state, 0, sizeof(*state));

  if (!WindowPopulateState(error_reporter, &config->window, &state->window,
                           sample_rate)) {
    error_reporter->Report("Failed to populate window state");
    return 0;
  }

  if (!FftPopulateState(error_reporter, &state->fft, state->window.size)) {
    error_reporter->Report("Failed to populate fft state");
    return 0;
  }
  FftInit(&state->fft);

  if (!FilterbankPopulateState(error_reporter, &config->filterbank,
                               &state->filterbank, sample_rate,
                               state->fft.fft_size / 2 + 1)) {
    error_reporter->Report("Failed to populate filterbank state");
    return 0;
  }

  if (!NoiseReductionPopulateState(error_reporter, &config->noise_reduction,
                                   &state->noise_reduction,
                                   state->filterbank.num_channels)) {
    error_reporter->Report("Failed to populate noise reduction state");
    return 0;
  }

  int input_correction_bits =
      MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
  if (!PcanGainControlPopulateState(
          error_reporter, &config->pcan_gain_control, &state->pcan_gain_control,
          state->noise_reduction.estimate, state->filterbank.num_channels,
          state->noise_reduction.smoothing_bits, input_correction_bits)) {
    error_reporter->Report("Failed to populate pcan gain control state");
    return 0;
  }

  if (!LogScalePopulateState(error_reporter, &config->log_scale,
                             &state->log_scale)) {
    error_reporter->Report("Failed to populate log scale state");
    return 0;
  }

  FrontendReset(state);

  // All good, return a true value.
  return 1;
}
