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
#include "tensorflow/lite/experimental/microfrontend/lib/pcan_gain_control_util.h"

#include <math.h>
#include <stdio.h>

#define kint16max 0x00007FFF

void PcanGainControlFillConfigWithDefaults(
    struct PcanGainControlConfig* config) {
  config->enable_pcan = 0;
  config->strength = 0.95;
  config->offset = 80.0;
  config->gain_bits = 21;
}

int16_t PcanGainLookupFunction(const struct PcanGainControlConfig* config,
                               int32_t input_bits, uint32_t x) {
  const float x_as_float = ((float) x) / ((uint32_t) 1 << input_bits);
  const float gain_as_float = ((uint32_t) 1 << config->gain_bits) *
      powf(x_as_float + config->offset, -config->strength);

  if (gain_as_float > kint16max) {
    return kint16max;
  }
  return (int16_t) (gain_as_float + 0.5f);
}

int PcanGainControlPopulateState(const struct PcanGainControlConfig* config,
                                 struct PcanGainControlState* state,
                                 uint32_t* noise_estimate,
                                 const int num_channels,
                                 const uint16_t smoothing_bits,
                                 const int32_t input_correction_bits) {
  state->enable_pcan = config->enable_pcan;
  if (!state->enable_pcan) {
    return 1;
  }
  state->noise_estimate = noise_estimate;
  state->num_channels = num_channels;
  state->gain_lut = malloc(kWideDynamicFunctionLUTSize * sizeof(int16_t));
  if (state->gain_lut == NULL) {
    fprintf(stderr, "Failed to allocate gain LUT\n");
    return 0;
  }
  state->snr_shift = config->gain_bits - input_correction_bits - kPcanSnrBits;

  const int32_t input_bits = smoothing_bits - input_correction_bits;
  state->gain_lut[0] = PcanGainLookupFunction(config, input_bits, 0);
  state->gain_lut[1] = PcanGainLookupFunction(config, input_bits, 1);
  state->gain_lut -= 6;
  int interval;
  for (interval = 2; interval <= kWideDynamicFunctionBits; ++interval) {
    const uint32_t x0 = (uint32_t) 1 << (interval - 1);
    const uint32_t x1 = x0 + (x0 >> 1);
    const uint32_t x2 = (interval == kWideDynamicFunctionBits)
        ? x0 + (x0 - 1) : 2 * x0;

    const int16_t y0 = PcanGainLookupFunction(config, input_bits, x0);
    const int16_t y1 = PcanGainLookupFunction(config, input_bits, x1);
    const int16_t y2 = PcanGainLookupFunction(config, input_bits, x2);

    const int32_t diff1 = (int32_t) y1 - y0;
    const int32_t diff2 = (int32_t) y2 - y0;
    const int32_t a1 = 4 * diff1 - diff2;
    const int32_t a2 = diff2 - a1;

    state->gain_lut[4 * interval] = y0;
    state->gain_lut[4 * interval + 1] = (int16_t) a1;
    state->gain_lut[4 * interval + 2] = (int16_t) a2;
  }
  state->gain_lut += 6;
  return 1;
}

void PcanGainControlFreeStateContents(struct PcanGainControlState* state) {
  free(state->gain_lut);
}
