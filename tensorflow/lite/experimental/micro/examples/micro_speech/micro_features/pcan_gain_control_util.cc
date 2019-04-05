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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_util.h"

#include <math.h>

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h"

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
  const float x_as_float =
      (static_cast<float>(x)) / (static_cast<uint32_t>(1) << input_bits);
  const float gain_as_float =
      (static_cast<uint32_t>(1) << config->gain_bits) *
      powf(x_as_float + config->offset, -config->strength);

  if (gain_as_float > kint16max) {
    return kint16max;
  }
  return static_cast<int16_t>(gain_as_float + 0.5f);
}

int PcanGainControlPopulateState(tflite::ErrorReporter* error_reporter,
                                 const struct PcanGainControlConfig* config,
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
  STATIC_ALLOC_ENSURE_ARRAY_SIZE(
      state->gain_lut, (kWideDynamicFunctionLUTSize * sizeof(int16_t)));
  state->snr_shift = config->gain_bits - input_correction_bits - kPcanSnrBits;

  const int32_t input_bits = smoothing_bits - input_correction_bits;
  state->gain_lut[0] = PcanGainLookupFunction(config, input_bits, 0);
  state->gain_lut[1] = PcanGainLookupFunction(config, input_bits, 1);
  int16_t* temp_gain_lut = state->gain_lut - 6;
  int interval;
  for (interval = 2; interval <= kWideDynamicFunctionBits; ++interval) {
    const uint32_t x0 = static_cast<uint32_t>(1) << (interval - 1);
    const uint32_t x1 = x0 + (x0 >> 1);
    const uint32_t x2 =
        (interval == kWideDynamicFunctionBits) ? x0 + (x0 - 1) : 2 * x0;

    const int16_t y0 = PcanGainLookupFunction(config, input_bits, x0);
    const int16_t y1 = PcanGainLookupFunction(config, input_bits, x1);
    const int16_t y2 = PcanGainLookupFunction(config, input_bits, x2);

    const int32_t diff1 = static_cast<int32_t>(y1) - y0;
    const int32_t diff2 = static_cast<int32_t>(y2) - y0;
    const int32_t a1 = 4 * diff1 - diff2;
    const int32_t a2 = diff2 - a1;

    temp_gain_lut[4 * interval] = y0;
    temp_gain_lut[4 * interval + 1] = static_cast<int16_t>(a1);
    temp_gain_lut[4 * interval + 2] = static_cast<int16_t>(a2);
  }
  return 1;
}
