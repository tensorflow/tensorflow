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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_PCAN_GAIN_CONTROL_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_PCAN_GAIN_CONTROL_UTIL_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"

struct PcanGainControlConfig {
  // set to false (0) to disable this module
  int enable_pcan;
  // gain normalization exponent (0.0 disables, 1.0 full strength)
  float strength;
  // positive value added in the normalization denominator
  float offset;
  // number of fractional bits in the gain
  int gain_bits;
};

void PcanGainControlFillConfigWithDefaults(
    struct PcanGainControlConfig* config);

int16_t PcanGainLookupFunction(const struct PcanGainControlConfig* config,
                               int32_t input_bits, uint32_t x);

int PcanGainControlPopulateState(tflite::ErrorReporter* error_reporter,
                                 const struct PcanGainControlConfig* config,
                                 struct PcanGainControlState* state,
                                 uint32_t* noise_estimate,
                                 const int num_channels,
                                 const uint16_t smoothing_bits,
                                 const int32_t input_correction_bits);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_PCAN_GAIN_CONTROL_UTIL_H_
