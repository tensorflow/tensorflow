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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_PCAN_GAIN_CONTROL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_PCAN_GAIN_CONTROL_H_

#include <stdint.h>
#include <stdlib.h>

#define kPcanSnrBits 12
#define kPcanOutputBits 6

#define kWideDynamicFunctionBits 32
#define kWideDynamicFunctionLUTSize (4 * kWideDynamicFunctionBits - 3)

struct PcanGainControlState {
  int enable_pcan;
  uint32_t* noise_estimate;
  int num_channels;
  int16_t gain_lut[kWideDynamicFunctionLUTSize];
  int32_t snr_shift;
};

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut);

uint32_t PcanShrink(const uint32_t x);

void PcanGainControlApply(struct PcanGainControlState* state, uint32_t* signal);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_PCAN_GAIN_CONTROL_H_
