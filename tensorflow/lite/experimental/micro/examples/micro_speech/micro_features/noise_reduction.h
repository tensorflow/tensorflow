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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_NOISE_REDUCTION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_NOISE_REDUCTION_H_

#define kNoiseReductionBits 14

#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h"

struct NoiseReductionState {
  int smoothing_bits;
  uint16_t even_smoothing;
  uint16_t odd_smoothing;
  uint16_t min_signal_remaining;
  int num_channels;
  uint32_t estimate[kFeatureSliceSize];
};

// Removes stationary noise from each channel of the signal using a low pass
// filter.
void NoiseReductionApply(struct NoiseReductionState* state, uint32_t* signal);

void NoiseReductionReset(struct NoiseReductionState* state);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_NOISE_REDUCTION_H_
