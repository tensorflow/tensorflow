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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FILTERBANK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FILTERBANK_H_

#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h"

#define kFilterbankBits 12

struct FilterbankState {
  int num_channels;
  int start_index;
  int end_index;
  int16_t channel_frequency_starts[kFeatureSliceSize + 1];
  int16_t channel_weight_starts[kFeatureSliceSize + 1];
  int16_t channel_widths[kFeatureSliceSize + 1];
  int16_t weights[316];
  int16_t unweights[316];
  uint64_t work[kFeatureSliceSize + 1];
};

// Converts the relevant complex values of an FFT output into energy (the
// square magnitude).
void FilterbankConvertFftComplexToEnergy(struct FilterbankState* state,
                                         struct complex_int16_t* fft_output,
                                         int32_t* energy);

// Computes the mel-scale filterbank on the given energy array. Output is cached
// internally - to fetch it, you need to call FilterbankSqrt.
void FilterbankAccumulateChannels(struct FilterbankState* state,
                                  const int32_t* energy);

// Applies an integer square root to the 64 bit intermediate values of the
// filterbank, and returns a pointer to them. Memory will be invalidated the
// next time FilterbankAccumulateChannels is called.
uint32_t* FilterbankSqrt(struct FilterbankState* state, int scale_down_shift);

void FilterbankReset(struct FilterbankState* state);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FILTERBANK_H_
