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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FFT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FFT_H_

#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h"

struct complex_int16_t {
  int16_t real;
  int16_t imag;
};

struct FftState {
  int16_t input[kMaxAudioSampleSize];
  struct complex_int16_t output[kMaxAudioSampleSize + 2];
  size_t fft_size;
  size_t input_size;
  // This magic number was derived from KissFFT's estimate of how much space it
  // will need to process the particular lengths and datatypes we need to for
  // these model settings. This size will need to be recalculated for different
  // models, but you will see a runtime error if it's not large enough.
  char scratch[2848];
  size_t scratch_size;
};

void FftCompute(struct FftState* state, const int16_t* input,
                int input_scale_shift);

void FftInit(struct FftState* state);

void FftReset(struct FftState* state);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FFT_H_
