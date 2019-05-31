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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_util.h"

#define FIXED_POINT 16
#include "kiss_fft.h"
#include "tools/kiss_fftr.h"

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h"

int FftPopulateState(tflite::ErrorReporter* error_reporter,
                     struct FftState* state, size_t input_size) {
  state->input_size = input_size;
  state->fft_size = 1;
  while (state->fft_size < state->input_size) {
    state->fft_size <<= 1;
  }

  STATIC_ALLOC_ENSURE_ARRAY_SIZE(state->input,
                                 (state->fft_size * sizeof(*state->input)));

  STATIC_ALLOC_ENSURE_ARRAY_SIZE(
      state->output, ((state->fft_size / 2 + 1) * sizeof(*state->output) * 2));

  // Ask kissfft how much memory it wants.
  size_t scratch_size = 0;
  kiss_fftr_cfg kfft_cfg =
      kiss_fftr_alloc(state->fft_size, 0, nullptr, &scratch_size);
  if (kfft_cfg != nullptr) {
    error_reporter->Report("Kiss memory sizing failed.");
    return 0;
  }
  STATIC_ALLOC_ENSURE_ARRAY_SIZE(state->scratch, scratch_size);
  state->scratch_size = scratch_size;
  // Let kissfft configure the scratch space we just allocated
  kfft_cfg = kiss_fftr_alloc(state->fft_size, 0, state->scratch, &scratch_size);
  if (reinterpret_cast<char*>(kfft_cfg) != state->scratch) {
    error_reporter->Report("Kiss memory preallocation strategy failed.");
    return 0;
  }
  return 1;
}
