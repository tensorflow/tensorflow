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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_NOISE_REDUCTION_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_NOISE_REDUCTION_UTIL_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"

struct NoiseReductionConfig {
  // scale the signal up by 2^(smoothing_bits) before reduction
  int smoothing_bits;
  // smoothing coefficient for even-numbered channels
  float even_smoothing;
  // smoothing coefficient for odd-numbered channels
  float odd_smoothing;
  // fraction of signal to preserve (1.0 disables this module)
  float min_signal_remaining;
};

// Populates the NoiseReductionConfig with "sane" default values.
void NoiseReductionFillConfigWithDefaults(struct NoiseReductionConfig* config);

// Prepares any buffers.
int NoiseReductionPopulateState(tflite::ErrorReporter* error_reporter,
                                const struct NoiseReductionConfig* config,
                                struct NoiseReductionState* state,
                                int num_channels);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_NOISE_REDUCTION_UTIL_H_
