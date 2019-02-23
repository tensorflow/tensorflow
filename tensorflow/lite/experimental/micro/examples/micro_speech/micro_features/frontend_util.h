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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FRONTEND_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FRONTEND_UTIL_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/fft_util.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/filterbank_util.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/frontend.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale_util.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/noise_reduction_util.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/pcan_gain_control_util.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window_util.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"

struct FrontendConfig {
  struct WindowConfig window;
  struct FilterbankConfig filterbank;
  struct NoiseReductionConfig noise_reduction;
  struct PcanGainControlConfig pcan_gain_control;
  struct LogScaleConfig log_scale;
};

// Fills the frontendConfig with "sane" defaults.
void FrontendFillConfigWithDefaults(struct FrontendConfig* config);

// Prepares any buffers.
int FrontendPopulateState(tflite::ErrorReporter* error_reporter,
                          const struct FrontendConfig* config,
                          struct FrontendState* state, int sample_rate);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_FRONTEND_UTIL_H_
