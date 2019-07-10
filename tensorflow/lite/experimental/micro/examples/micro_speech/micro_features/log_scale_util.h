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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_LOG_SCALE_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_LOG_SCALE_UTIL_H_

#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"

struct LogScaleConfig {
  // set to false (0) to disable this module
  int enable_log;
  // scale results by 2^(scale_shift)
  int scale_shift;
};

// Populates the LogScaleConfig with "sane" default values.
void LogScaleFillConfigWithDefaults(struct LogScaleConfig* config);

// Allocates any buffers.
int LogScalePopulateState(tflite::ErrorReporter* error_reporter,
                          const struct LogScaleConfig* config,
                          struct LogScaleState* state);

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_LOG_SCALE_UTIL_H_
