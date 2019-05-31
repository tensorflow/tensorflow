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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/log_scale_util.h"

void LogScaleFillConfigWithDefaults(struct LogScaleConfig* config) {
  config->enable_log = 1;
  config->scale_shift = 6;
}

int LogScalePopulateState(tflite::ErrorReporter* error_reporter,
                          const struct LogScaleConfig* config,
                          struct LogScaleState* state) {
  state->enable_log = config->enable_log;
  state->scale_shift = config->scale_shift;
  return 1;
}
