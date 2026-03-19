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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_UTIL_H_

#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/experimental/microfrontend/lib/log_scale.h"

#ifdef __cplusplus
extern "C" {
#endif

struct LogScaleConfig {
  // set to false (0) to disable this module
  int enable_log;
  // scale results by 2^(scale_shift)
  int scale_shift;
};

// Populates the LogScaleConfig with "sane" default values.
void LogScaleFillConfigWithDefaults(struct LogScaleConfig* config);

// Allocates any buffers.
int LogScalePopulateState(const struct LogScaleConfig* config,
                          struct LogScaleState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_LOG_SCALE_UTIL_H_
