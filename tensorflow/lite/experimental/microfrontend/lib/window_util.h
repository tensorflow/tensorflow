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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_UTIL_H_

#include "tensorflow/lite/experimental/microfrontend/lib/window.h"

#ifdef __cplusplus
extern "C" {
#endif

struct WindowConfig {
  // length of window frame in milliseconds
  size_t size_ms;
  // length of step for next frame in milliseconds
  size_t step_size_ms;
};

// Populates the WindowConfig with "sane" default values.
void WindowFillConfigWithDefaults(struct WindowConfig* config);

// Allocates any buffers.
int WindowPopulateState(const struct WindowConfig* config,
                        struct WindowState* state, int sample_rate);

// Frees any allocated buffers.
void WindowFreeStateContents(struct WindowState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_WINDOW_UTIL_H_
