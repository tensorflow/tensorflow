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
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/window_util.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/static_alloc.h"

// Needed because some platforms don't have M_PI defined.
#define WINDOW_PI (3.14159265358979323846f)

void WindowFillConfigWithDefaults(struct WindowConfig* config) {
  config->size_ms = 25;
  config->step_size_ms = 10;
}

int WindowPopulateState(tflite::ErrorReporter* error_reporter,
                        const struct WindowConfig* config,
                        struct WindowState* state, int sample_rate) {
  state->size = config->size_ms * sample_rate / 1000;
  state->step = config->step_size_ms * sample_rate / 1000;

  STATIC_ALLOC_ENSURE_ARRAY_SIZE(state->coefficients,
                                 (state->size * sizeof(*state->coefficients)));

  // Populate the window values.
  const float arg = WINDOW_PI * 2.0 / (static_cast<float>(state->size));
  int i;
  for (i = 0; i < state->size; ++i) {
    float float_value = 0.5 - (0.5 * cos(arg * (i + 0.5)));
    // Scale it to fixed point and round it.
    state->coefficients[i] =
        floor(float_value * (1 << kFrontendWindowBits) + 0.5);
  }

  state->input_used = 0;
  STATIC_ALLOC_ENSURE_ARRAY_SIZE(state->input,
                                 (state->size * sizeof(*state->input)));

  STATIC_ALLOC_ENSURE_ARRAY_SIZE(state->output,
                                 (state->size * sizeof(*state->output)));
  return 1;
}
