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
#include "tensorflow/lite/experimental/microfrontend/lib/noise_reduction_io.h"

void NoiseReductionWriteMemmapPreamble(
    FILE* fp, const struct NoiseReductionState* state) {
  fprintf(fp, "static uint32_t noise_reduction_estimate[%d];\n",
          state->num_channels);
  fprintf(fp, "\n");
}

void NoiseReductionWriteMemmap(FILE* fp,
                               const struct NoiseReductionState* state,
                               const char* variable) {
  fprintf(fp, "%s->even_smoothing = %d;\n", variable, state->even_smoothing);
  fprintf(fp, "%s->odd_smoothing = %d;\n", variable, state->odd_smoothing);
  fprintf(fp, "%s->min_signal_remaining = %d;\n", variable,
          state->min_signal_remaining);
  fprintf(fp, "%s->num_channels = %d;\n", variable, state->num_channels);

  fprintf(fp, "%s->estimate = noise_reduction_estimate;\n", variable);
}
