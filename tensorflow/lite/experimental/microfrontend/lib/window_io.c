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
#include "tensorflow/lite/experimental/microfrontend/lib/window_io.h"

void WindowWriteMemmapPreamble(FILE* fp, const struct WindowState* state) {
  fprintf(fp, "static int16_t window_coefficients[] = {\n");
  for (int i = 0; i < state->size; ++i) {
    fprintf(fp, "%d", state->coefficients[i]);
    if (i < state->size - 1) {
      fprintf(fp, ", ");
    }
  }
  fprintf(fp, "};\n");
  fprintf(fp, "static int16_t window_input[%zu];\n", state->size);
  fprintf(fp, "static int16_t window_output[%zu];\n", state->size);
  fprintf(fp, "\n");
}

void WindowWriteMemmap(FILE* fp, const struct WindowState* state,
                       const char* variable) {
  fprintf(fp, "%s->size = %zu;\n", variable, state->size);
  fprintf(fp, "%s->coefficients = window_coefficients;\n", variable);
  fprintf(fp, "%s->step = %zu;\n", variable, state->step);

  fprintf(fp, "%s->input = window_input;\n", variable);
  fprintf(fp, "%s->input_used = %zu;\n", variable, state->input_used);
  fprintf(fp, "%s->output = window_output;\n", variable);
  fprintf(fp, "%s->max_abs_output_value = %d;\n", variable,
          state->max_abs_output_value);
}
