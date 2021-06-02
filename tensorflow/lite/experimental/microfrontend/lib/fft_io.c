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
#include "tensorflow/lite/experimental/microfrontend/lib/fft_io.h"

void FftWriteMemmapPreamble(FILE* fp, const struct FftState* state) {
  MICROFRONTEND_FPRINTF(fp, "static int16_t fft_input[%zu];\n",
                        state->fft_size);
  MICROFRONTEND_FPRINTF(fp, "static struct complex_int16_t fft_output[%zu];\n",
                        state->fft_size / 2 + 1);
  MICROFRONTEND_FPRINTF(fp, "static char fft_scratch[%zu];\n",
                        state->scratch_size);
  MICROFRONTEND_FPRINTF(fp, "\n");
}

void FftWriteMemmap(FILE* fp, const struct FftState* state,
                    const char* variable) {
  MICROFRONTEND_FPRINTF(fp, "%s->input = fft_input;\n", variable);
  MICROFRONTEND_FPRINTF(fp, "%s->output = fft_output;\n", variable);
  MICROFRONTEND_FPRINTF(fp, "%s->fft_size = %zu;\n", variable, state->fft_size);
  MICROFRONTEND_FPRINTF(fp, "%s->input_size = %zu;\n", variable,
                        state->input_size);
  MICROFRONTEND_FPRINTF(fp, "%s->scratch = fft_scratch;\n", variable);
  MICROFRONTEND_FPRINTF(fp, "%s->scratch_size = %zu;\n", variable,
                        state->scratch_size);
}
