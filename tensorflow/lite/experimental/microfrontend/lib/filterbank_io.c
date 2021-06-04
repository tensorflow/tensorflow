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
#include "tensorflow/lite/experimental/microfrontend/lib/filterbank_io.h"

static void PrintArray(FILE* fp, const char* name, const int16_t* values,
                       size_t size) {
  MICROFRONTEND_FPRINTF(fp, "static int16_t filterbank_%s[] = {", name);
  int i;
  for (i = 0; i < size; ++i) {
    MICROFRONTEND_FPRINTF(fp, "%d", values[i]);
    if (i < size - 1) {
      MICROFRONTEND_FPRINTF(fp, ", ");
    }
  }
  MICROFRONTEND_FPRINTF(fp, "};\n");
}

void FilterbankWriteMemmapPreamble(FILE* fp,
                                   const struct FilterbankState* state) {
  const int num_channels_plus_1 = state->num_channels + 1;

  PrintArray(fp, "channel_frequency_starts", state->channel_frequency_starts,
             num_channels_plus_1);
  PrintArray(fp, "channel_weight_starts", state->channel_weight_starts,
             num_channels_plus_1);
  PrintArray(fp, "channel_widths", state->channel_widths, num_channels_plus_1);
  int num_weights = 0;
  int i;
  for (i = 0; i < num_channels_plus_1; ++i) {
    num_weights += state->channel_widths[i];
  }
  PrintArray(fp, "weights", state->weights, num_weights);
  PrintArray(fp, "unweights", state->unweights, num_weights);

  MICROFRONTEND_FPRINTF(fp, "static uint64_t filterbank_work[%d];\n",
                        num_channels_plus_1);
  MICROFRONTEND_FPRINTF(fp, "\n");
}

void FilterbankWriteMemmap(FILE* fp, const struct FilterbankState* state,
                           const char* variable) {
  MICROFRONTEND_FPRINTF(fp, "%s->num_channels = %d;\n", variable,
                        state->num_channels);
  MICROFRONTEND_FPRINTF(fp, "%s->start_index = %d;\n", variable,
                        state->start_index);
  MICROFRONTEND_FPRINTF(fp, "%s->end_index = %d;\n", variable,
                        state->end_index);

  MICROFRONTEND_FPRINTF(
      fp,
      "%s->channel_frequency_starts = filterbank_channel_frequency_starts;\n",
      variable);
  MICROFRONTEND_FPRINTF(
      fp, "%s->channel_weight_starts = filterbank_channel_weight_starts;\n",
      variable);
  MICROFRONTEND_FPRINTF(fp, "%s->channel_widths = filterbank_channel_widths;\n",
                        variable);
  MICROFRONTEND_FPRINTF(fp, "%s->weights = filterbank_weights;\n", variable);
  MICROFRONTEND_FPRINTF(fp, "%s->unweights = filterbank_unweights;\n",
                        variable);
  MICROFRONTEND_FPRINTF(fp, "%s->work = filterbank_work;\n", variable);
}
