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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_UTIL_H_

#include "tensorflow/lite/experimental/microfrontend/lib/filterbank.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FilterbankConfig {
  // number of frequency channel buckets for filterbank
  int num_channels;
  // maximum frequency to include
  float upper_band_limit;
  // minimum frequency to include
  float lower_band_limit;
  // unused
  int output_scale_shift;
};

// Fills the frontendConfig with "sane" defaults.
void FilterbankFillConfigWithDefaults(struct FilterbankConfig* config);

// Allocates any buffers.
int FilterbankPopulateState(const struct FilterbankConfig* config,
                            struct FilterbankState* state, int sample_rate,
                            int spectrum_size);

// Frees any allocated buffers.
void FilterbankFreeStateContents(struct FilterbankState* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICROFRONTEND_LIB_FILTERBANK_UTIL_H_
