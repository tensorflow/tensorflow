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

// This data was extracted from the larger feature data held in
// no_features_data.cc and consists of the 26th spectrogram slice of 43 values.
// This is the expected result of running the sample data in
// yes_30ms_sample_data.cc through through the preprocessing pipeline.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_YES_POWER_SPECTRUM_DATA_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_YES_POWER_SPECTRUM_DATA_H_

#include <cstdint>

constexpr int g_yes_power_spectrum_data_size = 43;
extern const uint8_t g_yes_power_spectrum_data[];

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_MICRO_EXAMPLES_MICRO_SPEECH_YES_POWER_SPECTRUM_DATA_H_
