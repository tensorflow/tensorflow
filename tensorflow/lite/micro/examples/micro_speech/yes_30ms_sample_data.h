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

// This data was created from the PCM data in a WAV file held in v2 of the
// Speech Commands test dataset, at the path:
// speech_commands_test_set_v0.02/yes/f2e59fea_nohash_1.wav
// The data was extracted starting at an offset of 8,000, which corresponds to
// the 26th spectrogram slice. It's designed to be used to test the
// preprocessing pipeline, to ensure that the expected spectrogram slice is
// produced given this input.

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_YES_30MS_SAMPLE_DATA_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_YES_30MS_SAMPLE_DATA_H_

#include <cstdint>

extern const int g_yes_30ms_sample_data_size;
extern const int16_t g_yes_30ms_sample_data[];

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_YES_30MS_SAMPLE_DATA_H_
