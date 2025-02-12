/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_EXAMPLES_WAV_TO_SPECTROGRAM_WAV_TO_SPECTROGRAM_H_
#define TENSORFLOW_EXAMPLES_WAV_TO_SPECTROGRAM_WAV_TO_SPECTROGRAM_H_

#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

// Runs a TensorFlow graph to convert an audio file into a visualization. Takes
// in the path to the audio file, the window size and stride parameters
// controlling the spectrogram creation, the brightness scaling to use, and a
// path to save the output PNG file to.
absl::Status WavToSpectrogram(const tensorflow::string& input_wav,
                              int32_t window_size, int32_t stride,
                              float brightness,
                              const tensorflow::string& output_image);

#endif  // TENSORFLOW_EXAMPLES_WAV_TO_SPECTROGRAM_WAV_TO_SPECTROGRAM_H_
