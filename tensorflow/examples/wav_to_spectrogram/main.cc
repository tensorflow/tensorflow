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

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/examples/wav_to_spectrogram/wav_to_spectrogram.h"

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  tensorflow::string input_wav =
      "tensorflow/core/kernels/spectrogram_test_data/short_test_segment.wav";
  int32_t window_size = 256;
  int32_t stride = 128;
  float brightness = 64.0f;
  tensorflow::string output_image = "spectrogram.png";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("input_wav", &input_wav, "audio file to load"),
      tensorflow::Flag("window_size", &window_size,
                       "frequency sample window width"),
      tensorflow::Flag("stride", &stride,
                       "how far apart to place frequency windows"),
      tensorflow::Flag("brightness", &brightness,
                       "controls how bright the output image is"),
      tensorflow::Flag("output_image", &output_image,
                       "where to save the spectrogram image to"),
  };
  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  absl::Status wav_status = WavToSpectrogram(input_wav, window_size, stride,
                                             brightness, output_image);
  if (!wav_status.ok()) {
    LOG(ERROR) << "WavToSpectrogram failed with " << wav_status;
    return -1;
  }

  return 0;
}
