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

#include "tensorflow/examples/wav_to_spectrogram/wav_to_spectrogram.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/wav/wav_io.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

TEST(WavToSpectrogramTest, WavToSpectrogramTest) {
  const tensorflow::string input_wav =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "input_wav.wav");
  const tensorflow::string output_image = tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(), "output_image.png");
  float audio[8] = {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
  tensorflow::string wav_string;
  TF_ASSERT_OK(
      tensorflow::wav::EncodeAudioAsS16LEWav(audio, 44100, 1, 8, &wav_string));
  TF_ASSERT_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                             input_wav, wav_string));
  TF_ASSERT_OK(WavToSpectrogram(input_wav, 4, 4, 64.0f, output_image));
  TF_EXPECT_OK(tensorflow::Env::Default()->FileExists(output_image));
}
