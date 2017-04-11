/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/wav/wav_io.h"

#include <string>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace wav {

TEST(WavIO, BadArguments) {
  float audio[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  string result;

  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(nullptr, 44100, 2, 3, &result).code());
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 44100, 2, 3, nullptr).code());

  const size_t kuint32max_plus_one = static_cast<size_t>(kuint32max) + 1;
  const size_t kuint16max_plus_one = static_cast<size_t>(kuint16max) + 1;

  // Zero values are invalid.
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 0, 2, 3, &result).code());
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 44100, 0, 3, &result).code());
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 44100, 2, 0, &result).code());

  // Sample rates 2^32 and greater are invalid.
  EXPECT_EQ(
      error::INVALID_ARGUMENT,
      EncodeAudioAsS16LEWav(audio, kuint32max_plus_one, 2, 3, &result).code());

  // Channels 2^16 and greater are invalid.
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 44100, kuint16max_plus_one, 3, &result)
                .code());

  // Frames that push the file size above 2^32 are invalid.
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 44100, 2, 1073741813, &result).code());
}

TEST(WavIO, BasicEven) {
  float audio[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  string result;
  TF_EXPECT_OK(EncodeAudioAsS16LEWav(audio, 44100, 2, 3, &result));
  EXPECT_EQ(56, result.size());
  TF_EXPECT_OK(EncodeAudioAsS16LEWav(audio, 22050, 1, 6, &result));
  EXPECT_EQ(56, result.size());
  TF_EXPECT_OK(EncodeAudioAsS16LEWav(audio, 8000, 1, 6, &result));
  EXPECT_EQ(56, result.size());
}

TEST(WavIO, BasicOdd) {
  float audio[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f};
  string result;
  TF_EXPECT_OK(EncodeAudioAsS16LEWav(audio, 22050, 1, 5, &result));
  EXPECT_EQ(54, result.size());
}

TEST(WavIO, EncodeThenDecode) {
  float audio[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  string wav_data;
  TF_ASSERT_OK(EncodeAudioAsS16LEWav(audio, 44100, 2, 3, &wav_data));
  std::vector<float> decoded_audio;
  uint32 decoded_sample_count;
  uint16 decoded_channel_count;
  uint32 decoded_sample_rate;
  TF_ASSERT_OK(DecodeLin16WaveAsFloatVector(
      wav_data, &decoded_audio, &decoded_sample_count, &decoded_channel_count,
      &decoded_sample_rate));
  EXPECT_EQ(2, decoded_channel_count);
  EXPECT_EQ(3, decoded_sample_count);
  EXPECT_EQ(44100, decoded_sample_rate);
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(audio[i], decoded_audio[i], 1e-4f) << "i=" << i;
  }
}

}  // namespace wav
}  // namespace tensorflow
