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

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace wav {

// These are defined in wav_io.cc, and the signatures are here so we don't have
// to expose them in the public header.
Status ExpectText(const string& data, const string& expected_text, int* offset);
Status ReadString(const string& data, int expected_length, string* value,
                  int* offset);

TEST(WavIO, BadArguments) {
  float audio[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  tstring result;

  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(nullptr, 44100, 2, 3, &result).code());
  TF_EXPECT_OK(EncodeAudioAsS16LEWav(nullptr, 44100, 2, 0, &result));

  EXPECT_EQ(
      error::INVALID_ARGUMENT,
      EncodeAudioAsS16LEWav(audio, 44100, 2, 3, (tstring*)nullptr).code());

  const size_t kuint32max_plus_one = static_cast<size_t>(kuint32max) + 1;
  const size_t kuint16max_plus_one = static_cast<size_t>(kuint16max) + 1;

  // Zero values are invalid.
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 0, 2, 3, &result).code());
  EXPECT_EQ(error::INVALID_ARGUMENT,
            EncodeAudioAsS16LEWav(audio, 44100, 0, 3, &result).code());

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

TEST(WavIO, BasicMono) {
  std::vector<uint8> wav_data = {
      'R', 'I', 'F', 'F',  // ChunkID
      44, 0, 0, 0,         // ChunkSize: 36 + SubChunk2Size
      'W', 'A', 'V', 'E',  // Format
      'f', 'm', 't', ' ',  // Subchunk1ID
      16, 0, 0, 0,         // Subchunk1Size
      1, 0,                // AudioFormat: 1=PCM
      1, 0,                // NumChannels
      0x44, 0xac, 0, 0,    // SampleRate: 44100
      0x88, 0x58, 0x1, 0,  // BytesPerSecond: SampleRate * NumChannels *
                           //                 BitsPerSample/8
      2, 0,                // BytesPerSample: NumChannels * BitsPerSample/8
      16, 0,               // BitsPerSample
      'd', 'a', 't', 'a',  // Subchunk2ID
      8, 0, 0, 0,          // Subchunk2Size: NumSamples * NumChannels *
                           //                BitsPerSample/8
      0, 0,                // Sample 1: 0
      0xff, 0x7f,          // Sample 2: 32767 (saturated)
      0, 0,                // Sample 3: 0
      0x00, 0x80,          // Sample 4: -32768 (saturated)
  };
  string expected(wav_data.begin(), wav_data.end());
  float audio[] = {0.0f, 1.0f, 0.0f, -1.0f};
  string result;
  TF_EXPECT_OK(EncodeAudioAsS16LEWav(audio, 44100, 1, 4, &result));
  EXPECT_EQ(expected, result);
}

TEST(WavIO, BasicStereo) {
  std::vector<uint8> wav_data = {
      'R', 'I', 'F', 'F',  // ChunkID
      44, 0, 0, 0,         // ChunkSize: 36 + SubChunk2Size
      'W', 'A', 'V', 'E',  // Format
      'f', 'm', 't', ' ',  // Subchunk1ID
      16, 0, 0, 0,         // Subchunk1Size
      1, 0,                // AudioFormat: 1=PCM
      2, 0,                // NumChannels
      0x44, 0xac, 0, 0,    // SampleRate: 44100
      0x10, 0xb1, 0x2, 0,  // BytesPerSecond: SampleRate * NumChannels *
                           //                 BitsPerSample/8
      4, 0,                // BytesPerSample: NumChannels * BitsPerSample/8
      16, 0,               // BitsPerSample
      'd', 'a', 't', 'a',  // Subchunk2ID
      8, 0, 0, 0,          // Subchunk2Size: NumSamples * NumChannels *
                           //                BitsPerSample/8
      0, 0,                // Sample 1: 0
      0xff, 0x7f,          // Sample 2: 32767 (saturated)
      0, 0,                // Sample 3: 0
      0x00, 0x80,          // Sample 4: -32768 (saturated)
  };
  string expected(wav_data.begin(), wav_data.end());
  float audio[] = {0.0f, 1.0f, 0.0f, -1.0f};
  string result;
  TF_EXPECT_OK(EncodeAudioAsS16LEWav(audio, 44100, 2, 2, &result));
  EXPECT_EQ(expected, result);
}

// Test how chunk sizes larger than 2GB are handled, since they're stored as
// unsigned int32s, so there are lots of ways for conversions to confuse the
// decoding logic. The expected behavior is to fail with an error, since such
// large WAV files are not common, and are unsupported by many readers.
// See b/72655902.
TEST(WavIO, ChunkSizeOverflow) {
  std::vector<uint8> wav_data = {
      'R', 'I', 'F', 'F',      // ChunkID
      60, 0, 0, 0,             // ChunkSize: 36 + SubChunk2Size
      'W', 'A', 'V', 'E',      // Format
      'f', 'm', 't', ' ',      // Subchunk1ID
      16, 0, 0, 0,             // Subchunk1Size
      1, 0,                    // AudioFormat: 1=PCM
      1, 0,                    // NumChannels
      0x44, 0xac, 0, 0,        // SampleRate: 44100
      0x88, 0x58, 0x1, 0,      // BytesPerSecond: SampleRate * NumChannels *
                               //                 BitsPerSample/8
      2, 0,                    // BytesPerSample: NumChannels * BitsPerSample/8
      16, 0,                   // BitsPerSample
      'd', 'a', 't', 'a',      // Subchunk2ID
      8, 0, 0, 0,              // Subchunk2Size: NumSamples * NumChannels *
                               //                BitsPerSample/8
      0, 0,                    // Sample 1: 0
      0xff, 0x7f,              // Sample 2: 32767 (saturated)
      0, 0,                    // Sample 3: 0
      0x00, 0x80,              // Sample 4: -32768 (saturated)
      'f', 'o', 'o', 'o',      // Subchunk2ID
      0xff, 0xff, 0xff, 0xf8,  // Chunk size that could cause an infinite loop.
      0, 0,                    // Sample 1: 0
      0xff, 0x7f,              // Sample 2: 32767 (saturated)
      0, 0,                    // Sample 3: 0
      0x00, 0x80,              // Sample 4: -32768 (saturated)
  };
  string wav_data_string(wav_data.begin(), wav_data.end());
  std::vector<float> decoded_audio;
  uint32 decoded_sample_count;
  uint16 decoded_channel_count;
  uint32 decoded_sample_rate;
  Status decode_status = DecodeLin16WaveAsFloatVector(
      wav_data_string, &decoded_audio, &decoded_sample_count,
      &decoded_channel_count, &decoded_sample_rate);
  EXPECT_FALSE(decode_status.ok());
  EXPECT_TRUE(absl::StrContains(decode_status.message(), "too large"))
      << decode_status.message();
}

TEST(WavIO, IncrementOffset) {
  int new_offset = -1;
  TF_EXPECT_OK(IncrementOffset(0, 10, 20, &new_offset));
  EXPECT_EQ(10, new_offset);

  new_offset = -1;
  TF_EXPECT_OK(IncrementOffset(10, 4, 20, &new_offset));
  EXPECT_EQ(14, new_offset);

  new_offset = -1;
  TF_EXPECT_OK(IncrementOffset(99, 1, 100, &new_offset));
  EXPECT_EQ(100, new_offset);

  new_offset = -1;
  EXPECT_FALSE(IncrementOffset(-1, 1, 100, &new_offset).ok());

  new_offset = -1;
  EXPECT_FALSE(IncrementOffset(0, -1, 100, &new_offset).ok());

  new_offset = -1;
  EXPECT_FALSE(IncrementOffset(std::numeric_limits<int>::max(), 1,
                               std::numeric_limits<int>::max(), &new_offset)
                   .ok());

  new_offset = -1;
  EXPECT_FALSE(IncrementOffset(101, 1, 100, &new_offset).ok());
}

TEST(WavIO, ExpectText) {
  std::vector<uint8> test_data = {
      'E', 'x', 'p', 'e', 'c', 't', 'e', 'd',
  };
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  TF_EXPECT_OK(ExpectText(test_string, "Expected", &offset));
  EXPECT_EQ(8, offset);

  offset = 0;
  Status expect_status = ExpectText(test_string, "Unexpected", &offset);
  EXPECT_FALSE(expect_status.ok());

  offset = 0;
  TF_EXPECT_OK(ExpectText(test_string, "Exp", &offset));
  EXPECT_EQ(3, offset);
  TF_EXPECT_OK(ExpectText(test_string, "ected", &offset));
  EXPECT_EQ(8, offset);
  expect_status = ExpectText(test_string, "foo", &offset);
  EXPECT_FALSE(expect_status.ok());
}

TEST(WavIO, ReadString) {
  std::vector<uint8> test_data = {
      'E', 'x', 'p', 'e', 'c', 't', 'e', 'd',
  };
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  string read_value;
  TF_EXPECT_OK(ReadString(test_string, 2, &read_value, &offset));
  EXPECT_EQ("Ex", read_value);
  EXPECT_EQ(2, offset);

  TF_EXPECT_OK(ReadString(test_string, 6, &read_value, &offset));
  EXPECT_EQ("pected", read_value);
  EXPECT_EQ(8, offset);

  Status read_status = ReadString(test_string, 3, &read_value, &offset);
  EXPECT_FALSE(read_status.ok());
}

TEST(WavIO, ReadValueInt8) {
  std::vector<uint8> test_data = {0x00, 0x05, 0xff, 0x80};
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  int8_t read_value;
  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(0, read_value);
  EXPECT_EQ(1, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(5, read_value);
  EXPECT_EQ(2, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(-1, read_value);
  EXPECT_EQ(3, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(-128, read_value);
  EXPECT_EQ(4, offset);

  Status read_status = ReadValue(test_string, &read_value, &offset);
  EXPECT_FALSE(read_status.ok());
}

TEST(WavIO, ReadValueUInt8) {
  std::vector<uint8> test_data = {0x00, 0x05, 0xff, 0x80};
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  uint8 read_value;
  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(0, read_value);
  EXPECT_EQ(1, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(5, read_value);
  EXPECT_EQ(2, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(255, read_value);
  EXPECT_EQ(3, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(128, read_value);
  EXPECT_EQ(4, offset);

  Status read_status = ReadValue(test_string, &read_value, &offset);
  EXPECT_FALSE(read_status.ok());
}

TEST(WavIO, ReadValueInt16) {
  std::vector<uint8> test_data = {
      0x00, 0x00,  // 0
      0xff, 0x00,  // 255
      0x00, 0x01,  // 256
      0xff, 0xff,  // -1
      0x00, 0x80,  // -32768
  };
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  int16_t read_value;
  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(0, read_value);
  EXPECT_EQ(2, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(255, read_value);
  EXPECT_EQ(4, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(256, read_value);
  EXPECT_EQ(6, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(-1, read_value);
  EXPECT_EQ(8, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(-32768, read_value);
  EXPECT_EQ(10, offset);

  Status read_status = ReadValue(test_string, &read_value, &offset);
  EXPECT_FALSE(read_status.ok());
}

TEST(WavIO, ReadValueUInt16) {
  std::vector<uint8> test_data = {
      0x00, 0x00,  // 0
      0xff, 0x00,  // 255
      0x00, 0x01,  // 256
      0xff, 0xff,  // 65535
      0x00, 0x80,  // 32768
  };
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  uint16 read_value;
  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(0, read_value);
  EXPECT_EQ(2, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(255, read_value);
  EXPECT_EQ(4, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(256, read_value);
  EXPECT_EQ(6, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(65535, read_value);
  EXPECT_EQ(8, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(32768, read_value);
  EXPECT_EQ(10, offset);

  Status read_status = ReadValue(test_string, &read_value, &offset);
  EXPECT_FALSE(read_status.ok());
}

TEST(WavIO, ReadValueInt32) {
  std::vector<uint8> test_data = {
      0x00, 0x00, 0x00, 0x00,  // 0
      0xff, 0x00, 0x00, 0x00,  // 255
      0x00, 0xff, 0x00, 0x00,  // 65280
      0x00, 0x00, 0xff, 0x00,  // 16,711,680
      0xff, 0xff, 0xff, 0xff,  // -1
  };
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  int32_t read_value;
  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(0, read_value);
  EXPECT_EQ(4, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(255, read_value);
  EXPECT_EQ(8, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(65280, read_value);
  EXPECT_EQ(12, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(16711680, read_value);
  EXPECT_EQ(16, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(-1, read_value);
  EXPECT_EQ(20, offset);

  Status read_status = ReadValue(test_string, &read_value, &offset);
  EXPECT_FALSE(read_status.ok());
}

TEST(WavIO, ReadValueUInt32) {
  std::vector<uint8> test_data = {
      0x00, 0x00, 0x00, 0x00,  // 0
      0xff, 0x00, 0x00, 0x00,  // 255
      0x00, 0xff, 0x00, 0x00,  // 65280
      0x00, 0x00, 0xff, 0x00,  // 16,711,680
      0xff, 0xff, 0xff, 0xff,  // 4,294,967,295
  };
  string test_string(test_data.begin(), test_data.end());

  int offset = 0;
  uint32 read_value;
  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(0, read_value);
  EXPECT_EQ(4, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(255, read_value);
  EXPECT_EQ(8, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(65280, read_value);
  EXPECT_EQ(12, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(16711680, read_value);
  EXPECT_EQ(16, offset);

  TF_EXPECT_OK(ReadValue(test_string, &read_value, &offset));
  EXPECT_EQ(4294967295, read_value);
  EXPECT_EQ(20, offset);

  Status read_status = ReadValue(test_string, &read_value, &offset);
  EXPECT_FALSE(read_status.ok());
}

}  // namespace wav
}  // namespace tensorflow
