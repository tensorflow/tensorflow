/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/jxl/jxl_io.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace jxl {
namespace {

std::string ReadTestFile(const std::string& filename) {
  std::string file_path = GetDataDependencyFilepath(
      io::JoinPath("tensorflow/core/lib/jxl/testdata", filename));
  std::string data;
  TF_CHECK_OK(ReadFileToString(Env::Default(), file_path, &data));
  return data;
}

TEST(JxlIoTest, HasJxlHeader) {
  std::string jxl_data = ReadTestFile("random_128x96_rbg_q100.jxl");
  EXPECT_TRUE(HasJxlHeader(jxl_data));

  std::string non_jxl = "not a valid jxl header content here";
  EXPECT_FALSE(HasJxlHeader(non_jxl));
}

TEST(JxlIoTest, DecodeHeader) {
  std::string jxl_data = ReadTestFile("random_128x96_rbg_q100.jxl");
  int width = 0, height = 0, channels = 0, bit_depth = 0;
  EXPECT_TRUE(DecodeHeader(jxl_data, &width, &height, &channels, &bit_depth));
  EXPECT_EQ(width, 128);
  EXPECT_EQ(height, 96);
  EXPECT_EQ(channels, 3);
  EXPECT_EQ(bit_depth, 8);

  // Calling with nullptr pointers should succeed safely.
  EXPECT_TRUE(DecodeHeader(jxl_data, nullptr, nullptr, nullptr, nullptr));
  EXPECT_TRUE(DecodeHeader(jxl_data, &width, nullptr, nullptr, nullptr));
  EXPECT_EQ(width, 128);
}

TEST(JxlIoTest, DecodeImageUint8) {
  std::string jxl_data = ReadTestFile("random_128x96_rbg_q100.jxl");
  int width = 0, height = 0, channels = 0;
  ASSERT_TRUE(DecodeHeader(jxl_data, &width, &height, &channels));
  std::vector<uint8_t> output(width * height * channels);
  EXPECT_TRUE(DecodeImage(jxl_data, channels, output.data(), output.size()));
}

TEST(JxlIoTest, DecodeImageUint16) {
  std::string jxl_data = ReadTestFile("random_128x96_rbg_q100.jxl");
  int width = 0, height = 0, channels = 0;
  ASSERT_TRUE(DecodeHeader(jxl_data, &width, &height, &channels));
  std::vector<uint16_t> output(width * height * channels);
  size_t output_bytes = output.size() * sizeof(uint16_t);
  EXPECT_TRUE(DecodeImage16(jxl_data, channels, output.data(), output_bytes));
}

TEST(JxlIoTest, DecodeImageFloat) {
  std::string jxl_data = ReadTestFile("random_128x96_rbg_q100.jxl");
  int width = 0, height = 0, channels = 0;
  ASSERT_TRUE(DecodeHeader(jxl_data, &width, &height, &channels));
  std::vector<float> output(width * height * channels);
  size_t output_bytes = output.size() * sizeof(float);
  EXPECT_TRUE(
      DecodeImageFloat(jxl_data, channels, output.data(), output_bytes));

  std::vector<uint8_t> output_u8(width * height * channels);
  EXPECT_TRUE(
      DecodeImage(jxl_data, channels, output_u8.data(), output_u8.size()));
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_NEAR(output[i], output_u8[i] / 255.0f, 1e-2f);
  }
}

TEST(JxlIoTest, EncodeDecodeLosslessUint8Roundtrip) {
  const int width = 32;
  const int height = 24;
  const int channels = 3;
  std::vector<uint8_t> original(width * height * channels);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<uint8_t>((i * 17 + 31) % 256);
  }

  std::string encoded;
  EXPECT_TRUE(WriteImageToBuffer(original.data(), width, height, channels,
                                 /*channel_bits=*/8, /*distance=*/0.0f,
                                 /*effort=*/3, &encoded));
  EXPECT_FALSE(encoded.empty());
  EXPECT_TRUE(HasJxlHeader(encoded));

  int dec_w = 0, dec_h = 0, dec_c = 0;
  EXPECT_TRUE(DecodeHeader(encoded, &dec_w, &dec_h, &dec_c));
  EXPECT_EQ(dec_w, width);
  EXPECT_EQ(dec_h, height);
  EXPECT_EQ(dec_c, channels);

  std::vector<uint8_t> decoded(width * height * channels);
  EXPECT_TRUE(DecodeImage(encoded, channels, decoded.data(), decoded.size()));
  EXPECT_EQ(original, decoded);
}

TEST(JxlIoTest, EncodeDecodeLosslessUint16Roundtrip) {
  const int width = 32;
  const int height = 24;
  const int channels = 3;
  std::vector<uint16_t> original(width * height * channels);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<uint16_t>((i * 1013 + 7919) % 65536);
  }
  // Include explicit edge cases to verify no 8-bit clamping or bit truncation:
  const uint16_t kEdgeValues[] = {0, 1, 255, 256, 1000, 32768, 65534, 65535};
  for (size_t i = 0; i < sizeof(kEdgeValues) / sizeof(kEdgeValues[0]); ++i) {
    original[i] = kEdgeValues[i];
  }

  std::string encoded;
  EXPECT_TRUE(WriteImageToBuffer(original.data(), width, height, channels,
                                 /*channel_bits=*/16, /*distance=*/0.0f,
                                 /*effort=*/3, &encoded));
  EXPECT_FALSE(encoded.empty());

  std::vector<uint16_t> decoded(width * height * channels);
  size_t output_bytes = decoded.size() * sizeof(uint16_t);
  EXPECT_TRUE(DecodeImage16(encoded, channels, decoded.data(), output_bytes));
  EXPECT_EQ(original, decoded);
}

TEST(JxlIoTest, EncodeWithTstring) {
  const int width = 16;
  const int height = 16;
  const int channels = 1;
  std::vector<uint8_t> original(width * height * channels, 128);

  tstring encoded;
  EXPECT_TRUE(WriteImageToBuffer(original.data(), width, height, channels,
                                 /*channel_bits=*/8, /*distance=*/0.0f,
                                 /*effort=*/3, &encoded));
  EXPECT_GT(encoded.size(), 0);

  std::vector<uint8_t> decoded(width * height * channels);
  EXPECT_TRUE(DecodeImage(encoded, channels, decoded.data(), decoded.size()));
  EXPECT_EQ(original, decoded);
}

TEST(JxlIoTest, EncodeLossy) {
  const int width = 32;
  const int height = 32;
  const int channels = 3;
  std::vector<uint8_t> original(width * height * channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      original[(y * width + x) * channels + 0] = static_cast<uint8_t>(x * 7);
      original[(y * width + x) * channels + 1] = static_cast<uint8_t>(y * 7);
      original[(y * width + x) * channels + 2] =
          static_cast<uint8_t>((x + y) * 3);
    }
  }

  std::string encoded;
  EXPECT_TRUE(WriteImageToBuffer(original.data(), width, height, channels,
                                 /*channel_bits=*/8, /*distance=*/1.0f,
                                 /*effort=*/3, &encoded));
  EXPECT_FALSE(encoded.empty());

  std::vector<uint8_t> decoded(width * height * channels);
  EXPECT_TRUE(DecodeImage(encoded, channels, decoded.data(), decoded.size()));

  double sum_diff = 0.0;
  for (size_t i = 0; i < original.size(); ++i) {
    sum_diff +=
        std::abs(static_cast<int>(original[i]) - static_cast<int>(decoded[i]));
  }
  double mean_diff = sum_diff / original.size();
  EXPECT_LT(mean_diff, 2.0);
}

TEST(JxlIoTest, EncodeDecodeLosslessFloatRoundtrip) {
  const int width = 32;
  const int height = 24;
  const int channels = 3;
  std::vector<float> original(width * height * channels);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = static_cast<float>((i * 17 + 31) % 256) / 255.0f;
  }
  const float kEdgeValues[] = {0.0f, 0.1f, 0.5f, 0.75f, 1.0f};
  for (size_t i = 0; i < sizeof(kEdgeValues) / sizeof(kEdgeValues[0]); ++i) {
    original[i] = kEdgeValues[i];
  }

  std::string encoded;
  EXPECT_TRUE(WriteImageToBuffer(original.data(), width, height, channels,
                                 /*channel_bits=*/32, /*distance=*/0.0f,
                                 /*effort=*/3, &encoded));
  EXPECT_FALSE(encoded.empty());
  EXPECT_TRUE(HasJxlHeader(encoded));

  int dec_w = 0, dec_h = 0, dec_c = 0;
  EXPECT_TRUE(DecodeHeader(encoded, &dec_w, &dec_h, &dec_c));
  EXPECT_EQ(dec_w, width);
  EXPECT_EQ(dec_h, height);
  EXPECT_EQ(dec_c, channels);

  std::vector<float> decoded(width * height * channels);
  EXPECT_TRUE(DecodeImageFloat(encoded, channels, decoded.data(),
                               decoded.size() * sizeof(float)));
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_NEAR(original[i], decoded[i], 1e-5f);
  }
}

TEST(JxlIoTest, EncodeDecodeLosslessFloat16Roundtrip) {
  const int width = 32;
  const int height = 24;
  const int channels = 3;
  std::vector<Eigen::half> original(width * height * channels);
  for (size_t i = 0; i < original.size(); ++i) {
    original[i] = Eigen::half(static_cast<float>((i * 17 + 31) % 256) / 255.0f);
  }
  const float kEdgeValues[] = {0.0f, 0.1f, 0.5f, 0.75f, 1.0f};
  for (size_t i = 0; i < sizeof(kEdgeValues) / sizeof(kEdgeValues[0]); ++i) {
    original[i] = Eigen::half(kEdgeValues[i]);
  }

  std::string encoded;
  EXPECT_TRUE(WriteImageToBuffer(original.data(), width, height, channels,
                                 /*channel_bits=*/16, /*distance=*/0.0f,
                                 /*effort=*/3, &encoded, /*is_float=*/true));
  EXPECT_FALSE(encoded.empty());
  EXPECT_TRUE(HasJxlHeader(encoded));

  int dec_w = 0, dec_h = 0, dec_c = 0;
  EXPECT_TRUE(DecodeHeader(encoded, &dec_w, &dec_h, &dec_c));
  EXPECT_EQ(dec_w, width);
  EXPECT_EQ(dec_h, height);
  EXPECT_EQ(dec_c, channels);

  std::vector<Eigen::half> decoded(width * height * channels);
  EXPECT_TRUE(DecodeImageFloat16(encoded, channels, decoded.data(),
                                 decoded.size() * sizeof(Eigen::half)));
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_NEAR(static_cast<float>(original[i]), static_cast<float>(decoded[i]),
                1e-3f);
  }
}

}  // namespace
}  // namespace jxl
}  // namespace tensorflow
