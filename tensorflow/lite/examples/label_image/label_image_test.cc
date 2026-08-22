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

#include "tensorflow/lite/examples/label_image/label_image.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"

namespace tflite {
namespace label_image {
namespace {

std::string WriteTestBmp(const std::vector<uint8_t>& bytes,
                         const std::string& name) {
  const testing::TestInfo* test_info =
      testing::UnitTest::GetInstance()->current_test_info();
  const std::string filename =
      ::testing::TempDir() + test_info->test_suite_name() + "_" +
      test_info->name() + "_" + name + ".bmp";
  std::ofstream file(filename, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  return filename;
}

std::vector<uint8_t> ValidBmpHeader(int32_t pixel_offset, int32_t width,
                                    int32_t height, uint16_t bpp) {
  std::vector<uint8_t> bytes(
      std::max<size_t>(pixel_offset < 0 ? 30 : pixel_offset, 30), 0);
  bytes[0] = 'B';
  bytes[1] = 'M';
  auto write_le16 = [&bytes](size_t offset, uint16_t value) {
    bytes[offset] = value & 0xff;
    bytes[offset + 1] = value >> 8;
  };
  auto write_le32 = [&bytes](size_t offset, int32_t value) {
    const uint32_t unsigned_value = static_cast<uint32_t>(value);
    bytes[offset] = unsigned_value & 0xff;
    bytes[offset + 1] = (unsigned_value >> 8) & 0xff;
    bytes[offset + 2] = (unsigned_value >> 16) & 0xff;
    bytes[offset + 3] = (unsigned_value >> 24) & 0xff;
  };
  write_le32(10, pixel_offset);
  write_le32(18, width);
  write_le32(22, height);
  write_le16(28, bpp);
  return bytes;
}

}  // namespace

TEST(LabelImageTest, GraceHopper) {
  std::string lena_file =
      "tensorflow/lite/examples/label_image/testdata/"
      "grace_hopper.bmp";
  int height, width, channels;
  Settings s;
  s.input_type = kTfLiteUInt8;
  std::vector<uint8_t> input =
      read_bmp(lena_file, &width, &height, &channels, &s);
  ASSERT_EQ(height, 606);
  ASSERT_EQ(width, 517);
  ASSERT_EQ(channels, 3);

  std::vector<uint8_t> output(606 * 517 * 3);
  resize<uint8_t>(output.data(), input.data(), 606, 517, 3, 214, 214, 3, &s);
  ASSERT_EQ(output[0], 0x15);
  ASSERT_EQ(output[214 * 214 * 3 - 1], 0x11);
}

TEST(LabelImageTest, RejectsTruncatedBmpHeader) {
  const std::string filename = WriteTestBmp({'B', 'M'}, "truncated_header");
  int height, width, channels;
  Settings s;
  auto result = read_bmp(filename, &width, &height, &channels, &s);
  EXPECT_TRUE(result.empty());
}

TEST(LabelImageTest, RejectsPixelDataOutsideFile) {
  std::vector<uint8_t> bytes = ValidBmpHeader(128, 1, 1, 24);
  const std::string filename = WriteTestBmp(bytes, "bad_pixel_offset");
  int height, width, channels;
  Settings s;
  auto result = read_bmp(filename, &width, &height, &channels, &s);
  EXPECT_TRUE(result.empty());
}

TEST(LabelImageTest, RejectsShortPixelData) {
  std::vector<uint8_t> bytes = ValidBmpHeader(54, 2, 2, 24);
  bytes.resize(54 + 8);
  const std::string filename = WriteTestBmp(bytes, "short_pixel_data");
  int height, width, channels;
  Settings s;
  auto result = read_bmp(filename, &width, &height, &channels, &s);
  EXPECT_TRUE(result.empty());
}

TEST(LabelImageTest, RejectsRowSizeOverflow) {
  std::vector<uint8_t> bytes =
      ValidBmpHeader(54, std::numeric_limits<int32_t>::max(), 1, 32);
  const std::string filename = WriteTestBmp(bytes, "row_size_overflow");
  int height, width, channels;
  Settings s;
  auto result = read_bmp(filename, &width, &height, &channels, &s);
  EXPECT_TRUE(result.empty());
}

TEST(LabelImageTest, GetTopN) {
  uint8_t in[] = {1, 1, 2, 2, 4, 4, 16, 32, 128, 64};

  std::vector<std::pair<float, int>> top_results;
  get_top_n<uint8_t>(in, 10, 5, 0.025, &top_results, kTfLiteUInt8);
  ASSERT_EQ(top_results.size(), 4);
  ASSERT_EQ(top_results[0].second, 8);
}

}  // namespace label_image
}  // namespace tflite
