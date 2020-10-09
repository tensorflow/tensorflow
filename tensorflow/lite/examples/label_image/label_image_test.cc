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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"
#include "tensorflow/lite/examples/label_image/label_image.h"

namespace tflite {
namespace label_image {

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

TEST(LabelImageTest, GetTopN) {
  uint8_t in[] = {1, 1, 2, 2, 4, 4, 16, 32, 128, 64};

  std::vector<std::pair<float, int>> top_results;
  get_top_n<uint8_t>(in, 10, 5, 0.025, &top_results, kTfLiteUInt8);
  ASSERT_EQ(top_results.size(), 4);
  ASSERT_EQ(top_results[0].second, 8);
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
