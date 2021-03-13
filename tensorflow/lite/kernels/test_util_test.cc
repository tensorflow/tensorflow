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
#include "tensorflow/lite/kernels/test_util.h"

#include <stdint.h>

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(TestUtilTest, QuantizeVector) {
  std::vector<float> data = {-1.0, -0.5, 0.0, 0.5, 1.0, 1000.0};
  auto q_data = Quantize<uint8_t>(data, /*scale=*/1.0, /*zero_point=*/0);
  std::vector<uint8_t> expected = {0, 0, 0, 1, 1, 255};
  EXPECT_THAT(q_data, ElementsAreArray(expected));
}

TEST(TestUtilTest, QuantizeVectorScalingDown) {
  std::vector<float> data = {-1.0, -0.5, 0.0, 0.5, 1.0, 1000.0};
  auto q_data = Quantize<uint8_t>(data, /*scale=*/10.0, /*zero_point=*/0);
  std::vector<uint8_t> expected = {0, 0, 0, 0, 0, 100};
  EXPECT_THAT(q_data, ElementsAreArray(expected));
}

TEST(TestUtilTest, QuantizeVectorScalingUp) {
  std::vector<float> data = {-1.0, -0.5, 0.0, 0.5, 1.0, 1000.0};
  auto q_data = Quantize<uint8_t>(data, /*scale=*/0.1, /*zero_point=*/0);
  std::vector<uint8_t> expected = {0, 0, 0, 5, 10, 255};
  EXPECT_THAT(q_data, ElementsAreArray(expected));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
