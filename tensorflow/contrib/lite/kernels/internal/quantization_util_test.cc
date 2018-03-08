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
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace {

using ::testing::Pair;

TEST(QuantizationUtilTest, QuantizeMultiplierSmallerThanOne) {
  auto quantize = [](double d) {
    int32_t q;
    int s;
    QuantizeMultiplierSmallerThanOne(d, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  EXPECT_DEATH(quantize(-0.1), "");
  EXPECT_DEATH(quantize(0.0), "");
  EXPECT_THAT(quantize(0.25), Pair(1073741824, 1));

  // Around 0.5 we can see the change in exponent and how we try hard to
  // void hitting max int32.
  EXPECT_THAT(quantize(0.50 - 5e-9), Pair(2147483627, 1));
  EXPECT_THAT(quantize(0.50 - 1e-10), Pair(1073741824, 0));
  EXPECT_THAT(quantize(0.50), Pair(1073741824, 0));

  EXPECT_THAT(quantize(0.75), Pair(1610612736, 0));
  EXPECT_THAT(quantize(1 - 1e-9), Pair(2147483646, 0));

  // If we get close enough to 1.0 it crashes and dies in one of two ways:
  // Either the shift becomes negative or we trigger the 'less-than-one' CHECK.
  EXPECT_DEATH(quantize(1 - 1e-15), "");
  EXPECT_DEATH(quantize(1 - 1e-17), "");
  EXPECT_DEATH(quantize(1.0), "");
}

TEST(QuantizationUtilTest, QuantizeMultiplierGreaterThanOne) {
  auto quantize = [](double d) {
    int32_t q;
    int s;
    QuantizeMultiplierGreaterThanOne(d, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  // If we are close enough to 1.0 it crashes.
  EXPECT_DEATH(quantize(1 + 1e-16), "");

  EXPECT_THAT(quantize(1 + 1e-11), Pair(1073741824, 1));
  EXPECT_THAT(quantize(1.25), Pair(1342177280, 1));
  EXPECT_THAT(quantize(1.50), Pair(1610612736, 1));
  EXPECT_THAT(quantize(1.75), Pair(1879048192, 1));

  // Around the powers of two we see the change in exponent. Also,
  // we try hard to avoid hitting max int32.
  EXPECT_THAT(quantize(2 - 1e-9), Pair(2147483647, 1));
  EXPECT_THAT(quantize(2 - 1e-11), Pair(1073741824, 2));
  EXPECT_THAT(quantize(2), Pair(1073741824, 2));
}

TEST(QuantizationUtilTest, PreprocessSoftmaxScaling) {
  auto quantize = [](double beta, double scale, int integer_bits) {
    int32_t q;
    int s;
    PreprocessSoftmaxScaling(beta, scale, integer_bits, &q, &s);
    return std::pair<int32_t, int>{q, s};
  };

  // If beta * scale is greater than fits in the number of integer bits, the
  // result is move near the maximum. Otherwise they quantize as expected.
  // With 4 integer bits we can represent up to 16.0.
  EXPECT_THAT(quantize(1.0, 16.0, 4), Pair(2147483647, 31));
  EXPECT_THAT(quantize(1.0, 8.0, 4), Pair(1073741824, 31));
  // But with 5 bits we can go further.
  EXPECT_THAT(quantize(2.0, 16.0, 5), Pair(2147483647, 31));
  EXPECT_THAT(quantize(2.0, 8.0, 5), Pair(1073741824, 31));
}

TEST(QuantizationUtilTest, CalculateInputRadius) {
  EXPECT_EQ(CalculateInputRadius(4, 27), 15);
  EXPECT_EQ(CalculateInputRadius(3, 27), 14);
  EXPECT_EQ(CalculateInputRadius(3, 28), 7);
  EXPECT_EQ(CalculateInputRadius(4, 2), 503316480);
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
