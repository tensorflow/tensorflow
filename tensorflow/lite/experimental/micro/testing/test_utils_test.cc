/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(F2QTest) {
  using tflite::testing::F2Q;
  // [0, 127.5] -> zero_point=0, scale=0.5
  TF_LITE_MICRO_EXPECT_EQ(0, F2Q(0, 0, 127.5));
  TF_LITE_MICRO_EXPECT_EQ(254, F2Q(127, 0, 127.5));
  TF_LITE_MICRO_EXPECT_EQ(255, F2Q(127.5, 0, 127.5));
  // [-10, 245] -> zero_point=-10, scale=1.0
  TF_LITE_MICRO_EXPECT_EQ(0, F2Q(-10, -10, 245));
  TF_LITE_MICRO_EXPECT_EQ(1, F2Q(-9, -10, 245));
  TF_LITE_MICRO_EXPECT_EQ(128, F2Q(118, -10, 245));
  TF_LITE_MICRO_EXPECT_EQ(253, F2Q(243, -10, 245));
  TF_LITE_MICRO_EXPECT_EQ(254, F2Q(244, -10, 245));
  TF_LITE_MICRO_EXPECT_EQ(255, F2Q(245, -10, 245));
}

TF_LITE_MICRO_TEST(F2QSTest) {
  using tflite::testing::F2QS;
  // [-64, 63.5] -> zero_point=0, scale=0.5
  TF_LITE_MICRO_EXPECT_EQ(2, F2QS(1, -64, 63.5));
  TF_LITE_MICRO_EXPECT_EQ(4, F2QS(2, -64, 63.5));
  TF_LITE_MICRO_EXPECT_EQ(6, F2QS(3, -64, 63.5));
  TF_LITE_MICRO_EXPECT_EQ(-10, F2QS(-5, -64, 63.5));
  TF_LITE_MICRO_EXPECT_EQ(-128, F2QS(-64, -64, 63.5));
  TF_LITE_MICRO_EXPECT_EQ(127, F2QS(63.5, -64, 63.5));
  // [-127, 128] -> zero_point=1, scale=1.0
  TF_LITE_MICRO_EXPECT_EQ(0, F2QS(1, -127, 128));
  TF_LITE_MICRO_EXPECT_EQ(-1, F2QS(0, -127, 128));
  TF_LITE_MICRO_EXPECT_EQ(126, F2QS(127, -127, 128));
  TF_LITE_MICRO_EXPECT_EQ(127, F2QS(128, -127, 128));
  TF_LITE_MICRO_EXPECT_EQ(-127, F2QS(-126, -127, 128));
  TF_LITE_MICRO_EXPECT_EQ(-128, F2QS(-127, -127, 128));
}

TF_LITE_MICRO_TEST(F2Q32Test) {
  using tflite::testing::F2Q32;
  TF_LITE_MICRO_EXPECT_EQ(0, F2Q32(0, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(2, F2Q32(1, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(-2, F2Q32(-1, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(-100, F2Q32(-50, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(100, F2Q32(50, 0.5));
}

TF_LITE_MICRO_TEST(ZeroPointTest) {
  TF_LITE_MICRO_EXPECT_EQ(
      10, tflite::testing::ZeroPointFromMinMax<int8_t>(-69, 58.5));
  TF_LITE_MICRO_EXPECT_EQ(
      -10, tflite::testing::ZeroPointFromMinMax<int8_t>(-59, 68.5));
  TF_LITE_MICRO_EXPECT_EQ(
      0, tflite::testing::ZeroPointFromMinMax<uint8_t>(0, 255));
  TF_LITE_MICRO_EXPECT_EQ(
      64, tflite::testing::ZeroPointFromMinMax<uint8_t>(-32, 95.5));
}

TF_LITE_MICRO_TEST(ZeroPointRoundingTest) {
  TF_LITE_MICRO_EXPECT_EQ(
      -1, tflite::testing::ZeroPointFromMinMax<int8_t>(-126.51, 128.49));
  TF_LITE_MICRO_EXPECT_EQ(
      -1, tflite::testing::ZeroPointFromMinMax<int8_t>(-127.49, 127.51));
  TF_LITE_MICRO_EXPECT_EQ(
      0, tflite::testing::ZeroPointFromMinMax<int8_t>(-127.51, 127.49));
  TF_LITE_MICRO_EXPECT_EQ(
      0, tflite::testing::ZeroPointFromMinMax<int8_t>(-128.49, 126.51));
  TF_LITE_MICRO_EXPECT_EQ(
      1, tflite::testing::ZeroPointFromMinMax<int8_t>(-128.51, 126.49));
  TF_LITE_MICRO_EXPECT_EQ(
      1, tflite::testing::ZeroPointFromMinMax<int8_t>(-129.49, 125.51));
}

TF_LITE_MICRO_TEST(ScaleTest) {
  int min_int = std::numeric_limits<int32_t>::min();
  int max_int = std::numeric_limits<int32_t>::max();
  TF_LITE_MICRO_EXPECT_EQ(
      0.5, tflite::testing::ScaleFromMinMax<int32_t>(-0.5, max_int));
  TF_LITE_MICRO_EXPECT_EQ(
      1.0, tflite::testing::ScaleFromMinMax<int32_t>(min_int, max_int));
  TF_LITE_MICRO_EXPECT_EQ(0.25, tflite::testing::ScaleFromMinMax<int32_t>(
                                    min_int / 4, max_int / 4));
  TF_LITE_MICRO_EXPECT_EQ(0.5,
                          tflite::testing::ScaleFromMinMax<int8_t>(-64, 63.5));
  TF_LITE_MICRO_EXPECT_EQ(0.25,
                          tflite::testing::ScaleFromMinMax<int8_t>(0, 63.75));
  TF_LITE_MICRO_EXPECT_EQ(0.5,
                          tflite::testing::ScaleFromMinMax<uint8_t>(0, 127.5));
  TF_LITE_MICRO_EXPECT_EQ(
      0.25, tflite::testing::ScaleFromMinMax<uint8_t>(63.75, 127.5));
}

TF_LITE_MICRO_TEST(MinMaxTest) {
  TF_LITE_MICRO_EXPECT_EQ(
      -128, tflite::testing::MinFromZeroPointScale<int8_t>(0, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(
      127, tflite::testing::MaxFromZeroPointScale<int8_t>(0, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(
      -64, tflite::testing::MinFromZeroPointScale<int8_t>(0, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(
      63.5, tflite::testing::MaxFromZeroPointScale<int8_t>(0, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(
      -65, tflite::testing::MinFromZeroPointScale<int8_t>(2, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(
      62.5, tflite::testing::MaxFromZeroPointScale<int8_t>(2, 0.5));
}

TF_LITE_MICRO_TEST(ZeroPointScaleMinMaxSanityTest) {
  float min = -150.0f;
  float max = 105.0f;
  float scale = tflite::testing::ScaleFromMinMax<int8_t>(min, max);
  int zero_point = tflite::testing::ZeroPointFromMinMax<int8_t>(min, max);
  float min_test =
      tflite::testing::MinFromZeroPointScale<int8_t>(zero_point, scale);
  float max_test =
      tflite::testing::MaxFromZeroPointScale<int8_t>(zero_point, scale);
  TF_LITE_MICRO_EXPECT_EQ(min, min_test);
  TF_LITE_MICRO_EXPECT_EQ(max, max_test);
}

TF_LITE_MICRO_TESTS_END
