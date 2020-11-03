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

#include "tensorflow/lite/micro/micro_utils.h"

#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatToAsymmetricQuantizedUInt8Test) {
  using tflite::FloatToQuantizedType;
  // [0, 127.5] -> zero_point=0, scale=0.5
  TF_LITE_MICRO_EXPECT_EQ(0, FloatToQuantizedType<uint8_t>(0, 0.5, 0));
  TF_LITE_MICRO_EXPECT_EQ(254, FloatToQuantizedType<uint8_t>(127, 0.5, 0));
  TF_LITE_MICRO_EXPECT_EQ(255, FloatToQuantizedType<uint8_t>(127.5, 0.5, 0));
  // [-10, 245] -> zero_point=10, scale=1.0
  TF_LITE_MICRO_EXPECT_EQ(0, FloatToQuantizedType<uint8_t>(-10, 1.0, 10));
  TF_LITE_MICRO_EXPECT_EQ(1, FloatToQuantizedType<uint8_t>(-9, 1.0, 10));
  TF_LITE_MICRO_EXPECT_EQ(128, FloatToQuantizedType<uint8_t>(118, 1.0, 10));
  TF_LITE_MICRO_EXPECT_EQ(253, FloatToQuantizedType<uint8_t>(243, 1.0, 10));
  TF_LITE_MICRO_EXPECT_EQ(254, FloatToQuantizedType<uint8_t>(244, 1.0, 10));
  TF_LITE_MICRO_EXPECT_EQ(255, FloatToQuantizedType<uint8_t>(245, 1.0, 10));
}

TF_LITE_MICRO_TEST(FloatToAsymmetricQuantizedInt8Test) {
  using tflite::FloatToQuantizedType;
  // [-64, 63.5] -> zero_point=0, scale=0.5
  TF_LITE_MICRO_EXPECT_EQ(2, FloatToQuantizedType<int8_t>(1, 0.5, 0));
  TF_LITE_MICRO_EXPECT_EQ(4, FloatToQuantizedType<int8_t>(2, 0.5, 0));
  TF_LITE_MICRO_EXPECT_EQ(6, FloatToQuantizedType<int8_t>(3, 0.5, 0));
  TF_LITE_MICRO_EXPECT_EQ(-10, FloatToQuantizedType<int8_t>(-5, 0.5, 0));
  TF_LITE_MICRO_EXPECT_EQ(-128, FloatToQuantizedType<int8_t>(-64, 0.5, 0));
  TF_LITE_MICRO_EXPECT_EQ(127, FloatToQuantizedType<int8_t>(63.5, 0.5, 0));
  // [-127, 128] -> zero_point=-1, scale=1.0
  TF_LITE_MICRO_EXPECT_EQ(0, FloatToQuantizedType<int8_t>(1, 1.0, -1));
  TF_LITE_MICRO_EXPECT_EQ(-1, FloatToQuantizedType<int8_t>(0, 1.0, -1));
  TF_LITE_MICRO_EXPECT_EQ(126, FloatToQuantizedType<int8_t>(127, 1.0, -1));
  TF_LITE_MICRO_EXPECT_EQ(127, FloatToQuantizedType<int8_t>(128, 1.0, -1));
  TF_LITE_MICRO_EXPECT_EQ(-127, FloatToQuantizedType<int8_t>(-126, 1.0, -1));
  TF_LITE_MICRO_EXPECT_EQ(-128, FloatToQuantizedType<int8_t>(-127, 1.0, -1));
}

TF_LITE_MICRO_TEST(FloatToSymmetricQuantizedInt8Test) {
  using tflite::FloatToSymmetricQuantizedType;
  // [-64, 63.5] -> zero_point=0, scale=0.5
  TF_LITE_MICRO_EXPECT_EQ(2, FloatToSymmetricQuantizedType<int8_t>(1, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(4, FloatToSymmetricQuantizedType<int8_t>(2, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(6, FloatToSymmetricQuantizedType<int8_t>(3, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(-10, FloatToSymmetricQuantizedType<int8_t>(-5, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(-127,
                          FloatToSymmetricQuantizedType<int8_t>(-64, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(127,
                          FloatToSymmetricQuantizedType<int8_t>(63.5, 0.5));
  // [-127, 128] -> zero_point=-1, scale=1.0
  TF_LITE_MICRO_EXPECT_EQ(1, FloatToSymmetricQuantizedType<int8_t>(1, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(0, FloatToSymmetricQuantizedType<int8_t>(0, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(127, FloatToSymmetricQuantizedType<int8_t>(127, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(127, FloatToSymmetricQuantizedType<int8_t>(128, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(-126,
                          FloatToSymmetricQuantizedType<int8_t>(-126, 1.0));
  TF_LITE_MICRO_EXPECT_EQ(-127,
                          FloatToSymmetricQuantizedType<int8_t>(-127, 1.0));
}

TF_LITE_MICRO_TEST(FloatToAsymmetricQuantizedInt32Test) {
  using tflite::FloatToSymmetricQuantizedType;
  TF_LITE_MICRO_EXPECT_EQ(0, FloatToSymmetricQuantizedType<int32_t>(0, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(2, FloatToSymmetricQuantizedType<int32_t>(1, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(-2, FloatToSymmetricQuantizedType<int32_t>(-1, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(-100,
                          FloatToSymmetricQuantizedType<int32_t>(-50, 0.5));
  TF_LITE_MICRO_EXPECT_EQ(100, FloatToSymmetricQuantizedType<int32_t>(50, 0.5));
}

TF_LITE_MICRO_TEST(AsymmetricQuantizeInt8) {
  float values[] = {-10.3, -3.1, -2.1, -1.9, -0.9, 0.1, 0.9, 1.85, 2.9, 4.1};
  int8_t goldens[] = {-20, -5, -3, -3, -1, 1, 3, 5, 7, 9};
  constexpr int length = sizeof(values) / sizeof(float);
  int8_t quantized[length];
  tflite::Quantize(values, quantized, length, 0.5, 1);
  for (int i = 0; i < length; i++) {
    TF_LITE_MICRO_EXPECT_EQ(quantized[i], goldens[i]);
  }
}

TF_LITE_MICRO_TEST(AsymmetricQuantizeUInt8) {
  float values[] = {-10.3, -3.1, -2.1, -1.9, -0.9, 0.1, 0.9, 1.85, 2.9, 4.1};
  uint8_t goldens[] = {106, 121, 123, 123, 125, 127, 129, 131, 133, 135};
  constexpr int length = sizeof(values) / sizeof(float);
  uint8_t quantized[length];
  tflite::Quantize(values, quantized, length, 0.5, 127);
  for (int i = 0; i < length; i++) {
    TF_LITE_MICRO_EXPECT_EQ(quantized[i], goldens[i]);
  }
}

TF_LITE_MICRO_TEST(SymmetricQuantizeInt32) {
  float values[] = {-10.3, -3.1, -2.1, -1.9, -0.9, 0.1, 0.9, 1.85, 2.9, 4.1};
  int32_t goldens[] = {-21, -6, -4, -4, -2, 0, 2, 4, 6, 8};
  constexpr int length = sizeof(values) / sizeof(float);
  int32_t quantized[length];
  tflite::SymmetricQuantize(values, quantized, length, 0.5);
  for (int i = 0; i < length; i++) {
    TF_LITE_MICRO_EXPECT_EQ(quantized[i], goldens[i]);
  }
}

TF_LITE_MICRO_TESTS_END
