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

#include "tensorflow/lite/experimental/micro/memory_helpers.h"

#include "tensorflow/lite/experimental/micro/test_helpers.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestAlignPointerUp) {
  uint8_t* input0 = reinterpret_cast<uint8_t*>(0);

  uint8_t* input0_aligned1 = tflite::AlignPointerUp(input0, 1);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned1);

  uint8_t* input0_aligned2 = tflite::AlignPointerUp(input0, 2);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned2);

  uint8_t* input0_aligned3 = tflite::AlignPointerUp(input0, 3);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned3);

  uint8_t* input0_aligned16 = tflite::AlignPointerUp(input0, 16);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned16);

  uint8_t* input23 = reinterpret_cast<uint8_t*>(23);

  uint8_t* input23_aligned1 = tflite::AlignPointerUp(input23, 1);
  TF_LITE_MICRO_EXPECT_EQ(input23, input23_aligned1);

  uint8_t* input23_aligned2 = tflite::AlignPointerUp(input23, 2);
  uint8_t* expected23_aligned2 = reinterpret_cast<uint8_t*>(24);
  TF_LITE_MICRO_EXPECT_EQ(expected23_aligned2, input23_aligned2);

  uint8_t* input23_aligned3 = tflite::AlignPointerUp(input23, 3);
  uint8_t* expected23_aligned3 = reinterpret_cast<uint8_t*>(24);
  TF_LITE_MICRO_EXPECT_EQ(expected23_aligned3, input23_aligned3);

  uint8_t* input23_aligned16 = tflite::AlignPointerUp(input23, 16);
  uint8_t* expected23_aligned16 = reinterpret_cast<uint8_t*>(32);
  TF_LITE_MICRO_EXPECT_EQ(expected23_aligned16, input23_aligned16);
}

TF_LITE_MICRO_TEST(TestAlignPointerDown) {
  uint8_t* input0 = reinterpret_cast<uint8_t*>(0);

  uint8_t* input0_aligned1 = tflite::AlignPointerDown(input0, 1);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned1);

  uint8_t* input0_aligned2 = tflite::AlignPointerDown(input0, 2);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned2);

  uint8_t* input0_aligned3 = tflite::AlignPointerDown(input0, 3);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned3);

  uint8_t* input0_aligned16 = tflite::AlignPointerDown(input0, 16);
  TF_LITE_MICRO_EXPECT_EQ(input0, input0_aligned16);

  uint8_t* input23 = reinterpret_cast<uint8_t*>(23);

  uint8_t* input23_aligned1 = tflite::AlignPointerDown(input23, 1);
  TF_LITE_MICRO_EXPECT_EQ(input23, input23_aligned1);

  uint8_t* input23_aligned2 = tflite::AlignPointerDown(input23, 2);
  uint8_t* expected23_aligned2 = reinterpret_cast<uint8_t*>(22);
  TF_LITE_MICRO_EXPECT_EQ(expected23_aligned2, input23_aligned2);

  uint8_t* input23_aligned3 = tflite::AlignPointerDown(input23, 3);
  uint8_t* expected23_aligned3 = reinterpret_cast<uint8_t*>(21);
  TF_LITE_MICRO_EXPECT_EQ(expected23_aligned3, input23_aligned3);

  uint8_t* input23_aligned16 = tflite::AlignPointerDown(input23, 16);
  uint8_t* expected23_aligned16 = reinterpret_cast<uint8_t*>(16);
  TF_LITE_MICRO_EXPECT_EQ(expected23_aligned16, input23_aligned16);
}

TF_LITE_MICRO_TEST(TestAlignSizeUp) {
  TF_LITE_MICRO_EXPECT_EQ(1, tflite::AlignSizeUp(1, 1));
  TF_LITE_MICRO_EXPECT_EQ(2, tflite::AlignSizeUp(1, 2));
  TF_LITE_MICRO_EXPECT_EQ(3, tflite::AlignSizeUp(1, 3));
  TF_LITE_MICRO_EXPECT_EQ(16, tflite::AlignSizeUp(1, 16));

  TF_LITE_MICRO_EXPECT_EQ(23, tflite::AlignSizeUp(23, 1));
  TF_LITE_MICRO_EXPECT_EQ(24, tflite::AlignSizeUp(23, 2));
  TF_LITE_MICRO_EXPECT_EQ(24, tflite::AlignSizeUp(23, 3));
  TF_LITE_MICRO_EXPECT_EQ(32, tflite::AlignSizeUp(23, 16));
}

TF_LITE_MICRO_TEST(TestTypeSizeOf) {
  size_t size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteFloat32, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(float), size);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteInt16, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int16_t), size);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteInt32, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int32_t), size);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteUInt8, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(uint8_t), size);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteInt8, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int8_t), size);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteInt64, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(int64_t), size);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteBool, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(bool), size);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::TfLiteTypeSizeOf(kTfLiteComplex64, &size, micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(sizeof(float) * 2, size);

  TF_LITE_MICRO_EXPECT_NE(
      kTfLiteOk, tflite::TfLiteTypeSizeOf(static_cast<TfLiteType>(-1), &size,
                                          micro_test::reporter));
}

TF_LITE_MICRO_TEST(TestBytesRequiredForTensor) {
  const tflite::Tensor* tensor100 =
      tflite::testing::Create1dFlatbufferTensor(100);
  size_t bytes;
  size_t type_size;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::BytesRequiredForTensor(*tensor100, &bytes, &type_size,
                                                micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(400, bytes);
  TF_LITE_MICRO_EXPECT_EQ(4, type_size);

  const tflite::Tensor* tensor200 =
      tflite::testing::Create1dFlatbufferTensor(200);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::BytesRequiredForTensor(*tensor200, &bytes, &type_size,
                                                micro_test::reporter));
  TF_LITE_MICRO_EXPECT_EQ(800, bytes);
  TF_LITE_MICRO_EXPECT_EQ(4, type_size);
}

TF_LITE_MICRO_TESTS_END
