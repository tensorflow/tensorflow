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
#include <stdint.h>

#include <algorithm>
#include <complex>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/cast_test_common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/types/half.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(CastOpModel, CastInt4ToFloat) {
  CastOpModel m({TensorType_INT4, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.Set4BitInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              Pointwise(FloatingPointEq(), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
}

TEST(CastOpModel, CastInt4ToFloatLarge) {
  int num_elements = 40;
  absl::BitGen bitgen;
  auto i8rng = [&] {
    return absl::Uniform<int8_t>(absl::IntervalClosed, bitgen, -8, 7);
  };
  std::vector<int8_t> input(num_elements);
  std::generate(input.begin(), input.end(), i8rng);
  CastOpModel m({TensorType_INT4, {num_elements}},
                {TensorType_FLOAT32, {num_elements}});
  m.Set4BitInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  for (int i = 0; i < input.size(); ++i) {
    EXPECT_EQ(m.ExtractVector<float>(m.output())[i], input[i]);
  }
}

TEST(CastOpModel, CastInt2ToFloat) {
  CastOpModel m({TensorType_INT2, {2, 4}}, {TensorType_FLOAT32, {2, 4}});
  m.Set2BitInput({1, 0, -1, -2, 1, 0, -1, -2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              Pointwise(FloatingPointEq(),
                        {1.f, 0.f, -1.f, -2.f, 1.f, 0.f, -1.f, -2.f}));
}

TEST(CastOpModel, CastInt2ToFloatLarge) {
  int num_elements = 40;
  absl::BitGen bitgen;
  auto i2rng = [&] {
    return absl::Uniform<int8_t>(absl::IntervalClosed, bitgen, -2, 1);
  };
  std::vector<int8_t> input(num_elements);
  std::generate(input.begin(), input.end(), i2rng);
  CastOpModel m({TensorType_INT2, {num_elements}},
                {TensorType_FLOAT32, {num_elements}});
  m.Set2BitInput(input);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  for (int i = 0; i < input.size(); ++i) {
    EXPECT_EQ(m.ExtractVector<float>(m.output())[i], input[i]);
  }
}

TEST(CastOpModel, CastFloatToInt4) {
  CastOpModel m({TensorType_FLOAT32, {2, 4}}, {TensorType_INT4, {2, 4}});
  m.PopulateTensor<float>(m.input(), {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, -8.f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  TfLiteTensor* output = m.GetOutputTensor(0);
  int num_elements = NumElements(output);
  std::vector<int8_t> unpacked_output(num_elements);
  tensor_utils::UnpackPackedIntToInt8(
      reinterpret_cast<int8_t*>(output->data.data), num_elements,
      /*bit_width=*/4, unpacked_output.data());
  EXPECT_THAT(unpacked_output, ElementsAreArray({1, 2, 3, 4, 5, 6, 7, -8}));
}

TEST(CastOpModel, CastFloatToInt4Clamp) {
  CastOpModel m({TensorType_FLOAT32, {1, 4}}, {TensorType_INT4, {1, 4}});
  m.PopulateTensor<float>(m.input(), {100.f, -100.f, 7.9f, -8.9f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  TfLiteTensor* output = m.GetOutputTensor(0);
  int num_elements = NumElements(output);
  std::vector<int8_t> unpacked_output(num_elements);
  tensor_utils::UnpackPackedIntToInt8(
      reinterpret_cast<int8_t*>(output->data.data), num_elements,
      /*bit_width=*/4, unpacked_output.data());
  EXPECT_THAT(unpacked_output, ElementsAreArray({7, -8, 7, -8}));
}

TEST(CastOpModel, CastFloatToInt2) {
  CastOpModel m({TensorType_FLOAT32, {2, 4}}, {TensorType_INT2, {2, 4}});
  m.PopulateTensor<float>(m.input(),
                          {1.f, 0.f, -1.f, -2.f, 1.f, 0.f, -1.f, -2.f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  TfLiteTensor* output = m.GetOutputTensor(0);
  int num_elements = NumElements(output);
  std::vector<int8_t> unpacked_output(num_elements);
  tensor_utils::UnpackPackedIntToInt8(
      reinterpret_cast<int8_t*>(output->data.data), num_elements,
      /*bit_width=*/2, unpacked_output.data());
  EXPECT_THAT(unpacked_output, ElementsAreArray({1, 0, -1, -2, 1, 0, -1, -2}));
}

TEST(CastOpModel, CastFloatToInt2Clamp) {
  CastOpModel m({TensorType_FLOAT32, {1, 4}}, {TensorType_INT2, {1, 4}});
  m.PopulateTensor<float>(m.input(), {100.f, -100.f, 1.9f, -2.9f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  TfLiteTensor* output = m.GetOutputTensor(0);
  int num_elements = NumElements(output);
  std::vector<int8_t> unpacked_output(num_elements);
  tensor_utils::UnpackPackedIntToInt8(
      reinterpret_cast<int8_t*>(output->data.data), num_elements,
      /*bit_width=*/2, unpacked_output.data());
  EXPECT_THAT(unpacked_output, ElementsAreArray({1, -2, 1, -2}));
}

TEST(CastOpModel, CastFloatToUint8Infinity) {
  CastOpModel m({TensorType_FLOAT32, {2}}, {TensorType_UINT8, {2}});
  m.PopulateTensor<float>(m.input(), {std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity()});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint8_t>(m.output()),
              ElementsAreArray({std::numeric_limits<uint8_t>::max(),
                                std::numeric_limits<uint8_t>::min()}));
}

TEST(CastOpModel, CastFloatToInt16Infinity) {
  CastOpModel m({TensorType_FLOAT32, {2}}, {TensorType_INT16, {2}});
  m.PopulateTensor<float>(m.input(), {std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity()});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int16_t>(m.output()),
              ElementsAreArray({std::numeric_limits<int16_t>::max(),
                                std::numeric_limits<int16_t>::min()}));
}

TEST(CastOpModel, CastFloatToInt32Infinity) {
  CastOpModel m({TensorType_FLOAT32, {2}}, {TensorType_INT32, {2}});
  m.PopulateTensor<float>(m.input(), {std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity()});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::min()}));
}

TEST(CastOpModel, CastInt16ToFloat) {
  CastOpModel m({TensorType_INT16, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int16_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<float>(m.output()),
      Pointwise(FloatingPointEq(), {100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
}

TEST(CastOpModel, CastInt16ToInt32) {
  CastOpModel m({TensorType_INT16, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.PopulateTensor<int16_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 200, 300, 400, 500, 600}));
}

TEST(CastOpModel, CastInt32ToFloat) {
  CastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<float>(m.output()),
      Pointwise(FloatingPointEq(), {100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
}

TEST(CastOpModel, CastFloatToInt32) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_INT32, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastFloatToInt16) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_INT16, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int16_t>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastInt64ToFloat) {
  CastOpModel m({TensorType_INT64, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<int64_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<float>(m.output()),
      Pointwise(FloatingPointEq(), {100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
}

TEST(CastOpModel, CastFloatToInt64) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_INT64, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int64_t>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastFloatToBool) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_BOOL, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, -1.0f, 0.f, 0.4f, 0.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<bool>(m.output()),
              ElementsAreArray({true, true, false, true, true, true}));
}

TEST(CastOpModel, CastBoolToFloat) {
  CastOpModel m({TensorType_BOOL, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<bool>(m.input(), {true, true, false, true, false, true});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              Pointwise(FloatingPointEq(), {1.f, 1.0f, 0.f, 1.0f, 0.0f, 1.0f}));
}

TEST(CastOpModel, CastFloatToUInt8) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_UINT8, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 1.0f, 0.f, 0.4f, 1.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint8_t>(m.output()),
              ElementsAreArray({100, 1, 0, 0, 1, 1}));
}

TEST(CastOpModel, CastUInt8ToFloat) {
  CastOpModel m({TensorType_UINT8, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<uint8_t>(m.input(), {123, 0, 1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              Pointwise(FloatingPointEq(), {123.f, 0.f, 1.f, 2.f, 3.f, 4.f}));
}

TEST(CastOpModel, CastFloatToUInt16) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_UINT16, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 1.0f, 0.f, 0.4f, 1.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint16_t>(m.output()),
              ElementsAreArray({100, 1, 0, 0, 1, 1}));
}

TEST(CastOpModel, CastUInt16ToFloat) {
  CastOpModel m({TensorType_UINT16, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<uint16_t>(m.input(), {123, 0, 1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              Pointwise(FloatingPointEq(), {123.f, 0.f, 1.f, 2.f, 3.f, 4.f}));
}

TEST(CastOpModel, CastInt32ToUInt8) {
  CastOpModel m({TensorType_INT32, {3, 2}}, {TensorType_UINT8, {3, 2}});
  m.PopulateTensor<int32_t>(m.input(), {100, 1, 200, 2, 255, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint8_t>(m.output()),
              ElementsAreArray({100, 1, 200, 2, 255, 3}));
}

TEST(CastOpModel, CastUInt8ToInt32) {
  CastOpModel m({TensorType_UINT8, {3, 2}}, {TensorType_INT32, {3, 2}});
  m.PopulateTensor<uint8_t>(m.input(), {100, 1, 200, 2, 255, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 1, 200, 2, 255, 3}));
}

TEST(CastOpModel, CastComplex64ToFloat) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<float>(m.output()),
      Pointwise(FloatingPointEq(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(CastOpModel, CastFloatToComplex64) {
  CastOpModel m({TensorType_FLOAT32, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<float>(m.input(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 0.0f), std::complex<float>(2.0f, 0.0f),
           std::complex<float>(3.0f, 0.0f), std::complex<float>(4.0f, 0.0f),
           std::complex<float>(5.0f, 0.0f), std::complex<float>(6.0f, 0.0f)}));
}

TEST(CastOpModel, CastComplex64ToInt) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int>(m.output()),
              ElementsAreArray({1, 2, 3, 4, 5, 6}));
}

TEST(CastOpModel, CastIntToComplex64) {
  CastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<int>(m.input(), {1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 0.0f), std::complex<float>(2.0f, 0.0f),
           std::complex<float>(3.0f, 0.0f), std::complex<float>(4.0f, 0.0f),
           std::complex<float>(5.0f, 0.0f), std::complex<float>(6.0f, 0.0f)}));
}

TEST(CastOpModel, CastComplex64ToComplex64) {
  CastOpModel m({TensorType_COMPLEX64, {2, 3}}, {TensorType_COMPLEX64, {2, 3}});
  m.PopulateTensor<std::complex<float>>(
      m.input(),
      {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
       std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
       std::complex<float>(5.0f, 15.0f), std::complex<float>(6.0f, 16.0f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<std::complex<float>>(m.output()),
      ElementsAreArray(
          {std::complex<float>(1.0f, 11.0f), std::complex<float>(2.0f, 12.0f),
           std::complex<float>(3.0f, 13.0f), std::complex<float>(4.0f, 14.0f),
           std::complex<float>(5.0f, 15.0f),
           std::complex<float>(6.0f, 16.0f)}));
}

TEST(CastOpModel, CastUInt32ToInt32) {
  CastOpModel m({TensorType_UINT32, {2, 3}}, {TensorType_INT32, {2, 3}});
  m.PopulateTensor<uint32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 200, 300, 400, 500, 600}));
}

TEST(CastOpModel, CastInt32ToUInt32) {
  CastOpModel m({TensorType_INT32, {2, 3}}, {TensorType_UINT32, {2, 3}});
  m.PopulateTensor<int32_t>(m.input(), {100, 200, 300, 400, 500, 600});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint32_t>(m.output()),
              ElementsAreArray({100, 200, 300, 400, 500, 600}));
}

TEST(CastOpModel, CastUInt8ToInt8) {
  CastOpModel m({TensorType_UINT8, {2, 3}}, {TensorType_INT8, {2, 3}});
  m.PopulateTensor<uint8_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int8_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

TEST(CastOpModel, CastInt8ToUInt8) {
  CastOpModel m({TensorType_INT8, {2, 3}}, {TensorType_UINT8, {2, 3}});
  m.PopulateTensor<int8_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint8_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

TEST(CastOpModel, CastUInt16ToInt16) {
  CastOpModel m({TensorType_UINT16, {2, 3}}, {TensorType_INT16, {2, 3}});
  m.PopulateTensor<uint16_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int16_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

TEST(CastOpModel, CastInt16ToUInt16) {
  CastOpModel m({TensorType_INT16, {2, 3}}, {TensorType_UINT16, {2, 3}});
  m.PopulateTensor<int16_t>(m.input(), {10, 20, 30, 40, 50, 60});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<uint16_t>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
}

TEST(CastOpModel, CastFloatToFloat16) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_FLOAT16, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 1.0f, 0.f, 0.4f, 1.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<half>(m.output()),
      ElementsAreArray({static_cast<half>(100.f), static_cast<half>(1.0f),
                        static_cast<half>(0.f), static_cast<half>(0.4f),
                        static_cast<half>(1.999f), static_cast<half>(1.1f)}));
}

TEST(CastOpModel, CastFloatToBFloat16) {
  CastOpModel m({TensorType_FLOAT32, {3, 2}}, {TensorType_BFLOAT16, {3, 2}});
  m.PopulateTensor<float>(m.input(), {100.f, 1.0f, 0.f, 0.4f, 1.999f, 1.1f});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<Eigen::bfloat16>(m.output()),
              ElementsAreArray({static_cast<Eigen::bfloat16>(100.f),
                                static_cast<Eigen::bfloat16>(1.0f),
                                static_cast<Eigen::bfloat16>(0.f),
                                static_cast<Eigen::bfloat16>(0.4f),
                                static_cast<Eigen::bfloat16>(1.999f),
                                static_cast<Eigen::bfloat16>(1.1f)}));
}

TEST(CastOpModel, CastFloat16ToFloat) {
  CastOpModel m({TensorType_FLOAT16, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<half>(m.input(),
                         {static_cast<half>(100.f), static_cast<half>(1.0f),
                          static_cast<half>(0.f), static_cast<half>(0.4f),
                          static_cast<half>(1.999f), static_cast<half>(1.1f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear(
                  {100.f, 1.0f, 0.f, 0.399902344f, 1.99902344f, 1.09960938f},
                  /*max_abs_err=*/0.05f)));
}

TEST(CastOpModel, CastBFloat16ToFloat) {
  CastOpModel m({TensorType_BFLOAT16, {3, 2}}, {TensorType_FLOAT32, {3, 2}});
  m.PopulateTensor<Eigen::bfloat16>(
      m.input(),
      {static_cast<Eigen::bfloat16>(100.f), static_cast<Eigen::bfloat16>(1.0f),
       static_cast<Eigen::bfloat16>(0.f), static_cast<Eigen::bfloat16>(0.4f),
       static_cast<Eigen::bfloat16>(1.999f),
       static_cast<Eigen::bfloat16>(1.1)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray(ArrayFloatNear(
                  {100.f, 1.0f, 0.f, 0.400390625f, 2.f, 1.1015625f},
                  /*max_abs_err=*/0.05f)));
}

TEST(CastOpModel, CastFloat16ToInt32) {
  CastOpModel m({TensorType_FLOAT16, {1, 6}}, {TensorType_INT32, {1, 6}});
  m.PopulateTensor<half>(m.input(),
                         {static_cast<half>(100.f), static_cast<half>(20.f),
                          static_cast<half>(3.f), static_cast<half>(0.4f),
                          static_cast<half>(0.999f), static_cast<half>(1.1f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({100, 20, 3, 0, 0, 1}));
}

TEST(CastOpModel, CastInt32ToFloat16) {
  CastOpModel m({TensorType_INT32, {1, 6}}, {TensorType_FLOAT16, {1, 6}});
  m.PopulateTensor<int32_t>(m.input(), {100, 20, 3, 0, 1, -1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(
      m.ExtractVector<half>(m.output()),
      ElementsAreArray({static_cast<half>(100.f), static_cast<half>(20.f),
                        static_cast<half>(3.f), static_cast<half>(0.f),
                        static_cast<half>(1.f), static_cast<half>(-1.f)}));
}

TEST(CastOpModel, CastFloat16ToBFloat16) {
  CastOpModel m({TensorType_FLOAT16, {1, 6}}, {TensorType_BFLOAT16, {1, 6}});
  m.PopulateTensor<half>(m.input(),
                         {static_cast<half>(100.f), static_cast<half>(20.f),
                          static_cast<half>(3.f), static_cast<half>(0.4f),
                          static_cast<half>(0.999f), static_cast<half>(1.1f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<Eigen::bfloat16>(m.output()),
              ElementsAreArray({static_cast<Eigen::bfloat16>(100.f),
                                static_cast<Eigen::bfloat16>(20.f),
                                static_cast<Eigen::bfloat16>(3.f),
                                static_cast<Eigen::bfloat16>(0.4f),
                                static_cast<Eigen::bfloat16>(0.999f),
                                static_cast<Eigen::bfloat16>(1.1f)}));
}

TEST(CastOpModel, CastBFloat16ToFloat16) {
  CastOpModel m({TensorType_BFLOAT16, {1, 6}}, {TensorType_FLOAT16, {1, 6}});
  m.PopulateTensor<Eigen::bfloat16>(
      m.input(),
      {static_cast<Eigen::bfloat16>(100.f), static_cast<Eigen::bfloat16>(20.f),
       static_cast<Eigen::bfloat16>(3.f), static_cast<Eigen::bfloat16>(0.4f),
       static_cast<Eigen::bfloat16>(0.999f),
       static_cast<Eigen::bfloat16>(1.1f)});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<half>(m.output()),
              ElementsAreArray(ArrayFloatNear(
                  {static_cast<half>(100.f), static_cast<half>(20.f),
                   static_cast<half>(3.f), static_cast<half>(0.4f),
                   static_cast<half>(0.999f), static_cast<half>(1.1f)},
                  /*max_abs_err=*/0.05f)));
}

TEST(CastOpModel, CastConstInputCachingWorks) {
  // This tests the implementation of a performance optimization. If that
  // optimization is changed, this test will likely break/need to be updated.
  //
  // We are relying on the fact that casting a constant input can be cached and
  // that the output tensor does not need to be updated on every call.
  CastOpModel m({TensorType_INT8, {2, 3}},
                std::vector<int8_t>{10, 20, 30, 40, 50, 60},
                {TensorType_FLOAT32, {2, 3}});
  EXPECT_EQ(m.GetOutputTensor(0)->allocation_type, kTfLiteArenaRwPersistent);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({10, 20, 30, 40, 50, 60}));
  // We are cheating here. If the values of the output tensor are cached then if
  // we modify the cache and call the op again the output tensor values should
  // not change.
  float* output_data =
      reinterpret_cast<float*>(m.GetOutputTensor(0)->data.data);
  for (int i = 0; i < 6; ++i) {
    ++output_data[i];
  }
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({11, 21, 31, 41, 51, 61}));
}

}  // namespace
}  // namespace tflite
