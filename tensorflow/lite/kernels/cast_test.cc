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
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/cast_test_common.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

TEST(CastOpModel, CastInt4ToFloat) {
  CastOpModel m({TensorType_INT4, {2, 3}}, {TensorType_FLOAT32, {2, 3}});
  m.Set4BitInput({1, 2, 3, 4, 5, 6});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
}

TEST(CastOpModel, CastInt4ToFloatLarge) {
  int num_elements = 40;
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int8_t> i8dist(-8, 7);
  auto i8rng = [&] { return i8dist(rng); };
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
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
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
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
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
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({100.f, 200.f, 300.f, 400.f, 500.f, 600.f}));
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
              ElementsAreArray({1.f, 1.0f, 0.f, 1.0f, 0.0f, 1.0f}));
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
              ElementsAreArray({123.f, 0.f, 1.f, 2.f, 3.f, 4.f}));
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
              ElementsAreArray({123.f, 0.f, 1.f, 2.f, 3.f, 4.f}));
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
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
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
