/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/utils.h"

#include <sys/types.h>

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/c/common.h"

namespace tflite::tools {
namespace {
using ::testing::FloatEq;

// Helper function to test TfLiteTensorToFloat32Array.
template <typename T>
void TestTfLiteTensorToFloat32Array(TfLiteType type) {
  T data[] = {1, 2, 3, 4};
  TfLiteTensor tensor;
  tensor.data.data = data;
  tensor.type = type;
  // Create an int array with 1 dimension and the array size is 4.
  tensor.dims = TfLiteIntArrayCreate(1);
  tensor.dims->data[0] = 4;
  std::vector<float> result(4, 0.0);
  const auto status =
      utils::TfLiteTensorToFloat32Array(tensor, absl::MakeSpan(result));
  TfLiteIntArrayFree(tensor.dims);
  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_EQ(result.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(result[i], FloatEq(static_cast<float>(data[i])));
  }
}

// Helper function to test TfLiteTensorToFloat32Array.
template <typename T>
void TestTfLiteTensorToInt64Array(TfLiteType type) {
  T data[] = {1, 2, 3, 4};
  TfLiteTensor tensor;
  tensor.data.data = data;
  tensor.type = type;
  // Create an int array with 1 dimension and the array size is 4.
  tensor.dims = TfLiteIntArrayCreate(1);
  tensor.dims->data[0] = 4;
  std::vector<int64_t> result(4, 0);
  const auto status =
      utils::TfLiteTensorToInt64Array(tensor, absl::MakeSpan(result));
  TfLiteIntArrayFree(tensor.dims);
  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_EQ(result.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(result[i], static_cast<int64_t>(data[i]));
  }
}

template <typename T>
void TestFloat32ArrayToInputTensorData(TfLiteType type) {
  const std::vector<float> kFloatList = {1.0, 2.0, 3.0, 4.0};
  utils::InputTensorData tensor;
  const auto status = utils::Float32ArrayToInputTensorData(
      kFloatList, 4 * sizeof(T), type, tensor);
  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_EQ(tensor.bytes, 4 * sizeof(T));
  T* data = reinterpret_cast<T*>(tensor.data.get());
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(static_cast<float>(data[i]), FloatEq(kFloatList[i]));
  }
}

template <typename T>
void TestInt64ArrayToInputTensorData(TfLiteType type) {
  const std::vector<int64_t> kInt64List = {1, 2, 3, 4};
  utils::InputTensorData tensor;
  const auto status = utils::Int64ArrayToInputTensorData(
      kInt64List, 4 * sizeof(T), type, tensor);
  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_EQ(tensor.bytes, 4 * sizeof(T));
  T* data = reinterpret_cast<T*>(tensor.data.get());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(static_cast<int64_t>(data[i]), kInt64List[i]);
  }
}

// Tests TfLiteTensorToFloat32Array for supported TfLiteTypes.
TEST(Utils, TfLiteTensorToFloat32Array) {
  TestTfLiteTensorToFloat32Array<float>(kTfLiteFloat32);
  TestTfLiteTensorToFloat32Array<double>(kTfLiteFloat64);
}

TEST(Utils, TfLiteTensorToInt64Array) {
  TestTfLiteTensorToInt64Array<int8_t>(kTfLiteInt8);
  TestTfLiteTensorToInt64Array<uint8_t>(kTfLiteUInt8);
  TestTfLiteTensorToInt64Array<int16_t>(kTfLiteInt16);
  TestTfLiteTensorToInt64Array<uint16_t>(kTfLiteUInt16);
  TestTfLiteTensorToInt64Array<int>(kTfLiteInt32);
  TestTfLiteTensorToInt64Array<uint32_t>(kTfLiteUInt32);
  TestTfLiteTensorToInt64Array<int64_t>(kTfLiteInt64);
  TestTfLiteTensorToInt64Array<uint64_t>(kTfLiteUInt64);
}

// Tests ArrayToInputTensorData for converting float array to InputTensorData.
TEST(Utils, Float32ArrayToInputTensorData) {
  TestFloat32ArrayToInputTensorData<float>(kTfLiteFloat32);
  TestFloat32ArrayToInputTensorData<double>(kTfLiteFloat64);
  TestFloat32ArrayToInputTensorData<int8_t>(kTfLiteInt8);
  TestFloat32ArrayToInputTensorData<uint8_t>(kTfLiteUInt8);
  TestFloat32ArrayToInputTensorData<int16_t>(kTfLiteInt16);
  TestFloat32ArrayToInputTensorData<uint16_t>(kTfLiteUInt16);
  TestFloat32ArrayToInputTensorData<int>(kTfLiteInt32);
  TestFloat32ArrayToInputTensorData<uint32_t>(kTfLiteUInt32);
  TestFloat32ArrayToInputTensorData<int64_t>(kTfLiteInt64);
  TestFloat32ArrayToInputTensorData<uint64_t>(kTfLiteUInt64);
}

// Tests ArrayToInputTensorData for converting int64 array to InputTensorData.
TEST(Utils, Int64ArrayToInputTensorData) {
  TestInt64ArrayToInputTensorData<float>(kTfLiteFloat32);
  TestInt64ArrayToInputTensorData<double>(kTfLiteFloat64);
  TestInt64ArrayToInputTensorData<int8_t>(kTfLiteInt8);
  TestInt64ArrayToInputTensorData<uint8_t>(kTfLiteUInt8);
  TestInt64ArrayToInputTensorData<int16_t>(kTfLiteInt16);
  TestInt64ArrayToInputTensorData<uint16_t>(kTfLiteUInt16);
  TestInt64ArrayToInputTensorData<int>(kTfLiteInt32);
  TestInt64ArrayToInputTensorData<uint32_t>(kTfLiteUInt32);
  TestInt64ArrayToInputTensorData<int64_t>(kTfLiteInt64);
  TestInt64ArrayToInputTensorData<uint64_t>(kTfLiteUInt64);
}
}  // namespace
}  // namespace tflite::tools
