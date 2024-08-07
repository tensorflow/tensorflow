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
#include "tensorflow/lite/c/common.h"

namespace tflite::tools {
namespace {
using ::testing::FloatEq;

// Helper function to test TfLiteTensorToFloatArray.
template <typename T>
void TestTfLiteTensorToFloatArray(TfLiteType type) {
  T data[] = {1, 2, 3, 4};
  TfLiteTensor tensor;
  tensor.data.data = data;
  tensor.type = type;
  // Create an int array with 1 dimension and the array size is 4.
  tensor.dims = TfLiteIntArrayCreate(1);
  tensor.dims->data[0] = 4;
  std::vector<float> result;
  const auto status = utils::TfLiteTensorToFloatArray(tensor, result);
  TfLiteIntArrayFree(tensor.dims);
  ASSERT_EQ(status, kTfLiteOk);
  ASSERT_EQ(result.size(), 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_THAT(result[i], FloatEq(static_cast<float>(data[i])));
  }
}

// Tests TfLiteTensorToFloatArray for supported TfLiteTypes.
TEST(TfLiteTensorToFloatArray, SupportedTfLiteTypes) {
  TestTfLiteTensorToFloatArray<float>(kTfLiteFloat32);
  TestTfLiteTensorToFloatArray<double>(kTfLiteFloat64);
  TestTfLiteTensorToFloatArray<int8_t>(kTfLiteInt8);
  TestTfLiteTensorToFloatArray<uint8_t>(kTfLiteUInt8);
  TestTfLiteTensorToFloatArray<int16_t>(kTfLiteInt16);
  TestTfLiteTensorToFloatArray<uint16_t>(kTfLiteUInt16);
  TestTfLiteTensorToFloatArray<int>(kTfLiteInt32);
  TestTfLiteTensorToFloatArray<uint32_t>(kTfLiteUInt32);
  TestTfLiteTensorToFloatArray<int64_t>(kTfLiteInt64);
  TestTfLiteTensorToFloatArray<uint64_t>(kTfLiteUInt64);
}

}  // namespace
}  // namespace tflite::tools
