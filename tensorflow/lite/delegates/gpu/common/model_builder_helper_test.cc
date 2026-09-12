/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/model_builder_helper.h"

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace gpu {
namespace {

using ::testing::ElementsAre;

TEST(ModelBuilderHelperTest, CreateVectorCopyDataDifferentSize) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = kTfLiteInt32;
  int32_t src_data[4] = {1, 2, 3, 4};
  tflite_tensor.data.i32 = src_data;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = sizeof(src_data) / sizeof(src_data[0]);
  tflite_tensor.bytes = sizeof(src_data);

  int16_t dst[4];
  ASSERT_OK(CreateVectorCopyData(tflite_tensor, dst));
  EXPECT_THAT(dst, ElementsAre(1, 2, 3, 4));

  TfLiteIntArrayFree(tflite_tensor.dims);
}

TEST(ModelBuilderHelperTest, CreateVectorCopyDataFloatBytesLargerThanShape) {
  // Callers size the destination using the tensor shape, so the float copy
  // must not write more elements than the shape describes even when the
  // source buffer reports a larger byte count.
  float src_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  TfLiteTensor tflite_tensor = {};
  tflite_tensor.type = kTfLiteFloat32;
  tflite_tensor.data.f = src_data;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 1;
  tflite_tensor.bytes = sizeof(src_data);

  constexpr float kSentinel = -1.0f;
  float dst[4] = {kSentinel, kSentinel, kSentinel, kSentinel};
  ASSERT_OK(CreateVectorCopyData(tflite_tensor, dst));
  EXPECT_EQ(dst[0], 1.0f);
  EXPECT_EQ(dst[1], kSentinel);
  EXPECT_EQ(dst[2], kSentinel);
  EXPECT_EQ(dst[3], kSentinel);

  TfLiteIntArrayFree(tflite_tensor.dims);
}

TEST(ModelBuilderHelperTest, CreateVectorCopyDataFloatBytesSmallerThanShape) {
  float src_data[1] = {1.0f};
  TfLiteTensor tflite_tensor = {};
  tflite_tensor.type = kTfLiteFloat32;
  tflite_tensor.data.f = src_data;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 4;
  tflite_tensor.bytes = sizeof(src_data);

  float dst[4];
  ASSERT_NOT_OK(CreateVectorCopyData(tflite_tensor, dst));

  TfLiteIntArrayFree(tflite_tensor.dims);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
