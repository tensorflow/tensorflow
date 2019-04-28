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

#include "tensorflow/lite/delegates/gpu/common/model_builder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {
namespace gpu {
namespace {

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank0) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(1);
  tflite_tensor.dims->data[0] = 4;
  TensorRefFloat32 tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::FLOAT32);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 1, 1, 1));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank1) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = TfLiteType::kTfLiteInt32;
  tflite_tensor.dims = TfLiteIntArrayCreate(2);
  tflite_tensor.dims->data[0] = 4;
  tflite_tensor.dims->data[1] = 5;
  TensorRefFloat32 tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::INT32);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 1, 1, 5));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank2) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = TfLiteType::kTfLiteInt64;
  tflite_tensor.dims = TfLiteIntArrayCreate(3);
  tflite_tensor.dims->data[0] = 4;
  tflite_tensor.dims->data[1] = 5;
  tflite_tensor.dims->data[2] = 6;
  TensorRefFloat32 tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::INT64);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 1, 5, 6));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefSucceedsForRank3) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = TfLiteType::kTfLiteUInt8;
  tflite_tensor.dims = TfLiteIntArrayCreate(4);
  tflite_tensor.dims->data[0] = 4;
  tflite_tensor.dims->data[1] = 5;
  tflite_tensor.dims->data[2] = 6;
  tflite_tensor.dims->data[3] = 7;
  TensorRefFloat32 tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(tensor_ref.type, DataType::UINT8);
  EXPECT_EQ(tensor_ref.shape, BHWC(4, 5, 6, 7));
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefFailsForRankLT0) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(0);
  TensorRefFloat32 tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  // TODO(b/130054481): Cover scalar.
  EXPECT_FALSE(status.ok());
}

TEST(ModelBuilderTest, ConvertTfLiteTensorToTensorRefFailsForRankGT3) {
  TfLiteTensor tflite_tensor;
  tflite_tensor.type = TfLiteType::kTfLiteFloat32;
  tflite_tensor.dims = TfLiteIntArrayCreate(5);
  TensorRefFloat32 tensor_ref;
  const auto status =
      ConvertTfLiteTensorToTensorRef(tflite_tensor, &tensor_ref);
  TfLiteIntArrayFree(tflite_tensor.dims);
  EXPECT_FALSE(status.ok());
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
