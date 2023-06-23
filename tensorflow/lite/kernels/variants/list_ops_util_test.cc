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
#include "tensorflow/lite/kernels/variants/list_ops_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace variants {
namespace {

TEST(TensorAsShape, ScalarTensorReturnsEmptyIntArray) {
  TensorUniquePtr scalar_tensor =
      BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({}), kTfLiteDynamic);

  IntArrayUniquePtr shape_from_tensor = TensorAsShape(*scalar_tensor);
  ASSERT_THAT(shape_from_tensor.get(), DimsAre({}));
}

TEST(TensorAsShape, SingleElementTensorReturnsSize1Shape) {
  TensorUniquePtr single_el_tensor =
      BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({1}), kTfLiteDynamic);
  single_el_tensor->data.i32[0] = 10;

  IntArrayUniquePtr shape_from_tensor = TensorAsShape(*single_el_tensor);
  ASSERT_THAT(shape_from_tensor.get(), DimsAre({10}));
}

TEST(TensorAsShape, OneDMultipleElementShapeReturnsHighRankedShape) {
  TensorUniquePtr one_d_mul_el_tensor =
      BuildTfLiteTensor(kTfLiteInt32, BuildTfLiteArray({3}), kTfLiteDynamic);
  one_d_mul_el_tensor->data.i32[0] = 10;
  one_d_mul_el_tensor->data.i32[1] = 9;
  one_d_mul_el_tensor->data.i32[2] = 8;

  IntArrayUniquePtr shape_from_tensor = TensorAsShape(*one_d_mul_el_tensor);
  ASSERT_THAT(shape_from_tensor.get(), DimsAre({10, 9, 8}));
}

}  // namespace
}  // namespace variants
}  // namespace tflite
