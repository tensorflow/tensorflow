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
#include "tensorflow/lite/experimental/delegates/hexagon/utils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace {

TEST(UtilsTest, Get4DShapeTest_4DInput) {
  unsigned int batch_dim, height_dim, width_dim, depth_dim;
  TfLiteIntArray* shape_4d = TfLiteIntArrayCreate(4);
  shape_4d->data[0] = 4;
  shape_4d->data[1] = 3;
  shape_4d->data[2] = 2;
  shape_4d->data[3] = 1;
  EXPECT_EQ(
      Get4DShape(&batch_dim, &height_dim, &width_dim, &depth_dim, shape_4d),
      kTfLiteOk);
  EXPECT_EQ(batch_dim, shape_4d->data[0]);
  EXPECT_EQ(height_dim, shape_4d->data[1]);
  EXPECT_EQ(width_dim, shape_4d->data[2]);
  EXPECT_EQ(depth_dim, shape_4d->data[3]);

  TfLiteIntArrayFree(shape_4d);
}

TEST(UtilsTest, Get4DShapeTest_2DInput) {
  unsigned int batch_dim, height_dim, width_dim, depth_dim;
  TfLiteIntArray* shape_2d = TfLiteIntArrayCreate(2);
  shape_2d->data[0] = 4;
  shape_2d->data[1] = 3;
  EXPECT_EQ(
      Get4DShape(&batch_dim, &height_dim, &width_dim, &depth_dim, shape_2d),
      kTfLiteOk);
  EXPECT_EQ(batch_dim, 1);
  EXPECT_EQ(height_dim, 1);
  EXPECT_EQ(width_dim, shape_2d->data[0]);
  EXPECT_EQ(depth_dim, shape_2d->data[1]);

  TfLiteIntArrayFree(shape_2d);
}

TEST(UtilsTest, Get4DShapeTest_5DInput) {
  unsigned int batch_dim, height_dim, width_dim, depth_dim;
  TfLiteIntArray* shape_5d = TfLiteIntArrayCreate(5);
  EXPECT_EQ(
      Get4DShape(&batch_dim, &height_dim, &width_dim, &depth_dim, shape_5d),
      kTfLiteError);

  TfLiteIntArrayFree(shape_5d);
}

}  // namespace
}  // namespace tflite
