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
#include "tensorflow/lite/delegates/hexagon/utils.h"

#include <gtest/gtest.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/common.h"

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

// Regression test for the out-of-bounds access fixed in this change:
// IsNodeSupportedByHexagon must reject CONCATENATION and PACK nodes that have
// zero input or output tensors (which previously led to OOB reads in the
// Concat/Pack builders).
TEST(UtilsTest, ConcatAndPackEmptyInputsOrOutputsRejected) {
  TfLiteContext context = {};
  TfLiteRegistration reg = {};
  TfLiteNode node = {};

  TfLiteIntArray* empty_inputs = TfLiteIntArrayCreate(0);
  TfLiteIntArray* empty_outputs = TfLiteIntArrayCreate(0);
  TfLiteIntArray* valid_inputs = TfLiteIntArrayCreate(1);
  valid_inputs->data[0] = 0;
  TfLiteIntArray* valid_outputs = TfLiteIntArrayCreate(1);
  valid_outputs->data[0] = 1;

  TfLiteTensor tensors[2] = {};
  TfLiteIntArray* dims = TfLiteIntArrayCreate(2);
  dims->data[0] = 2;
  dims->data[1] = 2;
  tensors[0].dims = dims;
  tensors[0].type = kTfLiteUInt8;
  tensors[1].dims = dims;
  tensors[1].type = kTfLiteUInt8;

  context.tensors = tensors;
  context.tensors_size = 2;

  reg.version = 1;

  for (const int builtin_code :
       {kTfLiteBuiltinConcatenation, kTfLiteBuiltinPack}) {
    reg.builtin_code = builtin_code;

    // Empty inputs -> rejected.
    node.inputs = empty_inputs;
    node.outputs = valid_outputs;
    EXPECT_FALSE(IsNodeSupportedByHexagon(&reg, &node, &context));

    // Empty outputs -> rejected.
    node.inputs = valid_inputs;
    node.outputs = empty_outputs;
    EXPECT_FALSE(IsNodeSupportedByHexagon(&reg, &node, &context));

    // Valid inputs and outputs -> accepted.
    node.inputs = valid_inputs;
    node.outputs = valid_outputs;
    EXPECT_TRUE(IsNodeSupportedByHexagon(&reg, &node, &context));
  }

  TfLiteIntArrayFree(empty_inputs);
  TfLiteIntArrayFree(empty_outputs);
  TfLiteIntArrayFree(valid_inputs);
  TfLiteIntArrayFree(valid_outputs);
  TfLiteIntArrayFree(dims);
}

}  // namespace
}  // namespace tflite
