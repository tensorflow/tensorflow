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

#include "tensorflow/lite/delegates/gpu/gl/kernels/slice.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/test_util.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TEST(SliceTest, Identity) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 2, 2);

  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 0, 0);
  attr.ends = BHWC(input.shape.b, 1, 2, 2);
  attr.strides = BHWC(1, 1, 1, 1);

  SingleOpModel model({ToString(OperationType::SLICE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewSliceNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 2, 3, 4}));
}

TEST(SliceTest, NoStrides) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 2, 1);

  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 0, 0);
  attr.ends = BHWC(input.shape.b, 1, 2, 1);
  attr.strides = BHWC(1, 1, 1, 1);

  SingleOpModel model({ToString(OperationType::SLICE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewSliceNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 3}));
}

TEST(SliceTest, NoStridesStartOffset) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 1, 2);

  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 1, 0);
  attr.ends = BHWC(input.shape.b, 1, 2, 2);
  attr.strides = BHWC(1, 1, 1, 1);

  SingleOpModel model({ToString(OperationType::SLICE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewSliceNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {3, 4}));
}

TEST(SliceTest, StridesByHeight) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 4, 1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 1, 1);

  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 0, 0);
  attr.ends = BHWC(input.shape.b, 4, 1, 1);
  attr.strides = BHWC(1, 2, 1, 1);

  SingleOpModel model({ToString(OperationType::SLICE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewSliceNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 3}));
}

TEST(SliceTest, StridesByWidth) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 4, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 2, 1);

  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 1, 0);
  attr.ends = BHWC(input.shape.b, 1, 4, 1);
  attr.strides = BHWC(1, 1, 2, 1);

  SingleOpModel model({ToString(OperationType::SLICE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewSliceNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {2, 4}));
}

TEST(SliceTest, StridesByChannels) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 1, 2);

  SliceAttributes attr;
  attr.starts = BHWC(0, 0, 0, 1);
  attr.ends = BHWC(input.shape.b, 1, 1, 4);
  attr.strides = BHWC(1, 1, 1, 2);

  SingleOpModel model({ToString(OperationType::SLICE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewSliceNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {2, 4}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
