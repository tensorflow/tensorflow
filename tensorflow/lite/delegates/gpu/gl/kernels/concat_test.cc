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

#include "tensorflow/lite/delegates/gpu/gl/kernels/concat.h"

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

TEST(ConcatTest, TwoInputTensorsByUnalignedChannel) {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 2, 2, 1);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 3, 5, 7}));
  ASSERT_TRUE(model.PopulateTensor(1, {2, 4, 6, 8}));
  ASSERT_OK(model.Invoke(*NewConcatNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(ConcatTest, TwoInputTensorsByAlignedChannel) {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 1, 1, 4);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 1, 1, 4);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 1, 8);

  ConcatAttributes attr;
  attr.axis = Axis::CHANNELS;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_TRUE(model.PopulateTensor(1, {5, 6, 7, 8}));
  ASSERT_OK(model.Invoke(*NewAlignedConcatNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(ConcatTest, TwoInputTensorsByHeight) {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 1, 2, 1);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 3, 2, 1);

  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2}));
  ASSERT_TRUE(model.PopulateTensor(1, {3, 4, 5, 6}));
  ASSERT_OK(model.Invoke(*NewFlatConcatNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1, 2, 3, 4, 5, 6}));
}

TEST(ConcatTest, TwoInputTensorsByWidth) {
  TensorRef<BHWC> input1, input2, output;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 2, 1, 1);

  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 3, 1);

  ConcatAttributes attr;
  attr.axis = Axis::WIDTH;

  SingleOpModel model({ToString(OperationType::CONCAT), attr}, {input1, input2},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 4}));
  ASSERT_TRUE(model.PopulateTensor(1, {2, 3, 5, 6}));
  ASSERT_OK(model.Invoke(*NewFlatConcatNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1, 2, 3, 4, 5, 6}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
