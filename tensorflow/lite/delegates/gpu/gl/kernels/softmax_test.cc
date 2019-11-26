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

#include "tensorflow/lite/delegates/gpu/gl/kernels/softmax.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/test_util.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TEST(SoftmaxTest, Softmax) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {0.1, 0.2, 0.1, 0.2}));
  ASSERT_OK(model.Invoke(*NewSoftmaxNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 1, 1, 1}));
}

TEST(SoftmaxTest, DoesNotWorkForHeightAxis) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftmaxAttributes attr;
  attr.axis = Axis::HEIGHT;

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {0.1, 0.2, 0.3, 0.4}));
  EXPECT_FALSE(model.Invoke(*NewSoftmaxNodeShader()).ok());
}

TEST(SoftmaxTest, DoesNotWorkForWidthAxis) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftmaxAttributes attr;
  attr.axis = Axis::WIDTH;

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {0.1, 0.2, 0.3, 0.4}));
  EXPECT_FALSE(model.Invoke(*NewSoftmaxNodeShader()).ok());
}

TEST(SoftmaxTest, Softmax1x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 1, 1, 4);

  SoftmaxAttributes attr;
  attr.axis = Axis::CHANNELS;

  const double sum =
      std::exp(0.1) + std::exp(0.2) + std::exp(0.3) + std::exp(0.4);

  SingleOpModel model({ToString(OperationType::SOFTMAX), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {0.1, 0.2, 0.3, 0.4}));
  ASSERT_OK(model.Invoke(*NewSoftmaxNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {std::exp(0.1) / sum, std::exp(0.2) / sum,
                                  std::exp(0.3) / sum, std::exp(0.4) / sum}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
