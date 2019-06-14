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

TEST(SoftmaxTest, WorksForChannelsAxis) {
  TensorRefFloat32 input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRefFloat32 output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftMaxAttributes attr;
  attr.axis = Axis::CHANNELS;

  SingleOpModel model({ToString(OperationType::SOFT_MAX), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {0.1, 0.2, 0.1, 0.2}));
  ASSERT_OK(model.Invoke(*NewSoftMaxNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 1, 1, 1}));
}

TEST(SoftmaxTest, DoesNotWorkForHeightAxis) {
  TensorRefFloat32 input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRefFloat32 output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftMaxAttributes attr;
  attr.axis = Axis::HEIGHT;

  SingleOpModel model({ToString(OperationType::SOFT_MAX), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_THAT(
      model.Invoke(*NewSoftMaxNodeShader()).message(),
      testing::HasSubstr("Softmax is only supported for channels axis."));
}

TEST(SoftmaxTest, DoesNotWorkForWidthAxis) {
  TensorRefFloat32 input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRefFloat32 output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  SoftMaxAttributes attr;
  attr.axis = Axis::WIDTH;

  SingleOpModel model({ToString(OperationType::SOFT_MAX), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_THAT(
      model.Invoke(*NewSoftMaxNodeShader()).message(),
      testing::HasSubstr("Softmax is only supported for channels axis."));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
