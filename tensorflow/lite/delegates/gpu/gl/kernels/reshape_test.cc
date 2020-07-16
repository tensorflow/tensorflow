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

#include "tensorflow/lite/delegates/gpu/gl/kernels/reshape.h"

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

TEST(Reshape, 1x2x3To3x2x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 3);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 3, 2, 1);

  ReshapeAttributes attr;
  attr.new_shape = output.shape;

  SingleOpModel model({ToString(OperationType::RESHAPE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4, 5, 6}));
  ASSERT_OK(model.Invoke(*NewReshapeNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1, 2, 3, 4, 5, 6}));
}

TEST(Reshape, 3x1x2To2x1x3) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 1, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 1, 3);

  ReshapeAttributes attr;
  attr.new_shape = output.shape;

  SingleOpModel model({ToString(OperationType::RESHAPE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4, 5, 6}));
  ASSERT_OK(model.Invoke(*NewReshapeNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1, 2, 3, 4, 5, 6}));
}

TEST(Reshape, 1x1x4To2x2x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 4);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  ReshapeAttributes attr;
  attr.new_shape = output.shape;

  SingleOpModel model({ToString(OperationType::RESHAPE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewReshapeNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 2, 3, 4}));
}

TEST(Reshape, BatchIsUnsupported) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(4, 1, 1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 2, 2, 1);

  ReshapeAttributes attr;
  attr.new_shape = output.shape;

  SingleOpModel model({ToString(OperationType::RESHAPE), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_THAT(
      model.Invoke(*NewReshapeNodeShader()).message(),
      testing::HasSubstr("Only identical batch dimension is supported"));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
