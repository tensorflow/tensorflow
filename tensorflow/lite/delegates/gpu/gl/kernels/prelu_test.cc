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

#include "tensorflow/lite/delegates/gpu/gl/kernels/prelu.h"

#include <utility>

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

TEST(PReluTest, LinearAlphaNoClip) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  PReLUAttributes attr;
  attr.clip = 0;
  Tensor<Linear, DataType::FLOAT32> alpha;
  alpha.shape.v = 1;
  alpha.id = 1;
  alpha.data = {2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(OperationType::PRELU), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {-1.0, -2.0, 1.0, 2.0}));
  ASSERT_OK(model.Invoke(*NewPReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {-2, -4, 1, 2}));
}

TEST(PReluTest, LinearAlphaWithClip) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  PReLUAttributes attr;
  attr.clip = 1.0;
  Tensor<Linear, DataType::FLOAT32> alpha;
  alpha.shape.v = 1;
  alpha.id = 1;
  alpha.data = {2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(OperationType::PRELU), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {-1.0, -2.0, 1.0, 2.0}));
  ASSERT_OK(model.Invoke(*NewPReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {-2, -4, 1, 1}));
}

TEST(PReluTest, 2DAlphaNoClip) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  OperationType op_type = OperationType::PRELU;
  PReLUAttributes attr;
  attr.clip = 0;
  Tensor<HWC, DataType::FLOAT32> alpha;
  alpha.shape = HWC(2, 2, 1);
  alpha.id = 1;
  alpha.data = {1, 2, 2, 2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(op_type), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -1.0, 2.0, -3.0}));
  ASSERT_OK(model.Invoke(*NewPReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {0, -2, 2, -6}));
}

TEST(PReluTest, 2DAlphaWithClip) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  OperationType op_type = OperationType::PRELU;
  PReLUAttributes attr;
  attr.clip = 1.0;
  Tensor<HWC, DataType::FLOAT32> alpha;
  alpha.shape = HWC(2, 2, 1);
  alpha.id = 1;
  alpha.data = {1, 2, 2, 2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model({ToString(op_type), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -1.0, 2.0, -3.0}));
  ASSERT_OK(model.Invoke(*NewPReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {0, -2, 1, -6}));
}

TEST(PReluTest, 2DAlphaWidthNotEqualHeight) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 1, 1);

  OperationType op_type = OperationType::PRELU;
  PReLUAttributes attr;
  attr.clip = 0;
  Tensor<HWC, DataType::FLOAT32> alpha;
  alpha.shape = HWC(2, 1, 1);
  alpha.id = 1;
  alpha.data = {1, 1};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 1, 1);

  SingleOpModel model({ToString(op_type), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {-1.0, -1.0}));
  ASSERT_OK(model.Invoke(*NewPReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {-1, -1}));
}

TEST(PReluTest, 3DAlphaNoClip) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 2);

  OperationType op_type = OperationType::PRELU;
  PReLUAttributes attr;
  attr.clip = 0;
  Tensor<HWC, DataType::FLOAT32> alpha;
  alpha.shape = HWC(2, 2, 2);
  alpha.id = 1;
  alpha.data = {1, 1, 2, 2, 2, 2, 2, 2};
  attr.alpha = std::move(alpha);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(op_type), attr}, {input}, {output});
  ASSERT_TRUE(
      model.PopulateTensor(0, {0.0, 0.0, -1.0, -1.0, 2.0, 2.0, -3.0, -3.0}));
  ASSERT_OK(model.Invoke(*NewPReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0, 0, -2, -2, 2, 2, -6, -6}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
