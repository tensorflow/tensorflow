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

#include "tensorflow/lite/delegates/gpu/gl/kernels/relu.h"

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

class ReluTest : public ::testing::Test {
 public:
  ReluTest() = default;
  ~ReluTest() override = default;

  TensorRef<BHWC> GetTensorRef(int ref) {
    TensorRef<BHWC> tensor_ref;
    tensor_ref.type = DataType::FLOAT32;
    tensor_ref.ref = ref;
    tensor_ref.shape = BHWC(1, 2, 2, 1);
    return tensor_ref;
  }
};

TEST_F(ReluTest, Smoke) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_max = 0;
  attr.alpha = 0;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 0.0, 2.0, 8.0}));
}

TEST_F(ReluTest, ClipOnly) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_max = 6;
  attr.alpha = 0;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 0.0, 2.0, 6.0}));
}

TEST_F(ReluTest, AlphaOnly) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_max = 0;
  attr.alpha = 0.5;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-3.0, 0.0, 2.0, 8.0}));
}

TEST_F(ReluTest, ClipAndAlpha) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_max = 6;
  attr.alpha = 0.5;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-3.0, 0.0, 2.0, 6.0}));
}

TEST_F(ReluTest, ReLUN1Smoke) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_min = -1;
  attr.activation_max = 0;
  attr.alpha = 0;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-12.0f, -0.5f, 0.8f, 3.2f}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-1.0f, -0.5f, 0.8f, 3.2f}));
}

TEST_F(ReluTest, ReLUN1ClipOnly) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_min = -1;
  attr.activation_max = 1;
  attr.alpha = 0;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-12.0f, -0.5f, 0.8f, 3.2f}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-1.0f, -0.5f, 0.8f, 1.0f}));
}

TEST_F(ReluTest, ReLUN1AlphaOnly) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_min = -1;  // activation_min ignored if alpha != 0
  attr.activation_max = 0;
  attr.alpha = 0.5;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-3.0, 0.0, 2.0, 8.0}));
}

TEST_F(ReluTest, ReLUN1ClipAndAlpha) {
  OperationType op_type = OperationType::RELU;
  ReLUAttributes attr;
  attr.activation_min = -1;  // activation_min ignored if alpha != 0
  attr.activation_max = 6;
  attr.alpha = 0.5;
  SingleOpModel model({ToString(op_type), attr}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {-6.0, 0.0, 2.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewReLUNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-3.0, 0.0, 2.0, 6.0}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
