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

#include "tensorflow/lite/delegates/gpu/gl/kernels/add.h"

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

TEST(AddTest, TwoInputTensorsOfTheSameShape) {
  TensorRef<BHWC> augend, addend, output;
  augend.type = DataType::FLOAT32;
  augend.ref = 0;
  augend.shape = BHWC(1, 2, 2, 1);

  addend.type = DataType::FLOAT32;
  addend.ref = 1;
  addend.shape = BHWC(1, 2, 2, 1);

  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 1);

  ElementwiseAttributes attr;
  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)},
                      {augend, addend}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {-2.0, 0.2, 0.7, 0.8}));
  ASSERT_TRUE(model.PopulateTensor(1, {0.1, 0.2, 0.3, 0.5}));
  ASSERT_OK(model.Invoke(*NewAddNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-1.9, 0.4, 1.0, 1.3}));
}

TEST(AddTest, InputTensorAndScalar) {
  ElementwiseAttributes attr;
  attr.param = 0.1f;
  TensorRef<BHWC> input, output;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 1, 2);

  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = BHWC(1, 3, 1, 2);

  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {-2.0, 0.2, 0.7, 0.8, 1.1, 2.0}));
  ASSERT_OK(model.Invoke(*NewAddNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-1.9, 0.3, 0.8, 0.9, 1.2, 2.1}));
}

TEST(AddTest, InputTensorWithConstantBroadcast) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 2);

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> tensor;
  tensor.shape.v = 2;
  tensor.id = 1;
  tensor.data.push_back(10.0);
  tensor.data.push_back(20.0);
  attr.param = std::move(tensor);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)}, {input},
                      {output});
  ASSERT_TRUE(
      model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewAddNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6),
                        {11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0}));
}

TEST(AddTest, InputTensorWithRuntimeBroadcast) {
  TensorRef<BHWC> input1;
  input1.type = DataType::FLOAT32;
  input1.ref = 0;
  input1.shape = BHWC(1, 2, 2, 2);

  TensorRef<BHWC> input2;
  input2.type = DataType::FLOAT32;
  input2.ref = 1;
  input2.shape = BHWC(1, 1, 1, 2);

  ElementwiseAttributes attr;

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)},
                      {input1, input2}, {output});
  ASSERT_TRUE(
      model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {10.0, 20.0}));
  ASSERT_OK(model.Invoke(*NewAddNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6),
                        {11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0}));
}

TEST(AddTest, InputTensorWithConstantHWC) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 2);

  ElementwiseAttributes attr;
  Tensor<HWC, DataType::FLOAT32> tensor;
  tensor.shape = HWC(2, 2, 2);
  tensor.id = 1;
  tensor.data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  attr.param = std::move(tensor);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)}, {input},
                      {output});
  ASSERT_TRUE(
      model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewAddNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0}));
}

TEST(AddTest, InputTensorWithConstantHWCBroadcastChannels) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 2);

  ElementwiseAttributes attr;
  Tensor<HWC, DataType::FLOAT32> tensor;
  tensor.shape = HWC(2, 2, 1);
  tensor.id = 1;
  tensor.data = {1.0, 2.0, 3.0, 4.0};
  attr.param = std::move(tensor);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)}, {input},
                      {output});
  ASSERT_TRUE(
      model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewAddNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0}));
}

TEST(AddTest, InputTensorWithConstantHWCBroadcastWidth) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 2);

  ElementwiseAttributes attr;
  Tensor<HWC, DataType::FLOAT32> tensor;
  tensor.shape = HWC(2, 1, 2);
  tensor.id = 1;
  tensor.data = {1.0, 2.0, 3.0, 4.0};
  attr.param = std::move(tensor);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model({ToString(OperationType::ADD), std::move(attr)}, {input},
                      {output});
  ASSERT_TRUE(
      model.PopulateTensor(0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));
  ASSERT_OK(model.Invoke(*NewAddNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {2.0, 4.0, 4.0, 6.0, 8.0, 10.0, 10.0, 12.0}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
