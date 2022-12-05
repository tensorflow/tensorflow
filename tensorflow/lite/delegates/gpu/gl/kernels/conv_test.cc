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

#include "tensorflow/lite/delegates/gpu/gl/kernels/conv.h"

#include <utility>
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

TEST(ConvTest, O2H2W1I1Stride1x1Dilation1x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data = {1, 1};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(2, 2, 1, 1);
  weights.id = 2;
  weights.data = {1, 2, 3, 4};
  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(1, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 2, 2, 2);

  SingleOpModel model(
      {ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 1, 1, 1}));
  ASSERT_OK(model.Invoke(*NewConvolutionNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {4, 8, 4, 8, 2, 4, 2, 4}));
}

TEST(ConvTest, O1H2W2I1Stride1x1Dilation2x2) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 3, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data.push_back(0.0);
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 2, 2, 1);
  weights.id = 2;
  weights.data = {1, 2, 3, 4};
  attr.weights = std::move(weights);

  attr.dilations = HW(2, 2);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 1, 1, 1);

  SingleOpModel model(
      {ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 1, 1, 1, 1, 1, 1, 1, 1}));
  ASSERT_OK(model.Invoke(*NewConvolutionNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {10}));
}

TEST(ConvTest, O1H3W3I1Stride1x1Dilation1x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 1;
  bias.id = 1;
  bias.data.push_back(1.0);
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 3, 3, 1);
  weights.id = 2;
  weights.data = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(1, 1);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 1, 1, 1);

  SingleOpModel model(
      {ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 1, 1, 1}));
  ASSERT_OK(model.Invoke(*NewConvolutionNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {11}));
}

TEST(ConvTest, O2H1W1I2Stride1x1Dilation1x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 1, 2);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data = {1, 1};
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(2, 1, 1, 2);
  weights.id = 2;
  weights.data = {1, 2, 3, 4};
  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 2, 1, 2);

  SingleOpModel model(
      {ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 1, 1, 1}));
  ASSERT_OK(model.Invoke(*NewConvolution1x1NodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {4, 8, 4, 8}));
}

TEST(ConvTest, O1H1W1I1Stride2x2Dilation1x1) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 3, 3, 1);

  Convolution2DAttributes attr;
  Tensor<Linear, DataType::FLOAT32> bias;
  bias.shape.v = 2;
  bias.id = 1;
  bias.data.push_back(0.0);
  attr.bias = std::move(bias);

  Tensor<OHWI, DataType::FLOAT32> weights;
  weights.shape = OHWI(1, 1, 1, 1);
  weights.id = 2;
  weights.data.push_back(2.0);

  attr.weights = std::move(weights);

  attr.dilations = HW(1, 1);
  attr.padding.prepended = HW(0, 0);
  attr.padding.appended = HW(0, 0);
  attr.strides = HW(2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 3;
  output.shape = BHWC(1, 2, 2, 1);

  SingleOpModel model(
      {ToString(OperationType::CONVOLUTION_2D), std::move(attr)}, {input},
      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 0, 2, 0, 0, 0, 4, 0, 8}));
  ASSERT_OK(model.Invoke(*NewConvolutionNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {2, 4, 8, 16}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
