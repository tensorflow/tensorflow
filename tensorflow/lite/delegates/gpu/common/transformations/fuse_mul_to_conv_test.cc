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

#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace {

TEST(MergeConvolutionWithMulTest, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 4, 4, 8);

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  conv_attr.weights.shape = OHWI(16, 3, 2, 8);
  conv_attr.weights.data.resize(conv_attr.weights.shape.DimensionsProduct());
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(16);

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(16);
  mul_tensor.data.resize(16);
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  auto mul_node = graph.NewNode();
  mul_node->operation.type = ToString(OperationType::MUL);
  mul_node->operation.attributes = mul_attr;

  ASSERT_TRUE(graph.AddConsumer(conv_node->id, input->id).ok());

  Value* output;
  ASSERT_TRUE(AddOutput(&graph, mul_node, &output).ok());
  output->tensor.shape = BHWC(1, 4, 4, 16);

  Value* link1;
  ASSERT_TRUE(ConnectTwoNodes(&graph, conv_node, mul_node, &link1).ok());
  link1->tensor.shape = BHWC(1, 4, 4, 16);

  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewMergeConvolutionWithMul();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("merge_convolution_with_mul", transformation.get());

  EXPECT_EQ(1, graph.nodes().size());
  EXPECT_EQ(2, graph.values().size());
  EXPECT_EQ(ToString(OperationType::CONVOLUTION_2D),
            graph.nodes()[0]->operation.type);
}

TEST(MergeMulWithConvolutionTest, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 4, 4, 8);

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(8);
  mul_tensor.data.resize(8);
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  Convolution2DAttributes conv_attr;
  conv_attr.padding.prepended = HW(0, 0);
  conv_attr.padding.appended = HW(0, 0);
  conv_attr.strides = HW(1, 1);
  conv_attr.dilations = HW(1, 1);
  conv_attr.weights.shape = OHWI(16, 3, 2, 8);
  conv_attr.weights.data.resize(conv_attr.weights.shape.DimensionsProduct());
  conv_attr.bias.shape = Linear(16);
  conv_attr.bias.data.resize(16);

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  auto mul_node = graph.NewNode();
  mul_node->operation.type = ToString(OperationType::MUL);
  mul_node->operation.attributes = mul_attr;

  ASSERT_TRUE(graph.AddConsumer(mul_node->id, input->id).ok());

  Value* output;
  ASSERT_TRUE(AddOutput(&graph, conv_node, &output).ok());
  output->tensor.shape = BHWC(1, 4, 4, 16);

  Value* link1;
  ASSERT_TRUE(ConnectTwoNodes(&graph, mul_node, conv_node, &link1).ok());
  link1->tensor.shape = BHWC(1, 4, 4, 16);

  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewMergeMulWithConvolution();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("merge_mul_with_convolution", transformation.get());

  EXPECT_EQ(1, graph.nodes().size());
  EXPECT_EQ(2, graph.values().size());
  EXPECT_EQ(ToString(OperationType::CONVOLUTION_2D),
            graph.nodes()[0]->operation.type);
}

TEST(FuseMulAfterConvolution2DTest, Smoke) {
  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseConvolution2DWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.1f, 0.15f, 0.2f, 1.0f, 1.2f, 1.4f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {0.75f, 5.0f}));
}

TEST(FuseMulAfterDepthwiseConvolution2DTest, Smoke) {
  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1.5f, 2.5f, 1.0f, 2.0f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(4);
  mul_tensor.data = {0.5f, 2.0f, 4.0f, 0.25f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseDepthwiseConvolution2DWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.8f, 0.15f, 1.6f, 1.0f, 0.15f, 1.4f, 0.2f}));
  EXPECT_THAT(attr.bias.data,
              Pointwise(FloatNear(1e-6), {0.75f, 5.0f, 4.0f, 0.5f}));
}

TEST(FuseMulAfterConvolutionTransposedTest, Smoke) {
  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseConvolutionTransposedWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.1f, 0.15f, 0.2f, 1.0f, 1.2f, 1.4f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {0.75f, 5.0f}));
}

TEST(FuseMulAfterFullyConnectedTest, Smoke) {
  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseFullyConnectedWithMultiply(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6), {0.05f, 0.1f, 0.6f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {0.75f, 5.0f}));
}

TEST(FuseMulBeforeConvolution2DTest, Smoke) {
  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithConvolution2D(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.4f, 0.15f, 0.8f, 0.25f, 1.2f, 0.35f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.5f, 2.5f}));
}

TEST(FuseMulBeforeDepthwiseConvolution2DTest, Smoke) {
  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1.5f, 2.5f, 1.0f, 2.0f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(4);
  mul_tensor.data = {0.5f, 2.0f, 4.0f, 0.25f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithDepthwiseConvolution2D(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.4f, 0.15f, 0.8f, 0.25f, 1.2f, 0.35f, 1.6f}));
  EXPECT_THAT(attr.bias.data,
              Pointwise(FloatNear(1e-6), {1.5f, 2.5f, 1.0f, 2.0f}));
}

TEST(FuseMulBeforeConvolutionTransposedTest, Smoke) {
  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithConvolutionTransposed(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.05f, 0.4f, 0.15f, 0.8f, 0.25f, 1.2f, 0.35f, 1.6f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.5f, 2.5f}));
}

TEST(FuseMulBeforeFullyConnectedTest, Smoke) {
  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.5f, 2.5f};

  Tensor<Linear, DataType::FLOAT32> mul_tensor;
  mul_tensor.shape = Linear(2);
  mul_tensor.data = {0.5f, 2.0f};
  MultiplyAttributes mul_attr;
  mul_attr.param = mul_tensor;

  FuseMultiplyWithFullyConnected(mul_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6), {0.05f, 0.4f, 0.15f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.5f, 2.5f}));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
