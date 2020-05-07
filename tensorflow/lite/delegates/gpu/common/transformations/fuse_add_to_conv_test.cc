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

#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace {

TEST(MergeConvolutionWithAddTest, Smoke) {
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

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(16);
  add_tensor.data.resize(16);
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  auto add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::ADD);
  add_node->operation.attributes = add_attr;

  ASSERT_TRUE(graph.AddConsumer(conv_node->id, input->id).ok());

  Value* output;
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  output->tensor.shape = BHWC(1, 4, 4, 16);

  Value* link1;
  ASSERT_TRUE(ConnectTwoNodes(&graph, conv_node, add_node, &link1).ok());
  link1->tensor.shape = BHWC(1, 4, 4, 16);

  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewMergeConvolutionWithAdd();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("merge_convolution_with_add", transformation.get());

  EXPECT_EQ(1, graph.nodes().size());
  EXPECT_EQ(2, graph.values().size());
  EXPECT_EQ(ToString(OperationType::CONVOLUTION_2D),
            graph.nodes()[0]->operation.type);
}

TEST(MergeAddWithConvolutionTest, Smoke) {
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

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(8);
  add_tensor.data.resize(8);
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  auto conv_node = graph.NewNode();
  conv_node->operation.type = ToString(OperationType::CONVOLUTION_2D);
  conv_node->operation.attributes = conv_attr;
  auto add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::ADD);
  add_node->operation.attributes = add_attr;

  ASSERT_TRUE(graph.AddConsumer(add_node->id, input->id).ok());

  Value* output;
  ASSERT_TRUE(AddOutput(&graph, conv_node, &output).ok());
  output->tensor.shape = BHWC(1, 4, 4, 16);

  Value* link1;
  ASSERT_TRUE(ConnectTwoNodes(&graph, add_node, conv_node, &link1).ok());
  link1->tensor.shape = BHWC(1, 4, 4, 16);

  ASSERT_EQ(2, graph.nodes().size());
  ASSERT_EQ(3, graph.values().size());

  auto transformation = NewMergeAddWithConvolution();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("merge_add_with_convolution", transformation.get());

  EXPECT_EQ(1, graph.nodes().size());
  EXPECT_EQ(2, graph.values().size());
  EXPECT_EQ(ToString(OperationType::CONVOLUTION_2D),
            graph.nodes()[0]->operation.type);
}

TEST(FuseAddAfterConvolution2DTest, Smoke) {
  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {0.3f, 0.7f};
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  FuseConvolution2DWithAdd(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.4f, 1.9f}));
}

TEST(FuseAddAfterDepthwiseConvolution2DTest, Smoke) {
  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1.1f, 1.2f, 1.3f, 1.4f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(4);
  add_tensor.data = {0.3f, 0.7f, 0.5f, 0.1f};
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  FuseDepthwiseConvolution2DWithAdd(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data,
              Pointwise(FloatNear(1e-6), {1.4f, 1.9f, 1.8f, 1.5f}));
}

TEST(FuseAddAfterConvolutionTransposedTest, Smoke) {
  ConvolutionTransposedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {0.3f, 0.7f};
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  FuseConvolutionTransposedWithAdd(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.4f, 1.9f}));
}

TEST(FuseAddAfterFullyConnectedTest, Smoke) {
  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {0.3f, 0.7f};
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  FuseFullyConnectedWithAdd(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6), {0.1f, 0.2f, 0.3f, 0.4f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.4f, 1.9f}));
}

TEST(FuseAddBeforeConvolution2DTest, Smoke) {
  Convolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {2.0f, 0.5f};
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  FuseAddWithConvolution2D(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {2.2f, 4.3f}));
}

TEST(FuseAddBeforeDepthwiseConvolution2DTest, Smoke) {
  DepthwiseConvolution2DAttributes attr;
  attr.weights.shape = OHWI(2, 1, 2, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  attr.bias.shape = Linear(4);
  attr.bias.data = {1.1f, 1.2f, 1.3f, 1.4f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(4);
  add_tensor.data = {0.3f, 0.7f, 0.5f, 0.1f};
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  FuseAddWithDepthwiseConvolution2D(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6),
                        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}));
  EXPECT_THAT(attr.bias.data,
              Pointwise(FloatNear(1e-6), {1.22f, 1.56f, 1.72f, 2.38f}));
}

TEST(FuseAddBeforeFullyConnectedTest, Smoke) {
  FullyConnectedAttributes attr;
  attr.weights.shape = OHWI(2, 1, 1, 2);
  attr.weights.data = {0.1f, 0.2f, 0.3f, 0.4f};
  attr.bias.shape = Linear(2);
  attr.bias.data = {1.1f, 1.2f};

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(2);
  add_tensor.data = {0.5f, 2.0f};
  AddAttributes add_attr;
  add_attr.param = add_tensor;

  FuseAddWithFullyConnected(add_attr, &attr);

  EXPECT_THAT(attr.weights.data,
              Pointwise(FloatNear(1e-6), {0.1f, 0.2f, 0.3f, 0.4f}));
  EXPECT_THAT(attr.bias.data, Pointwise(FloatNear(1e-6), {1.55f, 2.15f}));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
