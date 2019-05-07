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

#include "tensorflow/lite/delegates/gpu/common/transformations/match_dilated_convolution.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"

namespace tflite {
namespace gpu {
namespace {

TEST(MatchDilatedConvolutionTest, MakesDilatedConvolution) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 95, 1, 17);

  SpaceToBatchAttributes sb_attr;
  sb_attr.block = HW(128, 1);
  sb_attr.padding.prepended = HW(128, 0);
  sb_attr.padding.appended = HW(161, 0);

  DepthwiseConvolution2DAttributes dw_attr;
  dw_attr.padding.prepended = HW(0, 0);
  dw_attr.padding.appended = HW(0, 0);
  dw_attr.strides = HW(1, 1);
  dw_attr.dilations = HW(1, 1);
  dw_attr.weights.shape = OHWI(1, 3, 1, 17);
  dw_attr.bias.shape = Linear(96);

  BatchToSpaceAttributes bs_attr;
  bs_attr.block = HW(128, 1);
  bs_attr.crop.prepended = HW(0, 0);
  bs_attr.crop.appended = HW(33, 0);

  auto sb_node = graph.NewNode();
  sb_node->operation.type = ToString(OperationType::SPACE_TO_BATCH);
  sb_node->operation.attributes = sb_attr;
  auto dw_node = graph.NewNode();
  dw_node->operation.type = ToString(OperationType::DEPTHWISE_CONVOLUTION);
  dw_node->operation.attributes = dw_attr;
  auto bs_node = graph.NewNode();
  bs_node->operation.type = ToString(OperationType::BATCH_TO_SPACE);
  bs_node->operation.attributes = bs_attr;

  ASSERT_TRUE(graph.AddConsumer(sb_node->id, input->id).ok());

  Value<TensorRefFloat32>* output;
  ASSERT_TRUE(AddOutput(&graph, bs_node, &output).ok());
  output->tensor.shape = BHWC(1, 95, 1, 17);

  Value<TensorRefFloat32>* sb_link;
  ASSERT_TRUE(ConnectTwoNodes(&graph, sb_node, dw_node, &sb_link).ok());
  sb_link->tensor.shape = BHWC(21, 128, 1, 17);

  Value<TensorRefFloat32>* bs_link;
  ASSERT_TRUE(ConnectTwoNodes(&graph, dw_node, bs_node, &bs_link).ok());
  bs_link->tensor.shape = BHWC(1, 95, 1, 17);

  ASSERT_EQ(graph.nodes().size(), 3);
  ASSERT_EQ(graph.values().size(), 4);

  auto transformation = NewMatchDilatedConvolution();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("match_dilated_convolution", transformation.get());

  ASSERT_EQ(graph.nodes().size(), 1);
  ASSERT_EQ(graph.values().size(), 2);
  ASSERT_EQ(graph.nodes()[0]->operation.type,
            ToString(OperationType::DEPTHWISE_CONVOLUTION));

  auto updated_dw_attr = absl::any_cast<DepthwiseConvolution2DAttributes>(
      graph.nodes()[0]->operation.attributes);
  EXPECT_EQ(updated_dw_attr.padding.prepended, HW(128, 0));
  EXPECT_EQ(updated_dw_attr.padding.appended, HW(128, 0));
  EXPECT_EQ(updated_dw_attr.dilations, HW(128, 1));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
