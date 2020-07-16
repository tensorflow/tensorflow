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

#include "tensorflow/lite/delegates/gpu/common/transformations/make_padding.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

namespace tflite {
namespace gpu {
namespace {

TEST(MakePadding, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 2, 3, 5);

  auto concat_node = graph.NewNode();
  ASSERT_TRUE(graph.AddConsumer(concat_node->id, input->id).ok());
  concat_node->operation.type = ToString(OperationType::CONCAT);
  ConcatAttributes attr;
  attr.axis = Axis::HEIGHT;
  concat_node->operation.attributes = attr;

  Value* output;
  ASSERT_TRUE(AddOutput(&graph, concat_node, &output).ok());
  output->tensor.shape = BHWC(1, 7, 3, 5);

  auto const_node = graph.NewNode();
  const_node->operation.type = ToString(OperationType::CONST);
  ConstTensorAttributes const_attr;
  const_attr.tensor.shape = BHWC(1, 5, 3, 5);
  const_attr.tensor.data =
      std::vector<float>(const_attr.tensor.shape.DimensionsProduct(), 0);
  const_node->operation.attributes = const_attr;

  Value* const_link;
  ASSERT_TRUE(
      ConnectTwoNodes(&graph, const_node, concat_node, &const_link).ok());
  const_link->tensor.shape = const_attr.tensor.shape;

  ASSERT_EQ(2, graph.nodes().size());

  auto transformation = NewMakePaddingFromConcat();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("make_padding", transformation.get());

  ASSERT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());
  auto pad_node = graph.nodes()[0];
  ASSERT_EQ(ToString(OperationType::PAD), pad_node->operation.type);
  auto pad_attr = absl::any_cast<PadAttributes>(pad_node->operation.attributes);
  EXPECT_EQ(BHWC(0, 0, 0, 0), pad_attr.prepended);
  EXPECT_EQ(BHWC(0, 5, 0, 0), pad_attr.appended);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
