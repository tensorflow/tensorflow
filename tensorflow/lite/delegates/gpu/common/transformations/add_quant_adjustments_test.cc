/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/transformations/add_quant_adjustments.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/any.h"
#include "absl/types/optional.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

void AddQuantParams(absl::optional<QuantizationParams>* params, float min,
                    float max, float scale) {
  params->emplace();
  params->value().min = min;
  params->value().max = max;
  params->value().scale = scale;
}

// Scenario:
// -> Add ->
//
// Since there is only one node output with no consumers, no new node should be
// added.
TEST(AddQuantAdjustments, OneNode) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 4, 4, 8);
  AddQuantParams(&input->quant_params, /*min=*/0.0, /*max=*/1.0,
                 /*scale=*/0.004);

  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(8);
  add_tensor.data.resize(8);
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;
  auto add_node = graph.NewNode();
  add_node->operation.type = ToString(OperationType::ADD);
  add_node->operation.attributes = add_attr;

  ASSERT_TRUE(graph.AddConsumer(add_node->id, input->id).ok());

  Value* output = nullptr;
  AddQuantParams(&input->quant_params, /*min=*/0.0, /*max=*/2.0,
                 /*scale=*/0.008);
  ASSERT_TRUE(AddOutput(&graph, add_node, &output).ok());
  output->tensor.shape = BHWC(1, 4, 4, 8);

  ASSERT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());

  auto transformation = NewAddQuantAdjustments();
  ModelTransformer transformer(&graph);
  transformer.Apply("add_quant_adjustments", transformation.get());

  EXPECT_EQ(1, graph.nodes().size());
  EXPECT_EQ(2, graph.values().size());
}

// Scenario:
// -> Add -> QuantizeAndDequantize -> Add ->
//        |                            ^
//        |                            |
//        ------------------------------
//
// A new QuantizeAndDequantize should only be added after the left/first 'Add'
// op, and it should connect to both its consumers.
TEST(AddQuantAdjustments, GeneralCase) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 4, 4, 8);
  AddQuantParams(&input->quant_params, /*min=*/0.0, /*max=*/1.0,
                 /*scale=*/0.004);

  // First Add.
  Tensor<Linear, DataType::FLOAT32> add_tensor;
  add_tensor.shape = Linear(8);
  add_tensor.data.resize(8);
  ElementwiseAttributes add_attr;
  add_attr.param = add_tensor;
  auto add1_node = graph.NewNode();
  add1_node->operation.type = ToString(OperationType::ADD);
  add1_node->operation.attributes = add_attr;
  // QuantizeAndDequantize.
  QuantizeAndDequantizeAttributes quant_attr;
  quant_attr.min = -1.0;
  quant_attr.max = 1.0;
  quant_attr.scale = 0.008;
  auto quant_node = graph.NewNode();
  quant_node->operation.type = ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
  quant_node->operation.attributes = quant_attr;
  // Second Add.
  auto add2_node = graph.NewNode();
  add2_node->operation.type = ToString(OperationType::ADD);

  // Connections.
  ASSERT_TRUE(graph.AddConsumer(add1_node->id, input->id).ok());
  Value* link1 = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, add1_node, quant_node, &link1).ok());
  AddQuantParams(&link1->quant_params, /*min=*/0.0, /*max=*/2.0,
                 /*scale=*/0.008);
  link1->tensor.shape = BHWC(1, 4, 4, 8);
  ASSERT_TRUE(graph.AddConsumer(add2_node->id, link1->id).ok());
  Value* link2 = nullptr;
  ASSERT_TRUE(ConnectTwoNodes(&graph, quant_node, add2_node, &link2).ok());
  AddQuantParams(&link2->quant_params, /*min=*/-1.0, /*max=*/1.0,
                 /*scale=*/0.008);
  link2->tensor.shape = BHWC(1, 4, 4, 8);
  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, add2_node, &output).ok());
  AddQuantParams(&output->quant_params, /*min=*/-1.0, /*max=*/1.0,
                 /*scale=*/0.008);
  output->tensor.shape = BHWC(1, 4, 4, 8);

  ASSERT_EQ(3, graph.nodes().size());
  ASSERT_EQ(4, graph.values().size());

  auto transformation = NewAddQuantAdjustments();
  ModelTransformer transformer(&graph);
  transformer.Apply("add_quant_adjustments", transformation.get());

  EXPECT_EQ(4, graph.nodes().size());
  EXPECT_EQ(5, graph.values().size());
  EXPECT_EQ(ToString(OperationType::ADD), graph.nodes()[0]->operation.type);
  // The new node should be inserted at index 1, just after add1.
  EXPECT_EQ(ToString(OperationType::QUANTIZE_AND_DEQUANTIZE),
            graph.nodes()[1]->operation.type);
  EXPECT_EQ(ToString(OperationType::QUANTIZE_AND_DEQUANTIZE),
            graph.nodes()[2]->operation.type);
  EXPECT_EQ(quant_node->id, graph.nodes()[2]->id);
  EXPECT_EQ(ToString(OperationType::ADD), graph.nodes()[3]->operation.type);
  auto new_quant_attr = absl::any_cast<QuantizeAndDequantizeAttributes>(
      graph.nodes()[1]->operation.attributes);
  EXPECT_EQ(0.0, new_quant_attr.min);
  EXPECT_EQ(2.0, new_quant_attr.max);
  const auto& new_quant_consumers = graph.FindConsumers(graph.values()[4]->id);
  EXPECT_EQ(2, new_quant_consumers.size());
  EXPECT_EQ(quant_node, new_quant_consumers[0]);
  EXPECT_EQ(add2_node, new_quant_consumers[1]);

  // Transformation should be idempotent.
  transformer.Apply("add_quant_adjustments", transformation.get());
  EXPECT_EQ(4, graph.nodes().size());
  EXPECT_EQ(5, graph.values().size());
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
