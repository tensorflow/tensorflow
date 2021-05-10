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

#include "tensorflow/lite/delegates/gpu/common/transformations/global_pooling_to_reduce_op.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

TEST(MakeMeanFromGlobalAveragePooling, Smoke) {
  GraphFloat32 graph;
  auto input = graph.NewValue();
  input->tensor.shape = BHWC(1, 4, 4, 8);

  Pooling2DAttributes attr;
  attr.padding.prepended = tflite::gpu::HW(0, 0);
  attr.padding.appended = tflite::gpu::HW(0, 0);
  attr.strides = tflite::gpu::HW(4, 4);
  attr.kernel = tflite::gpu::HW(4, 4);
  attr.type = tflite::gpu::PoolingType::AVERAGE;
  attr.output_indices = false;

  auto pool_node = graph.NewNode();
  pool_node->operation.type = ToString(OperationType::POOLING_2D);
  pool_node->operation.attributes = attr;

  ASSERT_TRUE(graph.AddConsumer(pool_node->id, input->id).ok());

  Value* output = nullptr;
  ASSERT_TRUE(AddOutput(&graph, pool_node, &output).ok());
  output->tensor.shape = BHWC(1, 1, 1, 8);

  ASSERT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());

  auto transformation = NewGlobalPoolingToReduceOp();
  ModelTransformer transformer(&graph, nullptr);
  transformer.Apply("global_average_pooling_to_mean", transformation.get());

  ASSERT_EQ(1, graph.nodes().size());
  ASSERT_EQ(2, graph.values().size());
  ASSERT_EQ(ToString(OperationType::MEAN), graph.nodes()[0]->operation.type);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
