/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/object_reader.h"

#include <gtest/gtest.h>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"

namespace tflite {
namespace gpu {
namespace {

TEST(ObjectReaderTest, GetInputTensorHandlesOptionalAndOutOfRange) {
  TfLiteTensor tensors[2] = {};
  TfLiteContext context = {};
  context.tensors = tensors;
  context.tensors_size = 2;

  TfLiteNode node = {};
  node.inputs = TfLiteIntArrayCreate(3);
  node.inputs->data[0] = 1;
  node.inputs->data[1] = kTfLiteOptionalTensor;
  node.inputs->data[2] = 7;
  node.outputs = TfLiteIntArrayCreate(0);

  GraphFloat32 graph;
  absl::flat_hash_map<int, Value*> tensor_to_value;
  ObjectReader reader(&graph, &context, &node, &tensor_to_value);

  EXPECT_EQ(reader.GetInputTensor(0), &tensors[1]);
  EXPECT_EQ(reader.GetInputTensor(1), nullptr);
  EXPECT_EQ(reader.GetInputTensor(2), nullptr);
  EXPECT_EQ(reader.GetInputTensor(3), nullptr);

  TfLiteIntArrayFree(node.inputs);
  TfLiteIntArrayFree(node.outputs);
}

TEST(ObjectReaderTest, GetOutputTensorHandlesOptionalAndOutOfRange) {
  TfLiteTensor tensors[2] = {};
  TfLiteContext context = {};
  context.tensors = tensors;
  context.tensors_size = 2;

  TfLiteNode node = {};
  node.inputs = TfLiteIntArrayCreate(0);
  node.outputs = TfLiteIntArrayCreate(3);
  node.outputs->data[0] = 0;
  node.outputs->data[1] = kTfLiteOptionalTensor;
  node.outputs->data[2] = 7;

  GraphFloat32 graph;
  absl::flat_hash_map<int, Value*> tensor_to_value;
  ObjectReader reader(&graph, &context, &node, &tensor_to_value);

  EXPECT_EQ(reader.GetOutputTensor(0), &tensors[0]);
  EXPECT_EQ(reader.GetOutputTensor(1), nullptr);
  EXPECT_EQ(reader.GetOutputTensor(2), nullptr);
  EXPECT_EQ(reader.GetOutputTensor(3), nullptr);

  TfLiteIntArrayFree(node.inputs);
  TfLiteIntArrayFree(node.outputs);
}

}  // namespace
}  // namespace gpu
}  // namespace tflite