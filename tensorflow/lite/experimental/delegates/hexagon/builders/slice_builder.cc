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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/slice_builder.h"

#include <vector>

#include "tensorflow/lite/kernels/internal/tensor.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {
template <typename T>
void GetBeginAndSizeVectors(int dimensions, const TfLiteTensor* begin,
                            const TfLiteTensor* size, std::vector<int>* begins,
                            std::vector<int>* sizes) {
  for (int i = 0; i < dimensions; ++i) {
    begins->push_back(GetTensorData<T>(begin)[i]);
    sizes->push_back(GetTensorData<T>(size)[i]);
  }
}
}  // namespace

TfLiteStatus SliceOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                              const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  static int quant_bound_shape[] = {1, 1, 1, 1};

  // Input data tensor.
  const int tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  // Start / Size
  const auto& begin_tensor = context->tensors[inputs->data[1]];
  const auto& size_tensor = context->tensors[inputs->data[2]];
  std::vector<int32_t> begins, sizes;
  if (begin_tensor.type == kTfLiteInt32) {
    GetBeginAndSizeVectors<int32_t>(input_tensor.dims->size, &begin_tensor,
                                    &size_tensor, &begins, &sizes);
  } else if (begin_tensor.type == kTfLiteInt64) {
    GetBeginAndSizeVectors<int64_t>(input_tensor.dims->size, &begin_tensor,
                                    &size_tensor, &begins, &sizes);
  } else {
    return kTfLiteError;
  }
  const int32_t begins_shape[] = {1, 1, 1, static_cast<int32_t>(begins.size())};
  auto begins_node = graph_builder_->AddConstNodeWithData(
      begins_shape, reinterpret_cast<char*>(begins.data()),
      sizeof(int32_t) * begins.size());
  auto sizes_node = graph_builder_->AddConstNodeWithData(
      begins_shape, reinterpret_cast<char*>(sizes.data()),
      sizeof(int32_t) * begins.size());
  AddInput(TensorID(begins_node->GetID(), 0));
  AddInput(TensorID(sizes_node->GetID(), 0));

  // Input min/max
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(input_tensor, &input_min_, &input_max_));
  auto* input_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input_min_),
      sizeof(input_min_));
  auto* input_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input_max_),
      sizeof(input_max_));
  AddInput(TensorID(input_min_const->GetID(), 0));
  AddInput(TensorID(input_max_const->GetID(), 0));

  // Outputs
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  return kTfLiteOk;
}

TfLiteStatus SliceOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                             TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

OpBuilder* CreateSliceOpBuilder(GraphBuilder* graph_builder, int op_type) {
  return new SliceOpBuilder(graph_builder, op_type);
}
}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
