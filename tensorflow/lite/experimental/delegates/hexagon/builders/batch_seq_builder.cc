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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/batch_seq_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {

TfLiteStatus BatchSeqBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  // Add config input.
  static const int config_shape[] = {1, 1, 1, 3};
  // TODO(b/152562126): Allow custom setting for BQ (preferred batch multiple),
  // and Options.
  // BQ is preferred batch multiple
  // Options is currently 0 or 1, 0 is default and batches
  // will run in increasing order, this behavior can be disabled by setting 1.
  // Refer to Hexagon NN docs for more details.
  int32_t config[] = {max_size_for_batch_, 1, 0};

  auto* input_config = graph_builder_->AddConstNodeWithData(
      config_shape, reinterpret_cast<char*>(&config), sizeof(int32_t) * 3);
  AddInput(TensorID(input_config->GetID(), 0));

  // Add Input batch details.
  const int input_batch_dims_shape[] = {1, 1, 1, input_batch_dims_->size};
  auto* input_batch_dims_node = graph_builder_->AddConstNodeWithData(
      input_batch_dims_shape, reinterpret_cast<char*>(input_batch_dims_->data),
      sizeof(input_batch_dims_[0]) * input_batch_dims_->size);
  AddInput(TensorID(input_batch_dims_node->GetID(), 0));

  // Add Output batch details.
  const int output_batch_dims_shape[] = {1, 1, 1, output_batch_dims_->size};
  auto* output_batch_dims_node = graph_builder_->AddConstNodeWithData(
      output_batch_dims_shape,
      reinterpret_cast<char*>(output_batch_dims_->data),
      sizeof(output_batch_dims_[0]) * output_batch_dims_->size);
  AddInput(TensorID(output_batch_dims_node->GetID(), 0));

  return kTfLiteOk;
}

OpBuilder* CreateBatchSeqBuilder(GraphBuilder* graph_builder, int op_type,
                                 int max_size_for_batch,
                                 TfLiteIntArray* input_batch_dimensions,
                                 TfLiteIntArray* output_batch_dimensions) {
  auto* builder = new BatchSeqBuilder(graph_builder, op_type);
  builder->SetMaxSizeForBatch(max_size_for_batch);
  builder->SetInputBatchDimensions(input_batch_dimensions);
  builder->SetOutputBatchDimensions(output_batch_dimensions);
  return builder;
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
