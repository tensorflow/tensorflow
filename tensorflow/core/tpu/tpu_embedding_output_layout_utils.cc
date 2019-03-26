/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_embedding_output_layout_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/tpu/tpu_embedding_output_layout.pb.h"

namespace tensorflow {
namespace tpu {

void AddDefaultEmbeddingOutputLayoutIfNeeded(
    TPUEmbeddingConfiguration* config) {
  if (config->has_output_layout()) {
    // Model or previous step has already filled this in.
    return;
  }

  TPUEmbeddingOutputLayout* layout = config->mutable_output_layout();
  // Create output tensors.
  for (const auto& table : config->table_descriptor()) {
    TPUEmbeddingOutputLayout::EmbeddingOutputTensor* output =
        layout->add_output();
    TPUEmbeddingOutputLayout::TwoDOutputTensor* two_d = output->mutable_two_d();
    two_d->set_dim1_size(table.dimension());
    two_d->set_dim0_size_per_sample(table.num_features());
  }

  // Create table output locations.
  for (int table_id = 0; table_id < config->table_descriptor_size();
       ++table_id) {
    TPUEmbeddingOutputLayout::TableDescriptor* output_table =
        layout->add_table();
    const auto& table = config->table_descriptor(table_id);
    for (int feature_index = 0; feature_index < table.num_features();
         ++feature_index) {
      TPUEmbeddingOutputLayout::FeatureDescriptor* output_feature =
          output_table->add_feature();
      TPUEmbeddingOutputLayout::OutputLocation* output_location =
          output_feature->add_output_location();
      output_location->set_tensor_index(table_id);
      output_location->set_dim0_offset(feature_index);
      output_location->set_dim1_offset(0);
    }
  }
}

Status ComputeOutputTensorShapes(const TPUEmbeddingConfiguration& config,
                                 std::vector<TensorShapeProto>* shapes) {
  if (!config.has_output_layout()) {
    return errors::InvalidArgument(
        "TPUEmbeddingConfiguration is missing output layout.");
  }
  const TPUEmbeddingOutputLayout& layout = config.output_layout();
  int batch_size = config.batch_size_per_tensor_core();

  for (int i = 0; i < layout.output_size(); ++i) {
    const auto& output = layout.output(i);
    TensorShapeProto shape;
    switch (output.output_format_case()) {
      case TPUEmbeddingOutputLayout::EmbeddingOutputTensor::OutputFormatCase::
          kTwoD: {
        auto* dim0 = shape.add_dim();
        dim0->set_size(output.two_d().dim0_size_per_sample() * batch_size);
        auto* dim1 = shape.add_dim();
        dim1->set_size(output.two_d().dim1_size());
        break;
      }
      case TPUEmbeddingOutputLayout::EmbeddingOutputTensor::OutputFormatCase::
          OUTPUT_FORMAT_NOT_SET: {
        return errors::InvalidArgument(
            "Output layout in TPUEmbeddingConfiguration has unset embedding "
            "output tensor format.");
      }
      default: {
        return errors::InvalidArgument(
            "Output layout in TPUEmbeddingConfiguration has invalid or "
            "unhandled embedding output tensor format.");
      }
    }
    shapes->push_back(shape);
  }
  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
