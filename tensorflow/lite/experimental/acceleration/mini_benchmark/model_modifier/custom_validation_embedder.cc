/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/custom_validation_embedder.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace acceleration {

constexpr int kModelVersion = 3;

MinibenchmarkStatus GenerateModelWithInput(
    const tflite::Model& plain_model,
    const std::vector<std::vector<uint8_t>>& new_input_buffer,
    flatbuffers::FlatBufferBuilder& output_model) {
  ModelT model;
  plain_model.UnPackTo(&model);

  if (plain_model.subgraphs()->size() == 0) {
    return kMinibenchmarkPreconditionNotMet;
  }

  // Create buffer with the new input.
  const SubGraph& main_graph = *(plain_model.subgraphs()->Get(0));
  if (main_graph.inputs()->size() != new_input_buffer.size()) {
    return kMinibenchmarkPreconditionNotMet;
  }

  std::vector<int> input_buffer_indexes;
  for (int i : *main_graph.inputs()) {
    input_buffer_indexes.push_back(main_graph.tensors()->Get(i)->buffer());
  }
  std::vector<flatbuffers::Offset<Buffer>> buffer_offset;
  buffer_offset.reserve(model.buffers.size());
  for (int i = 0; i < plain_model.buffers()->size(); i++) {
    auto input_buffer_index =
        std::find(input_buffer_indexes.begin(), input_buffer_indexes.end(), i);
    // If buffer i is an input, replace the data with new_input_buffer.
    if (input_buffer_index != input_buffer_indexes.end()) {
      auto new_input_iter =
          new_input_buffer.begin() +
          std::distance(input_buffer_indexes.begin(), input_buffer_index);
      buffer_offset.push_back(CreateBuffer(
          output_model, output_model.CreateVector(*new_input_iter)));
    } else {
      buffer_offset.push_back(
          CreateBuffer(output_model, model.buffers[i].get()));
    }
  }

  // Create all other fields.
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes;
  operator_codes.reserve(model.operator_codes.size());
  for (auto& iter : model.operator_codes) {
    operator_codes.push_back(CreateOperatorCode(output_model, iter.get()));
  }

  std::vector<flatbuffers::Offset<SubGraph>> subgraphs;
  subgraphs.reserve(model.subgraphs.size());
  for (auto& iter : model.subgraphs) {
    subgraphs.push_back(CreateSubGraph(output_model, iter.get()));
  }

  std::vector<flatbuffers::Offset<Metadata>> metadata;
  metadata.reserve(model.metadata.size());
  for (auto& iter : model.metadata) {
    metadata.push_back(CreateMetadata(output_model, iter.get()));
  }

  std::vector<flatbuffers::Offset<SignatureDef>> signature_defs;
  signature_defs.reserve(model.signature_defs.size());
  for (auto& iter : model.signature_defs) {
    signature_defs.push_back(CreateSignatureDef(output_model, iter.get()));
  }
  output_model.Finish(
      CreateModel(output_model, kModelVersion,
                  output_model.CreateVector(operator_codes),
                  output_model.CreateVector(subgraphs),
                  output_model.CreateString(model.description),
                  output_model.CreateVector(buffer_offset),
                  /* metadata_buffer */ 0, output_model.CreateVector(metadata),
                  output_model.CreateVector(signature_defs)),
      "TFL3");
  return kMinibenchmarkSuccess;
}

}  // namespace acceleration
}  // namespace tflite
