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
#include <string>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/tools/verifier.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/constants.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace acceleration {
namespace {

using ::flatbuffers::FlatBufferBuilder;

// Create options for call operator.
flatbuffers::Offset<flatbuffers::Vector<uint8_t>> CallOpCustomOptions(
    int primary_subgraph_index, int batch_size, FlatBufferBuilder& output) {
  flexbuffers::Builder flexbuffer_builder;
  flexbuffer_builder.Map([&] {
    flexbuffer_builder.Int("subgraph_index", primary_subgraph_index);
    flexbuffer_builder.Int("loop_count", batch_size);
  });
  flexbuffer_builder.Finish();
  return output.CreateVector(flexbuffer_builder.GetBuffer());
}

}  // namespace

void CustomValidationEmbedder::CreateTensorsFrom(
    const SubGraph& from_subgraph, const std::vector<int>& from_indexes,
    std::vector<std::vector<uint8_t>>* buffer_content,
    flatbuffers::FlatBufferBuilder& fbb, std::vector<int>& new_indexes,
    std::vector<flatbuffers::Offset<Buffer>>& buffers,
    std::vector<flatbuffers::Offset<Tensor>>& tensors) {
  int tensor_index_start = tensors.size();
  for (int i = 0; i < from_indexes.size(); i++) {
    TensorT base_tensor;
    from_subgraph.tensors()->Get(from_indexes[i])->UnPackTo(&base_tensor);
    if (!base_tensor.shape.empty() && base_tensor.shape[0] == 1) {
      base_tensor.shape[0] = batch_size_;
    }
    if (!base_tensor.shape_signature.empty() &&
        base_tensor.shape_signature[0] == 1) {
      base_tensor.shape_signature[0] = batch_size_;
    }
    // Set the index in buffer.
    base_tensor.buffer = buffers.size();

    tensors.push_back(CreateTensor(fbb, &base_tensor));
    new_indexes.push_back(tensor_index_start + i);
    // If buffer content is provided, embed the content. Otherwise create an
    // empty buffer.
    if (buffer_content && !(*buffer_content)[i].empty()) {
      buffers.push_back(
          CreateBuffer(fbb, fbb.CreateVector((*buffer_content)[i])));
    } else {
      buffers.push_back(CreateBuffer(fbb));
    }
  }
}

MinibenchmarkStatus CustomValidationEmbedder::BuildModel(
    const Model& main_model, flatbuffers::FlatBufferBuilder& fbb) {
  ModelT main_model_obj;
  main_model.UnPackTo(&main_model_obj);
  if (main_model_obj.subgraphs[0]->inputs.size() != custom_input_.size()) {
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Unexpected custom_input size. Expected: %d. Actual: %d.",
        main_model_obj.subgraphs[0]->inputs.size(), custom_input_.size());
    return kMinibenchmarkValidationSubgraphBuildFailed;
  }

  // Copy all the data from main_model.
  std::vector<flatbuffers::Offset<Metadata>> metadata;
  metadata.reserve(main_model_obj.metadata.size());
  for (auto& iter : main_model_obj.metadata) {
    metadata.push_back(CreateMetadata(fbb, iter.get()));
  }

  std::vector<flatbuffers::Offset<SignatureDef>> signature_defs;
  signature_defs.reserve(main_model_obj.signature_defs.size());
  for (auto& iter : main_model_obj.signature_defs) {
    signature_defs.push_back(CreateSignatureDef(fbb, iter.get()));
  }

  std::vector<flatbuffers::Offset<SubGraph>> subgraphs;
  subgraphs.reserve(main_model_obj.subgraphs.size());
  for (auto& iter : main_model_obj.subgraphs) {
    subgraphs.push_back(CreateSubGraph(fbb, iter.get()));
  }

  std::vector<flatbuffers::Offset<Buffer>> buffers;
  buffers.reserve(main_model_obj.buffers.size());
  for (auto& iter : main_model_obj.buffers) {
    buffers.push_back(CreateBuffer(fbb, iter.get()));
  }

  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes;
  operator_codes.reserve(main_model_obj.operator_codes.size());
  for (auto& iter : main_model_obj.operator_codes) {
    operator_codes.push_back(CreateOperatorCode(fbb, iter.get()));
  }

  // Create validation subgraph.
  operator_codes.push_back(CreateOperatorCode(
      fbb, BuiltinOperator_CUSTOM, fbb.CreateString("validation/call")));
  int operator_code_index = operator_codes.size() - 1;

  // Input and output tensors.
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  std::vector<int32_t> input;
  CreateTensorsFrom(*main_model.subgraphs()->Get(0),
                    main_model_obj.subgraphs[0]->inputs, &custom_input_, fbb,
                    input, buffers, tensors);

  std::vector<int32_t> output;
  CreateTensorsFrom(*main_model.subgraphs()->Get(0),
                    main_model_obj.subgraphs[0]->outputs, nullptr, fbb, output,
                    buffers, tensors);
  auto input_offset = fbb.CreateVector(input);
  auto output_offset = fbb.CreateVector(output);
  std::vector<flatbuffers::Offset<Operator>> operators{CreateOperator(
      fbb, operator_code_index, input_offset, output_offset,
      tflite::BuiltinOptions_NONE, 0,
      CallOpCustomOptions(/*primary_graph_index*/ 0, batch_size_, fbb),
      tflite::CustomOptionsFormat_FLEXBUFFERS)};
  subgraphs.push_back(
      CreateSubGraph(fbb, fbb.CreateVector(tensors), input_offset,
                     output_offset, fbb.CreateVector(operators),
                     fbb.CreateString(std::string(kValidationGraphName))));

  fbb.Finish(
      CreateModel(fbb, kModelSchemaVersion, fbb.CreateVector(operator_codes),
                  fbb.CreateVector(subgraphs),
                  fbb.CreateString(main_model_obj.description),
                  fbb.CreateVector(buffers),
                  /* metadata_buffer */ 0, fbb.CreateVector(metadata),
                  fbb.CreateVector(signature_defs)),
      "TFL3");

  if (Verify(fbb.GetBufferPointer(), fbb.GetSize(), error_reporter_)) {
    return kMinibenchmarkSuccess;
  } else {
    return kMinibenchmarkValidationSubgraphBuildFailed;
  }
}

}  // namespace acceleration
}  // namespace tflite
