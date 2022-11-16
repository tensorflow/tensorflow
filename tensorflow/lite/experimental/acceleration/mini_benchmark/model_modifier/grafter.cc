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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/grafter.h"

#include <stdint.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/idl.h"  // from @flatbuffers
#include "flatbuffers/reflection.h"  // from @flatbuffers
#include "flatbuffers/reflection_generated.h"  // from @flatbuffers
#include "tensorflow/lite/schema/reflection/schema_generated.h"

namespace fb = flatbuffers;

namespace tflite {
namespace acceleration {

namespace {

class Combiner : FlatbufferHelper {
 public:
  Combiner(flatbuffers::FlatBufferBuilder* fbb,
           std::vector<const Model*> models,
           std::vector<std::string> subgraph_names,
           const reflection::Schema* schema)
      : FlatbufferHelper(fbb, schema),
        fbb_(fbb),
        models_(models),
        subgraph_names_(subgraph_names),
        schema_(schema) {}
  absl::Status Combine() {
    auto operator_codes = OperatorCodes();
    if (!operator_codes.ok()) {
      return operator_codes.status();
    }
    auto subgraphs = SubGraphs();
    if (!subgraphs.ok()) {
      return subgraphs.status();
    }
    auto buffers = Buffers();
    if (!buffers.ok()) {
      return buffers.status();
    }
    auto metadata = Metadatas();
    if (!metadata.ok()) {
      return metadata.status();
    }
    auto signature_defs = SignatureDefs();
    if (!signature_defs.ok()) {
      return signature_defs.status();
    }
    fb::Offset<Model> model = CreateModel(
        *fbb_, 3, *operator_codes, *subgraphs,
        fbb_->CreateString(models_[0]->description()->str()), *buffers,
        /* metadata_buffer */ 0, *metadata, *signature_defs);
    fbb_->Finish(model, "TFL3");
    return absl::OkStatus();
  }

 private:
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<OperatorCode>>>>
  OperatorCodes() {
    std::vector<fb::Offset<OperatorCode>> codes;
    for (const Model* model : models_) {
      for (int i = 0; i < model->operator_codes()->size(); i++) {
        auto status = CopyTableToVector(
            "tflite.OperatorCode", model->operator_codes()->Get(i), &codes);
        if (!status.ok()) {
          return status;
        }
      }
    }
    return fbb_->CreateVector(codes);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<SubGraph>>>> SubGraphs() {
    std::vector<fb::Offset<SubGraph>> graphs;
    int buffer_offset = 0;
    int operator_code_offset = 0;
    int subgraph_index = 0;
    for (const Model* model : models_) {
      if (model->subgraphs()->size() != 1) {
        return absl::InvalidArgumentError(
            "Every model to be combined must have a single subgraph.");
      }
      auto graph =
          AdjustSubGraph(model->subgraphs()->Get(0), buffer_offset,
                         operator_code_offset, subgraph_names_[subgraph_index]);
      if (!graph.ok()) {
        return graph.status();
      }
      graphs.push_back(*graph);
      buffer_offset += model->buffers()->size();
      operator_code_offset += model->operator_codes()->size();
      ++subgraph_index;
    }
    return fbb_->CreateVector(graphs);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<Buffer>>>> Buffers() {
    std::vector<fb::Offset<Buffer>> buffers;
    for (const Model* model : models_) {
      for (int i = 0; i < model->buffers()->size(); i++) {
        auto status = CopyTableToVector("tflite.Buffer",
                                        model->buffers()->Get(i), &buffers);
        if (!status.ok()) {
          return status;
        }
      }
    }
    return fbb_->CreateVector(buffers);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<Metadata>>>> Metadatas() {
    std::vector<fb::Offset<Metadata>> metadatas;
    int buffer_offset = 0;
    for (const Model* model : models_) {
      for (int i = 0; model->metadata() && i < model->metadata()->size(); i++) {
        auto metadata =
            AdjustMetadata(model->metadata()->Get(i), buffer_offset);
        if (!metadata.ok()) {
          return metadata.status();
        }
        metadatas.push_back(*metadata);
        buffer_offset += model->buffers()->size();
      }
    }
    return fbb_->CreateVector(metadatas);
  }
  absl::StatusOr<fb::Offset<fb::Vector<fb::Offset<SignatureDef>>>>
  SignatureDefs() {
    std::vector<fb::Offset<SignatureDef>> signature_defs;
    const Model* model = models_[0];
    for (int i = 0;
         model->signature_defs() && i < model->signature_defs()->size(); i++) {
      auto status =
          CopyTableToVector("tflite.SignatureDef",
                            model->signature_defs()->Get(i), &signature_defs);
      if (!status.ok()) {
        return status;
      }
    }
    return fbb_->CreateVector(signature_defs);
  }

  absl::StatusOr<fb::Offset<SubGraph>> AdjustSubGraph(const SubGraph* graph,
                                                      int buffer_offset,
                                                      int operator_code_offset,
                                                      const std::string& name) {
    auto tensors = AdjustTensors(graph, buffer_offset);
    if (!tensors.ok()) {
      return tensors.status();
    }
    auto ops = AdjustOps(graph, operator_code_offset);
    if (!ops.ok()) {
      return ops.status();
    }
    return CreateSubGraph(*fbb_, fbb_->CreateVector(*tensors),
                          CopyIntVector(graph->inputs()),
                          CopyIntVector(graph->outputs()),
                          fbb_->CreateVector(*ops), fbb_->CreateString(name));
  }

  absl::StatusOr<std::vector<fb::Offset<Operator>>> AdjustOps(
      const SubGraph* graph, int operator_code_offset) {
    std::vector<fb::Offset<Operator>> ops;
    auto op_object = FindObject("tflite.Operator");
    const reflection::Field* builtin_options_field = nullptr;
    for (auto it = op_object->fields()->cbegin();
         it != op_object->fields()->cend(); it++) {
      auto candidate = *it;
      if (candidate->name()->str() == "builtin_options") {
        builtin_options_field = candidate;
        break;
      }
    }
    if (!builtin_options_field) {
      return absl::UnknownError(
          "Wasn't able to find the builtin_options field on tflite.Operator");
    }
    for (int i = 0; i < graph->operators()->size(); i++) {
      const Operator* op = graph->operators()->Get(i);
      fb::Offset<void> copied_builtin_options = 0;
      if (op->builtin_options() != nullptr) {
        const fb::Table* opt = (const fb::Table*)op;  // NOLINT
        auto& builtin_options_object = fb::GetUnionType(
            *schema_, *op_object, *builtin_options_field, *opt);
        copied_builtin_options =
            fb::CopyTable(*fbb_, *schema_, builtin_options_object,
                          *fb::GetFieldT(*opt, *builtin_options_field))
                .o;
      }
      ops.push_back(CreateOperator(
          *fbb_, op->opcode_index() + operator_code_offset,
          CopyIntVector(op->inputs()), CopyIntVector(op->outputs()),
          op->builtin_options_type(), copied_builtin_options,
          CopyIntVector(op->custom_options()), op->custom_options_format(),
          CopyIntVector(op->mutating_variable_inputs()),
          CopyIntVector(op->intermediates())));
    }
    return ops;
  }

  absl::StatusOr<std::vector<fb::Offset<Tensor>>> AdjustTensors(
      const SubGraph* graph, int buffer_offset) {
    std::vector<fb::Offset<Tensor>> tensors;
    auto orig_tensors = graph->tensors();
    for (auto iter = orig_tensors->cbegin(); iter != orig_tensors->cend();
         iter++) {
      auto i = *iter;
      std::vector<int32_t> shape{i->shape()->cbegin(), i->shape()->cend()};
      std::vector<int32_t> shape_signature;
      if (i->shape_signature()) {
        shape_signature.assign(i->shape_signature()->cbegin(),
                               i->shape_signature()->cend());
      }
      auto quantization =
          CopyTable("tflite.QuantizationParameters", i->quantization());
      if (!quantization.ok()) {
        return quantization.status();
      }
      auto sparsity = CopyTable("tflite.SparsityParameters", i->sparsity());
      if (!sparsity.ok()) {
        return sparsity.status();
      }
      tensors.push_back(CreateTensor(
          *fbb_, fbb_->CreateVector(shape), i->type(),
          i->buffer() + buffer_offset, fbb_->CreateString(i->name()->str()),
          *quantization, i->is_variable(), *sparsity,
          shape_signature.empty() ? 0 : fbb_->CreateVector(shape_signature)));
    }
    return tensors;
  }

  absl::StatusOr<fb::Offset<Metadata>> AdjustMetadata(const Metadata* metadata,
                                                      int buffer_offset) {
    return CreateMetadata(*fbb_,
                          metadata->name()
                              ? fbb_->CreateString(metadata->name()->str())
                              : 0,
                          metadata->buffer())
        .o;
  }

  flatbuffers::FlatBufferBuilder* fbb_;
  std::vector<const Model*> models_;
  std::vector<std::string> subgraph_names_;
  const reflection::Schema* schema_;
};

}  // namespace

absl::Status CombineModels(flatbuffers::FlatBufferBuilder* fbb,
                           std::vector<const Model*> models,
                           std::vector<std::string> subgraph_names,
                           const reflection::Schema* schema) {
  if (!fbb || !schema) {
    return absl::InvalidArgumentError(
        "Must provide FlatBufferBuilder and Schema");
  }
  if (models.size() < 2) {
    return absl::InvalidArgumentError("Must have 2+ models to combine");
  }
  Combiner combiner(fbb, models, subgraph_names, schema);
  return combiner.Combine();
}

FlatbufferHelper::FlatbufferHelper(flatbuffers::FlatBufferBuilder* fbb,
                                   const reflection::Schema* schema)
    : fbb_(fbb), schema_(schema) {}

const reflection::Object* FlatbufferHelper::FindObject(
    const std::string& name) {
  for (auto candidate = schema_->objects()->cbegin();
       candidate != schema_->objects()->cend(); candidate++) {
    if (candidate->name()->str() == name) {
      return *candidate;
    }
  }
  return nullptr;
}

}  // namespace acceleration
}  // namespace tflite
