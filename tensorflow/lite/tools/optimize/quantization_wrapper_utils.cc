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
#include "tensorflow/lite/tools/optimize/quantization_wrapper_utils.h"

#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/operator_property.h"

namespace tflite {
namespace impl {
class FlatBufferModel;
}
namespace optimize {
namespace {

#ifdef TFLITE_CUSTOM_LSTM
constexpr bool kUseCustomLSTM = true;
#else
constexpr bool kUseCustomLSTM = false;
#endif

void MakeTensor(const string& name, std::unique_ptr<TensorT>* tensor) {
  TensorT* tensor_raw = new TensorT;
  tensor_raw->name = name;
  tensor_raw->shape = {0};
  tensor_raw->type = TensorType_FLOAT32;

  tensor->reset(tensor_raw);
}

string CreateTensorName(int op_index, int tensor_index) {
  return "intermediate_" + std::to_string(op_index) + "_" +
         std::to_string(tensor_index);
}

bool IntermediateTensorExists(ModelT* model) {
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       ++subgraph_idx) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      if (!op->intermediates.empty()) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

TfLiteStatus LoadModel(const string& path, ModelT* model) {
  auto input_model = impl::FlatBufferModel::BuildFromFile(path.c_str());
  if (!input_model) {
    return kTfLiteError;
  }
  auto readonly_model = input_model->GetModel();
  if (!readonly_model) {
    return kTfLiteError;
  }
  readonly_model->UnPackTo(model);
  return kTfLiteOk;
}

TfLiteStatus AddIntermediateTensorsToFusedOp(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model) {
  // Return early when the model has no operator.
  if (model->subgraphs.size() == 1 && model->subgraphs[0]->operators.empty()) {
    return kTfLiteOk;
  }
  // Return early if the model already has intermediate tensors.
  if (IntermediateTensorExists(model)) {
    return kTfLiteOk;
  }
  // Process the model.
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       ++subgraph_idx) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      // Find ops that need additional tensor.
      OperatorT* op = subgraph->operators[op_idx].get();
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      if (property.intermediates.empty()) {
        continue;
      }
      // Add tensors.
      const int next_tensor_index = subgraph->tensors.size();
      int num_intermediates = property.intermediates.size();
      if (kUseCustomLSTM) {
        num_intermediates = 12;
      }
      for (int i = 0; i < num_intermediates; ++i) {
        std::unique_ptr<TensorT> intermediate_tensor;
        auto name = CreateTensorName(op_idx, i);
        MakeTensor(name, &intermediate_tensor);
        subgraph->tensors.push_back(std::move(intermediate_tensor));
        op->intermediates.push_back(next_tensor_index + i);
      }
    }
  }

  // Export the model.
  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model);
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

bool WriteFile(const std::string& out_file, const uint8_t* bytes,
               size_t num_bytes) {
  std::fstream stream(out_file, std::ios::binary | std::ios::out);
  for (size_t i = 0; i < num_bytes; i++) {
    stream << bytes[i];
  }
  return (!stream.bad() && !stream.fail());
}

}  // namespace optimize
}  // namespace tflite
