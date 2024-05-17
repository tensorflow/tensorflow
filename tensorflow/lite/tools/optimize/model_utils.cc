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
#include "tensorflow/lite/tools/optimize/model_utils.h"

#include <fstream>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/optimize/operator_property.h"

namespace tflite {
namespace optimize {
namespace utils {

namespace {

// Returns the index of the OpCode.
// If a OpCode doesn't exist, adds it and returns its index.
int32_t GetOrInsertOpCodeIndex(ModelT* model, const BuiltinOperator& op_code,
                               int32_t version) {
  for (size_t i = 0; i < model->operator_codes.size(); ++i) {
    if (GetBuiltinCode(model->operator_codes[i].get()) == op_code) {
      return i;
    }
  }
  model->operator_codes.push_back(std::make_unique<OperatorCodeT>());
  int op_code_idx = model->operator_codes.size() - 1;
  model->operator_codes[op_code_idx]->builtin_code = op_code;
  model->operator_codes[op_code_idx]->deprecated_builtin_code =
      ConvertBuiltinCodeToDeprecatedBuiltinCode(op_code);
  // Version 2 and onwards supports INT8 inputs.
  model->operator_codes[op_code_idx]->version = version;

  // Return the index of the newly placed OperatorCodeT.
  return op_code_idx;
}

}  // namespace

// Creates a Dequantize OperatorT object.
void MakeDequantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                            int32_t input, int32_t output) {
  OperatorT* op_raw = new OperatorT;
  // Version 2 and onwards supports INT8 inputs.
  op_raw->opcode_index =
      GetOrInsertOpCodeIndex(model, BuiltinOperator_DEQUANTIZE, 2);
  op_raw->inputs = {input};
  op_raw->outputs = {output};

  op->reset(op_raw);
}

// Creates a Quantize OperatorT object.
void MakeQuantizeOperator(ModelT* model, std::unique_ptr<OperatorT>* op,
                          int32_t input, int32_t output) {
  OperatorT* op_raw = new OperatorT;
  op_raw->opcode_index =
      GetOrInsertOpCodeIndex(model, BuiltinOperator_QUANTIZE, 1);
  op_raw->inputs = {input};
  op_raw->outputs = {output};

  op->reset(op_raw);
}

// Create a new TensorT object without quantization parameters.
void MakeTensor(const string& name, const std::vector<int32_t>& shape,
                const std::vector<int32_t>& shape_signature,
                const TensorType& type, std::unique_ptr<TensorT>* tensor) {
  TensorT* tensor_raw = new TensorT;
  tensor_raw->name = name;
  tensor_raw->shape = shape;
  if (!shape_signature.empty()) {
    tensor_raw->shape_signature = shape_signature;
  }
  tensor_raw->type = type;

  tensor->reset(tensor_raw);
}

// Create a new TensorT object with quantization parameters.
void MakeTensorWithQuantParam(const string& name,
                              const std::vector<int32_t>& shape,
                              const std::vector<int32_t>& shape_signature,
                              const TensorType& type, float scale,
                              int64_t zero_point,
                              std::unique_ptr<TensorT>* tensor) {
  MakeTensor(name, shape, shape_signature, type, tensor);
  (*tensor)->quantization = std::make_unique<QuantizationParametersT>();
  (*tensor)->quantization->scale.push_back(scale);
  (*tensor)->quantization->zero_point.push_back(zero_point);
}

bool QuantizationParametersExist(const TensorT* tensor) {
  return tensor->quantization != nullptr &&
         !tensor->quantization->scale.empty() &&
         !tensor->quantization->zero_point.empty();
}

bool HasBuffer(const ModelT* model, const SubGraphT* subgraph,
               int tensor_index) {
  const int buffer_index = subgraph->tensors[tensor_index]->buffer;
  BufferT* buffer = model->buffers[buffer_index].get();
  if (buffer == nullptr || buffer->data.empty()) {
    return false;
  }
  return true;
}

bool HasMinMax(const TensorT* tensor) {
  return tensor->quantization && !tensor->quantization->min.empty() &&
         !tensor->quantization->max.empty();
}

void SetOperatorCodeVersion(ModelT* model) {
  for (int subgraph_idx = 0, end = model->subgraphs.size(); subgraph_idx < end;
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Iterate backward to avoid messing with index.
    for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      if (property.quantizable && op_code->version < property.version) {
        // Only update the versions of quantizable operations if the original
        // version is lesser than minimum quantized one mentioned by
        // OperatorProperty.
        op_code->version = property.version;
      }
    }
  }
}

void WriteFile(const std::string& out_file, const uint8_t* bytes,
               size_t num_bytes) {
  std::fstream stream(out_file, std::ios::binary | std::ios::out);
  for (size_t i = 0; i < num_bytes; i++) {
    stream << bytes[i];
  }
  TFLITE_DCHECK(!stream.bad() && !stream.fail());
}

std::unique_ptr<flatbuffers::FlatBufferBuilder> FinishModel(
    const tflite::ModelT* model) {
  std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(
      new flatbuffers::FlatBufferBuilder());
  auto packed_model = tflite::Model::Pack(*builder, model);
  tflite::FinishModelBuffer(*builder, packed_model);
  return builder;
}

std::unique_ptr<tflite::ModelT> CreateMutableModelFromFile(
    const string& model_filepath) {
  auto fb_model =
      tflite::FlatBufferModel::BuildFromFile(model_filepath.c_str());
  auto tflite_model = fb_model->GetModel();
  auto copied_model = std::make_unique<tflite::ModelT>();
  tflite_model->UnPackTo(copied_model.get(), nullptr);
  return copied_model;
}

}  // namespace utils
}  // namespace optimize
}  // namespace tflite
