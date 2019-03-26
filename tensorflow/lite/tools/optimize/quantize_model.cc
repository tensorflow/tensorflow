/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/model_utils.h"
#include "tensorflow/lite/tools/optimize/subgraph_quantizer.h"

namespace tflite {
namespace optimize {

namespace {

// True if the tensor type has to be modified.
bool TensorTypeChangeRequired(const TensorT* tensor, const TensorType& type) {
  // The quantized model is type INT8, so if the user provided type is INT8, we
  // do not have to do any custom logic. Additionally, if the current tensor
  // isn't INT8 quantized, the custom type doesn't apply.
  return (type != TensorType_INT8 && tensor->type == TensorType_INT8 &&
          !tensor->quantization->scale.empty());
}

// Sets the input type, adding a Quantize node at the start of the model if
// necessary.
// Returns the new input tensor index.
int32_t SetInputType(ModelT* model, SubGraphT* subgraph,
                     const int32_t tensor_idx, const TensorType& input_type) {
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  if (!TensorTypeChangeRequired(tensor, input_type)) {
    return -1;
  }
  if (input_type == TensorType_FLOAT32) {
    // Create a new tensor to be the input of the quantize op.
    std::unique_ptr<TensorT> quantize_input;
    const string quant_name = tensor->name + "_quantize";
    utils::MakeTensor(quant_name, tensor->shape, TensorType_FLOAT32,
                      &quantize_input);
    const int32_t quantize_input_idx = subgraph->tensors.size();
    subgraph->tensors.push_back(std::move(quantize_input));

    // Create the Dequantize operation.
    std::unique_ptr<OperatorT> quantize_op;
    utils::MakeQuantizeOperator(model, &quantize_op, quantize_input_idx,
                                tensor_idx);

    // Insert the new op at the start of the model.
    subgraph->operators.insert(subgraph->operators.begin(),
                               std::move(quantize_op));
    return quantize_input_idx;
  }
  return -1;
}

// Sets the output type, adding a Dequantize node at the end of the model if
// necessary.
// Returns the new output tensor index.
int32_t SetOutputType(ModelT* model, SubGraphT* subgraph,
                      const int32_t tensor_idx, const TensorType& output_type) {
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  if (!TensorTypeChangeRequired(tensor, output_type)) {
    return -1;
  }
  if (output_type == TensorType_FLOAT32) {
    // Create a new tensor to be the output of the dequantize op.
    std::unique_ptr<TensorT> dequantize_output;
    const string dequant_name = tensor->name + "_dequantize";
    utils::MakeTensor(dequant_name, tensor->shape, TensorType_FLOAT32,
                      &dequantize_output);
    const int32_t dequantize_output_idx = subgraph->tensors.size();
    subgraph->tensors.push_back(std::move(dequantize_output));

    // Create the Dequantize operation.
    std::unique_ptr<OperatorT> dequantize_op;
    utils::MakeDequantizeOperator(model, &dequantize_op, tensor_idx,
                                  dequantize_output_idx);
    // Add the operator at the end of the model.
    subgraph->operators.push_back(std::move(dequantize_op));
    return dequantize_output_idx;
  }
  return -1;
}

// Sets the input and output types to the provided types. Quantize and
// Dequantize operations will be added if needed.
void SetInputAndOutputTypes(ModelT* model, SubGraphT* subgraph,
                            const TensorType& input_type,
                            const TensorType& output_type) {
  for (int i = 0; i < subgraph->inputs.size(); ++i) {
    const int32_t input_idx =
        SetInputType(model, subgraph, subgraph->inputs[i], input_type);
    if (input_idx < 0) {
      continue;
    }
    subgraph->inputs[i] = input_idx;
  }
  for (int i = 0; i < subgraph->outputs.size(); ++i) {
    const int32_t output_idx =
        SetOutputType(model, subgraph, subgraph->outputs[i], output_type);
    if (output_idx < 0) {
      continue;
    }
    subgraph->outputs[i] = output_idx;
  }
}

}  // namespace

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type,
                           ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    internal::SubgraphQuantizer quantizer(model, subgraph, error_reporter);
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      auto status = quantizer.QuantizeOperator(op_idx);
      if (status != kTfLiteOk) {
        OperatorT* op = subgraph->operators[op_idx].get();
        const BuiltinOperator op_code =
            model->operator_codes[op->opcode_index]->builtin_code;
        error_reporter->Report(
            "Failed to quantized operator: %s in subgraph %d, node: %d",
            EnumNameBuiltinOperator(op_code), subgraph_idx, op_idx);
        return kTfLiteError;
      }
    }
    // For each subgraph, set the types of quantize inputs and outputs to the
    // user defined ones.
    // TODO(suharshs,jianlijianli): Add support for user provided uint8 input
    // and output types. This requires a Requantization op.
    if ((input_type != TensorType_FLOAT32 && input_type != TensorType_INT8) ||
        (output_type != TensorType_FLOAT32 && output_type != TensorType_INT8)) {
      error_reporter->Report("Provided input and output type not supported");
      return kTfLiteError;
    }
    SetInputAndOutputTypes(model, subgraph, input_type, output_type);
  }

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model);
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, TensorType_FLOAT32, TensorType_FLOAT32,
                       error_reporter);
}

}  // namespace optimize
}  // namespace tflite
