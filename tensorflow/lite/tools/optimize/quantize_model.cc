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

// Sets the input type, adding a Leading Op node at the start of the model if
// necessary.
// Returns the new input tensor index.
int32_t SetInputType(ModelT* model, SubGraphT* subgraph,
                     const int32_t tensor_idx, const TensorType& input_type) {
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  if (!TensorTypeChangeRequired(tensor, input_type)) {
    return -1;
  }
  if (input_type == TensorType_FLOAT32 || input_type == TensorType_UINT8) {
    // Create a new tensor to be the input of the leading Op.
    std::unique_ptr<TensorT> leading_op_input;
    if (input_type == TensorType_FLOAT32) {
      // Add tensor for quantize operator. Scales and zero points are not
      // needed.
      const string leading_op_name = tensor->name;
      const string new_name_original_input = tensor->name + "_int8";
      tensor->name = new_name_original_input;
      utils::MakeTensor(leading_op_name, tensor->shape, input_type,
                        &leading_op_input);
    } else {
      // Get scale and zero point from the first tensor.
      const float scale = subgraph->tensors[tensor_idx]->quantization->scale[0];
      const int64_t zero_point =
          subgraph->tensors[tensor_idx]->quantization->zero_point[0];

      //  Add tensor for requantize operator. Scale is the existing scale and
      //  zero point is shifted by +128.
      TFLITE_DCHECK_GE(zero_point, -128);
      TFLITE_DCHECK_LE(zero_point, 127);
      const string leading_op_name = tensor->name;
      const string new_name_original_input = tensor->name + "_int8";
      tensor->name = new_name_original_input;
      utils::MakeTensorWithQuantParam(leading_op_name, tensor->shape,
                                      input_type, scale, zero_point + 128,
                                      &leading_op_input);
    }
    const int32_t leading_op_input_idx = subgraph->tensors.size();
    subgraph->tensors.push_back(std::move(leading_op_input));

    // Create the leading op, which is Quantize Op that quantize or requantize
    // the input.
    std::unique_ptr<OperatorT> leading_op;
    utils::MakeQuantizeOperator(model, &leading_op, leading_op_input_idx,
                                tensor_idx);

    // Insert the new op at the start of the model.
    subgraph->operators.insert(subgraph->operators.begin(),
                               std::move(leading_op));
    return leading_op_input_idx;
  }
  return -1;
}

// Sets the output type, adding a Tailing Op node at the end of the model if
// necessary.
// Returns the new output tensor index.
int32_t SetOutputType(ModelT* model, SubGraphT* subgraph,
                      const int32_t tensor_idx, const TensorType& output_type) {
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  if (!TensorTypeChangeRequired(tensor, output_type)) {
    return -1;
  }
  if (output_type == TensorType_FLOAT32 || output_type == TensorType_UINT8) {
    // Create a new tensor to be the output of the tailing op.
    std::unique_ptr<TensorT> tailing_op_output;
    if (output_type == TensorType_FLOAT32) {
      const string tailing_op_name = tensor->name;
      const string new_name_original_output = tensor->name + "_int8";
      tensor->name = new_name_original_output;
      utils::MakeTensor(tailing_op_name, tensor->shape, output_type,
                        &tailing_op_output);
    } else {
      // Get scale and zero point from the last tensor.
      const float scale = subgraph->tensors[tensor_idx]->quantization->scale[0];
      const int64_t zero_point =
          subgraph->tensors[tensor_idx]->quantization->zero_point[0];

      //  Add tensor for requantize operator. Scale is the existing scale and
      //  zero point is shifted by +128.
      TFLITE_DCHECK_GE(zero_point, -128);
      TFLITE_DCHECK_LE(zero_point, 127);
      const string tailing_op_name = tensor->name;
      const string new_name_original_output = tensor->name + "_int8";
      tensor->name = new_name_original_output;
      utils::MakeTensorWithQuantParam(tailing_op_name, tensor->shape,
                                      output_type, scale, zero_point + 128,
                                      &tailing_op_output);
    }
    const int32_t tailing_op_output_idx = subgraph->tensors.size();
    subgraph->tensors.push_back(std::move(tailing_op_output));

    // Create the tailing operation.
    std::unique_ptr<OperatorT> tailing_op;
    if (output_type == TensorType_FLOAT32) {
      // Tailing Op is Dequantize Op.
      utils::MakeDequantizeOperator(model, &tailing_op, tensor_idx,
                                    tailing_op_output_idx);
    } else {
      // Tailing Op is Quantize Op that does requantization.
      utils::MakeQuantizeOperator(model, &tailing_op, tensor_idx,
                                  tailing_op_output_idx);
    }
    // Add the operator at the end of the model.
    subgraph->operators.push_back(std::move(tailing_op));
    return tailing_op_output_idx;
  }
  return -1;
}

// Sets the input and output types to the provided types. Leading and
// tailing operations will be added if needed.
// For Float input and output, leading op is Quantize and tailing op is
// Dequantize.
// For Uint8 input and output, leading op is Quantize (uint8 to
// int8, can be thought as "requant") and tailing op is also Quantize (int8 to
// uint8, can be thought as "requant").
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

// Insert requant op for just concat operator. To improve accuracy, we have made
// the restriction that for int8 quantized concat, the inputs and outpus must
// have the same scale and zero point.
TfLiteStatus ResolveConflictsForConcat(ModelT* model, SubGraphT* subgraph,
                                       ErrorReporter* error_reporter) {
  // Iterate backward to make sure the insertion is not messing with
  // iteration.
  for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
    OperatorT* op = subgraph->operators[op_idx].get();
    const BuiltinOperator op_code =
        model->operator_codes[op->opcode_index]->builtin_code;
    if (op_code != BuiltinOperator_CONCATENATION) {
      continue;
    }
    // If an op is concat and requant is needed, use the min of min and max of
    // max, which means using the scale and zero point of output.
    TensorT* output_tensor = subgraph->tensors[op->outputs[0]].get();
    if (!utils::QuantizationParametersExist(output_tensor)) {
      error_reporter->Report(
          "Unable to get scale or zero point from the tensor at %d, which "
          "is the output tensor for concat.",
          op->outputs[0]);
      return kTfLiteError;
    }
    const float output_scale = output_tensor->quantization->scale[0];
    const float output_zp = output_tensor->quantization->zero_point[0];
    for (size_t input_idx = 0; input_idx < op->inputs.size(); ++input_idx) {
      TensorT* input_tensor = subgraph->tensors[op->inputs[input_idx]].get();
      if (!utils::QuantizationParametersExist(input_tensor)) {
        error_reporter->Report(
            "Unable to get scale or zero point from tensor at %d, which is "
            "an input tensor of concat.",
            op->inputs[input_idx]);
        return kTfLiteError;
      }
      if (input_tensor->quantization->scale[0] == output_scale &&
          input_tensor->quantization->zero_point[0] == output_zp) {
        // This input does not need to be requantized.
        continue;
      }

      std::unique_ptr<TensorT> additional_tensor;
      const string requant_tensor_name = input_tensor->name + "_requantized";
      utils::MakeTensorWithQuantParam(requant_tensor_name, input_tensor->shape,
                                      TensorType_INT8, output_scale, output_zp,
                                      &additional_tensor);
      const int32_t additional_tensor_idx = subgraph->tensors.size();
      subgraph->tensors.push_back(std::move(additional_tensor));

      // Add requant op before this input.
      std::unique_ptr<OperatorT> requant_op;
      utils::MakeQuantizeOperator(model, &requant_op, op->inputs[input_idx],
                                  additional_tensor_idx);
      op->inputs[input_idx] = additional_tensor_idx;

      subgraph->operators.insert(subgraph->operators.begin() + op_idx,
                                 std::move(requant_op));
    }
  }
  return kTfLiteOk;
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

    // Resolve conflicts for concat.
    ResolveConflictsForConcat(model, subgraph, error_reporter);

    // For each subgraph, set the types of quantize inputs and outputs to the
    // user defined ones.
    if ((input_type != TensorType_FLOAT32 && input_type != TensorType_INT8 &&
         input_type != TensorType_UINT8) ||
        (output_type != TensorType_FLOAT32 && output_type != TensorType_INT8 &&
         input_type != TensorType_UINT8)) {
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
