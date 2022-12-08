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
#include "tensorflow/lite/tools/optimize/quantize_model.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/tools/optimize/model_utils.h"
#include "tensorflow/lite/tools/optimize/operator_property.h"
#include "tensorflow/lite/tools/optimize/quantization_utils.h"

namespace tflite {
namespace optimize {

namespace {

// Bias tensors must be duplicated if it is used as a non-bias input in another
// op (quantized to 8 bit), in order to quantize to 32 bit.
TfLiteStatus DuplicateBiasesWithMultipleUses(ModelT* model,
                                             ErrorReporter* error_reporter) {
  std::set<int> input_uses;
  // Get all input uses for constant tensors.
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      auto* op = subgraph->operators[op_idx].get();
      for (const auto& idx_pair : property.inputs) {
        const int idx = idx_pair.first;
        if (op->inputs[idx] < 0 || idx >= op->inputs.size()) {
          continue;
        }
        const TensorT* input_tensor = subgraph->tensors[op->inputs[idx]].get();
        if (!input_tensor || (input_tensor->buffer < 0) ||
            (input_tensor->buffer >= model->buffers.size())) {
          continue;
        }
        const BufferT* buffer = model->buffers[input_tensor->buffer].get();
        if (buffer && !buffer->data.empty()) {
          input_uses.insert({op->inputs[idx]});
        }
      }
    }
  }

  std::map<int, int> bias_uses;
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      OperatorT* op = subgraph->operators[op_idx].get();
      for (const int bias_idx : property.biases) {
        if (bias_idx >= op->inputs.size() || op->inputs[bias_idx] < 0) {
          continue;
        }
        const TensorT* bias_tensor =
            subgraph->tensors[op->inputs[bias_idx]].get();
        if (!bias_tensor || (bias_tensor->buffer < 0) ||
            (bias_tensor->buffer >= model->buffers.size())) {
          continue;
        }
        const BufferT* bias_buffer = model->buffers[bias_tensor->buffer].get();
        if (!bias_buffer || bias_buffer->data.empty()) {
          continue;
        }
        if (input_uses.find(op->inputs[bias_idx]) != input_uses.end()) {
          // If used as input, duplicate the tensor and insert into bias uses.
          int bias_use_count = 1;
          auto inserted =
              bias_uses.insert({op->inputs[bias_idx], bias_use_count});
          if (!inserted.second) {
            bias_use_count = ++inserted.first->second;
          }
          std::unique_ptr<TensorT> new_tensor(new TensorT);
          new_tensor->name =
              absl::StrCat(bias_tensor->name, "_duplicate_", bias_use_count);
          new_tensor->shape = bias_tensor->shape;
          new_tensor->type = bias_tensor->type;
          if (bias_tensor->quantization) {
            new_tensor->quantization =
                std::make_unique<QuantizationParametersT>();
            new_tensor->quantization->scale.assign(
                bias_tensor->quantization->scale.begin(),
                bias_tensor->quantization->scale.end());
            new_tensor->quantization->zero_point.assign(
                bias_tensor->quantization->zero_point.begin(),
                bias_tensor->quantization->zero_point.end());
          }
          std::unique_ptr<BufferT> new_buffer(new BufferT);
          new_buffer->data.assign(bias_buffer->data.begin(),
                                  bias_buffer->data.end());
          model->buffers.push_back(std::move(new_buffer));
          new_tensor->buffer = model->buffers.size() - 1;
          subgraph->tensors.push_back(std::move(new_tensor));
          op->inputs[bias_idx] = subgraph->tensors.size() - 1;
        }
      }
    }
  }
  return kTfLiteOk;
}

bool IsFloatTensor(const SubGraphT* subgraph, int32_t tensor_idx) {
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  if (tensor->type != TensorType_FLOAT32) {
    // Skip non-real-valued tensor.
    return false;
  }
  return true;
}

// Gets the operator property from the operator_property list and additionally
// modifies the quantizable parameter based on the user's specified
// operator_names.
operator_property::OperatorProperty GetOperatorProperty(
    const std::unordered_set<string>& operator_names, const ModelT* model,
    int subgraph_index, int op_idx, const string& operator_name,
    const TensorType& activations_type, bool disable_per_channel = false) {
  operator_property::OperatorProperty property =
      operator_property::GetOperatorProperty(model, subgraph_index, op_idx);
  const SubGraphT* subgraph = model->subgraphs[subgraph_index].get();
  const OperatorT* op = subgraph->operators[op_idx].get();
  const BuiltinOperator op_code =
      GetBuiltinCode(model->operator_codes[op->opcode_index].get());
  if (activations_type == TensorType_INT16 && !property.quantizable_int16) {
    property.quantizable = false;
  }
  // The algorithm adds Dequantize and Quantize, so we don't require them to be
  // in the operator_names.
  if (op_code != BuiltinOperator_DEQUANTIZE &&
      op_code != BuiltinOperator_QUANTIZE) {
    property.quantizable =
        property.quantizable &&
        (operator_names.find(operator_name) != operator_names.end());
  }
  if (disable_per_channel) {
    for (auto& input : property.inputs) {
      if (input.second.per_axis) {
        input.second.per_axis = false;
      }
    }
  }
  return property;
}

bool IsRealValueOp(const std::unordered_set<string>& real_value_op_set,
                   const string& operator_name) {
  return real_value_op_set.find(operator_name) != real_value_op_set.end();
}

// Utility function to determine if tensor is constant and only has one use.
bool IsConstantWithOneUse(const ModelT* model, const SubGraphT* subgraph,
                          const int tensor_id) {
  if (!subgraph || (tensor_id >= subgraph->tensors.size())) {
    return false;
  }
  const auto& tensor = subgraph->tensors[tensor_id];
  if (!tensor || !model || (tensor->buffer < 0) ||
      (tensor->buffer >= model->buffers.size()) ||
      (!model->buffers[tensor->buffer]) ||
      (model->buffers[tensor->buffer]->data.empty())) {
    return false;
  }
  int uses = 0;
  for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
    const auto& op = subgraph->operators[op_idx];
    if (!op) {
      continue;
    }
    const std::vector<int32_t>& inputs = op->inputs;
    if ((std::find(inputs.begin(), inputs.end(), tensor_id) != inputs.end()) &&
        (++uses > 1)) {
      return false;
    }
  }
  return true;
}

// Creates a set that contains all quantizable ops that happen to take a
// non-float type in the source graph.
std::unordered_set<string> PopulateRealValueOpSet(
    ModelT* model, const std::unordered_set<string>& operator_names,
    const TensorType& activations_type) {
  std::unordered_set<string> real_value_op_set;
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      if (op->outputs.empty() && op_code != BuiltinOperator_ASSIGN_VARIABLE) {
        continue;
      }
      const string operator_name = op_code != BuiltinOperator_ASSIGN_VARIABLE
                                       ? subgraph->tensors[op->outputs[0]]->name
                                       : subgraph->tensors[op->inputs[0]]->name;
      operator_property::OperatorProperty property =
          GetOperatorProperty(operator_names, model, subgraph_idx, op_idx,
                              operator_name, activations_type);

      if (!property.quantizable) {
        real_value_op_set.insert(operator_name);
        continue;
      }

      for (const std::pair<int, operator_property::TensorProperty>& input :
           property.inputs) {
        const int32_t input_idx = input.first;
        const int32_t tensor_idx = op->inputs[input_idx];
        if (IsFloatTensor(subgraph, tensor_idx)) {
          real_value_op_set.insert(operator_name);
          break;
        }
      }
      for (const std::pair<int, operator_property::TensorProperty>& output :
           property.outputs) {
        const int32_t output_idx = output.first;
        const int32_t tensor_idx = op->outputs[output_idx];
        if (IsFloatTensor(subgraph, tensor_idx)) {
          real_value_op_set.insert(operator_name);
          break;
        }
      }

      if (property.arbitrary_inputs) {
        const int32_t tensor_idx = op->inputs[0];
        if (IsFloatTensor(subgraph, tensor_idx)) {
          real_value_op_set.insert(operator_name);
        }
      }

      if (property.arbitrary_outputs) {
        const int32_t tensor_idx = op->outputs[0];
        if (IsFloatTensor(subgraph, tensor_idx)) {
          real_value_op_set.insert(operator_name);
        }
      }
    }
  }
  return real_value_op_set;
}

// We set the builtin option quantized_bias_type for
// CONV_2D/FULLY_CONNECTED/TRANSPOSE_CONV, to ensure the correct
// accumulator is set even if no bias is used.
void SetOperatorPropertyBiasType(ModelT* model, const TensorType& bias_type) {
  for (int subgraph_idx = 0, end = model->subgraphs.size(); subgraph_idx < end;
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Iterate backward to avoid messing with index.
    for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
      if (op_code && op_code->builtin_code == BuiltinOperator_FULLY_CONNECTED) {
        auto* options = op->builtin_options.AsFullyConnectedOptions();
        if (options) {
          options->quantized_bias_type = bias_type;
        }
      }
      else if (op_code && op_code->builtin_code == BuiltinOperator_CONV_2D) {
        auto* options = op->builtin_options.AsConv2DOptions();
        if (options) {
          options->quantized_bias_type = bias_type;
        }
      } else if (op_code &&
                 op_code->builtin_code == BuiltinOperator_TRANSPOSE_CONV) {
        auto* options = op->builtin_options.AsTransposeConvOptions();
        if (options) {
          options->quantized_bias_type = bias_type;
        }
      }
    }
  }
}

TfLiteStatus QuantizeBias(ModelT* model, const TensorT* input_tensor,
                          const TensorT* weight_tensor, TensorT* bias_tensor,
                          bool is_per_channel, int channel_dim_index,
                          const TensorType& bias_type,
                          ErrorReporter* error_reporter) {
  if (bias_tensor->shape.size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Expected bias tensor shape to be 1.");
    return kTfLiteError;
  }

  if (input_tensor->type == tflite::TensorType_INT8 &&
      bias_type != tflite::TensorType_INT32) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Expected bias type to be TensorType_INT32 for Int8Quant.");
    return kTfLiteError;
  }

  if (input_tensor->type == tflite::TensorType_INT16 &&
      bias_type != tflite::TensorType_INT32 &&
      bias_type != tflite::TensorType_INT64) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Expected bias type to be TensorType_INT32 or "
                         "TensorType_INT64 for Int16Quant.");
    return kTfLiteError;
  }

  int32_t channel_dim_size = bias_tensor->shape[0];
  TF_LITE_ENSURE(error_reporter, weight_tensor->quantization);
  std::vector<float> weight_scales = weight_tensor->quantization->scale;

  if (is_per_channel) {
    if (bias_tensor->shape[0] != weight_tensor->shape[channel_dim_index]) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Channel mismatch between bias and weight tensors %d vs %d",
          bias_tensor->shape[0], weight_tensor->shape[channel_dim_index]);
      return kTfLiteError;
    }
    if (!input_tensor->quantization ||
        input_tensor->quantization->scale.size() != 1) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Input tensor missing quantization information");
      return kTfLiteError;
    }

    if (weight_scales.size() != channel_dim_size) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Mismatch weight scale dimension: %d",
                           weight_scales.size());
      return kTfLiteError;
    }
    if (bias_type == tflite::TensorType_INT64) {
      return utils::SymmetricPerChannelBiasQuantize<std::int64_t>(
          model, bias_tensor, input_tensor->quantization->scale[0],
          weight_scales.data(), channel_dim_size, error_reporter);
    } else {
      return utils::SymmetricPerChannelBiasQuantize<std::int32_t>(
          model, bias_tensor, input_tensor->quantization->scale[0],
          weight_scales.data(), channel_dim_size, error_reporter);
    }
  } else {
    if (weight_scales.size() != 1) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Expected per-layer weight scale dimension size 1, got %d",
          weight_scales.size());
      return kTfLiteError;
    }
    if (bias_type == tflite::TensorType_INT64) {
      return utils::SymmetricPerLayerBiasQuantize<std::int64_t>(
          model, bias_tensor,
          input_tensor->quantization->scale[0] * weight_scales[0],
          error_reporter);
    } else {
      return utils::SymmetricPerLayerBiasQuantize<std::int32_t>(
          model, bias_tensor,
          input_tensor->quantization->scale[0] * weight_scales[0],
          error_reporter);
    }
  }
  return kTfLiteError;
}

// True if the tensor type has to be modified.
bool TensorTypeChangeRequired(const TensorT* tensor, const TensorType& type) {
  // The quantized model is type INT8/INT16, so if the user provided type is
  // INT8/INT16, we do not have to do any custom logic. Additionally, if the
  // current tensor isn't INT8/INT16 quantized, the custom type doesn't apply.
  bool int8check = type != TensorType_INT8 && tensor->type == TensorType_INT8 &&
                   !tensor->quantization->scale.empty();
  bool int16check = type != TensorType_INT16 &&
                    tensor->type == TensorType_INT16 &&
                    !tensor->quantization->scale.empty();
  return (int8check || int16check);
}

// Check if input is consumed by quantize, which means we don't need to
// requantize if the output scale is the same as the input tensor's.
bool InputQuantizeRequired(const ModelT* model, const SubGraphT* subgraph,
                           int32_t input_idx) {
  std::vector<OperatorT*> quantize_ops;
  for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
    OperatorT* op = subgraph->operators[op_idx].get();
    if (std::find(op->inputs.begin(), op->inputs.end(), input_idx) !=
        op->inputs.end()) {
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      if (op_code != BuiltinOperator_QUANTIZE) {
        return true;
      }
      quantize_ops.push_back(op);
    }
  }
  if (quantize_ops.size() == 1) {
    const auto* tensor = subgraph->tensors[input_idx].get();
    const auto* op = quantize_ops[0];
    const int32_t output_idx = op->outputs[0];
    const auto output_type = subgraph->tensors[output_idx]->type;
    const float output_scale =
        subgraph->tensors[output_idx]->quantization->scale[0];
    const int64_t output_zero_point =
        subgraph->tensors[output_idx]->quantization->zero_point[0];
    if (output_type == tensor->type &&
        output_scale == tensor->quantization->scale[0] &&
        output_zero_point == tensor->quantization->zero_point[0]) {
      return false;
    }
  }
  return true;
}

// Sets the input type, adding a Leading Op node at the start of the model if
// necessary.
// Returns the new input tensor index.
int32_t SetInputType(ModelT* model, SubGraphT* subgraph,
                     const int32_t tensor_idx, const TensorType& input_type,
                     const TensorType& activations_type) {
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  if (!TensorTypeChangeRequired(tensor, input_type)) {
    return -1;
  }
  if (input_type == TensorType_FLOAT32 || input_type == TensorType_UINT8) {
    std::string type_string =
        activations_type == TensorType_INT16 ? "int16" : "int8";
    // Create a new tensor to be the input of the leading Op.
    std::unique_ptr<TensorT> leading_op_input;
    if (input_type == TensorType_FLOAT32) {
      // Add tensor for quantize operator. Scales and zero points are not
      // needed.
      const string leading_op_name = tensor->name;
      const string new_name_original_input = tensor->name + "_" + type_string;
      tensor->name = new_name_original_input;
      utils::MakeTensor(leading_op_name, tensor->shape, tensor->shape_signature,
                        input_type, &leading_op_input);
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
      const string new_name_original_input = tensor->name + "_" + type_string;
      tensor->name = new_name_original_input;
      utils::MakeTensorWithQuantParam(
          leading_op_name, tensor->shape, tensor->shape_signature, input_type,
          scale, zero_point + 128, &leading_op_input);
    }

    // Check if quantize op already exists.
    if (!InputQuantizeRequired(model, subgraph, tensor_idx)) {
      subgraph->tensors[tensor_idx] = std::move(leading_op_input);
      return tensor_idx;
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
                      const int32_t tensor_idx, const TensorType& output_type,
                      const TensorType& activations_type) {
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  if (!TensorTypeChangeRequired(tensor, output_type)) {
    return -1;
  }
  if (output_type == TensorType_FLOAT32 || output_type == TensorType_UINT8) {
    std::string type_string =
        activations_type == TensorType_INT16 ? "int16" : "int8";
    // Create a new tensor to be the output of the tailing op.
    std::unique_ptr<TensorT> tailing_op_output;
    if (output_type == TensorType_FLOAT32) {
      const string tailing_op_name = tensor->name;
      const string new_name_original_output = tensor->name + "_" + type_string;
      tensor->name = new_name_original_output;
      utils::MakeTensor(tailing_op_name, tensor->shape, tensor->shape_signature,
                        output_type, &tailing_op_output);
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
      const string new_name_original_output = tensor->name + "_" + type_string;
      tensor->name = new_name_original_output;
      utils::MakeTensorWithQuantParam(
          tailing_op_name, tensor->shape, tensor->shape_signature, output_type,
          scale, zero_point + 128, &tailing_op_output);
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
TfLiteStatus SetInputAndOutputTypes(ModelT* model, const TensorType& input_type,
                                    const TensorType& output_type,
                                    const TensorType& activations_type,
                                    ErrorReporter* error_reporter) {
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    SignatureDefT* signature_def = nullptr;
    for (const auto& sig_def : model->signature_defs) {
      if (sig_def->subgraph_index == subgraph_idx) {
        signature_def = sig_def.get();
        break;
      }
    }
    for (int i = 0; i < subgraph->inputs.size(); ++i) {
      TensorT* tensor = subgraph->tensors[subgraph->inputs[i]].get();
      // TODO(suharshs): Add support for this case if it ever comes up.
      if (tensor->type == TensorType_FLOAT32 && input_type != tensor->type) {
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "Unsupported input type %s for input tensor %d of type %s.",
            EnumNameTensorType(input_type), subgraph->inputs[i],
            EnumNameTensorType(tensor->type));
        return kTfLiteError;
      }
      const int32_t input_idx = SetInputType(
          model, subgraph, subgraph->inputs[i], input_type, activations_type);
      if (input_idx < 0) {
        continue;
      }
      if (signature_def != nullptr) {
        for (const auto& input : signature_def->inputs) {
          if (input->tensor_index == subgraph->inputs[i]) {
            input->tensor_index = input_idx;
            break;
          }
        }
      }
      subgraph->inputs[i] = input_idx;
    }
    for (int i = 0; i < subgraph->outputs.size(); ++i) {
      TensorT* tensor = subgraph->tensors[subgraph->outputs[i]].get();
      // TODO(suharshs): Add support for this case if it ever comes up.
      if (tensor->type == TensorType_FLOAT32 && output_type != tensor->type) {
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "Unsupported output type %s for output tensor '%s' of type %s.",
            EnumNameTensorType(output_type), tensor->name.c_str(),
            EnumNameTensorType(tensor->type));
        return kTfLiteError;
      }
      const int32_t output_idx = SetOutputType(
          model, subgraph, subgraph->outputs[i], output_type, activations_type);
      if (output_idx < 0) {
        continue;
      }
      if (signature_def != nullptr) {
        for (const auto& output : signature_def->outputs) {
          if (output->tensor_index == subgraph->outputs[i]) {
            output->tensor_index = output_idx;
            break;
          }
        }
      }
      subgraph->outputs[i] = output_idx;
    }
  }
  return kTfLiteOk;
}

// Requantize a constant quantized tensor.
template <typename TensorDataType>
TfLiteStatus RequantizeConstant(
    const std::vector<uint8_t>& buffer_data, const TensorT* tensor,
    const std::unique_ptr<QuantizationParametersT>& new_quantization,
    std::vector<uint8_t>& new_buffer_data) {
  if (new_buffer_data.size() != buffer_data.size()) {
    new_buffer_data.resize(buffer_data.size());
  }
  const auto& quantization = tensor->quantization;
  const std::vector<float>& scales = quantization->scale;
  if (scales.empty()) {
    // No existing quantization, assumes that new quantization parameters
    // are correct.
    new_buffer_data.assign(buffer_data.begin(), buffer_data.end());
    return kTfLiteOk;
  }
  const std::vector<int64_t>& zero_points = quantization->zero_point;
  const int num_elements = buffer_data.size() / sizeof(TensorDataType);
  std::vector<float> float_values(num_elements);
  const TensorDataType* buffer_values =
      reinterpret_cast<const TensorDataType*>(buffer_data.data());
  // This logic is for per-channel quantization, but works for per-tensor.
  const int kPerChannelMaxDim = 4;
  const std::vector<int32_t>& tensor_shape = tensor->shape;
  RuntimeShape unextended_tensor_dims(tensor_shape.size(), tensor_shape.data());
  RuntimeShape tensor_dims =
      RuntimeShape::ExtendedShape(kPerChannelMaxDim, unextended_tensor_dims);
  const int channel_dim_index = quantization->quantized_dimension +
                                kPerChannelMaxDim -
                                unextended_tensor_dims.DimensionsCount();
  int indices[kPerChannelMaxDim];
  for (indices[0] = 0; indices[0] < tensor_dims.Dims(0); indices[0]++) {
    for (indices[1] = 0; indices[1] < tensor_dims.Dims(1); indices[1]++) {
      for (indices[2] = 0; indices[2] < tensor_dims.Dims(2); indices[2]++) {
        for (indices[3] = 0; indices[3] < tensor_dims.Dims(3); indices[3]++) {
          const float scale = scales.size() > 1
                                  ? scales[indices[channel_dim_index]]
                                  : scales[0];
          const int64_t zp = zero_points.size() > 1
                                 ? zero_points[indices[channel_dim_index]]
                                 : zero_points[0];
          const int index = Offset(tensor_dims, indices);
          float_values[index] = scale * (buffer_values[index] - zp);
        }
      }
    }
  }

  // Only have to deal with per-tensor for new parameters.
  if (tensor->type == TensorType_INT16) {
    std::vector<int16_t> requant_int16 = utils::SymmetricQuantizeFloatsToInt16(
        float_values.data(), float_values.size(), new_quantization->scale[0]);
    uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(requant_int16.data());
    new_buffer_data.assign(uint8_buffer, uint8_buffer + buffer_data.size());
    return kTfLiteOk;
  } else if (tensor->type == TensorType_INT8) {
    const int32_t q_min = std::numeric_limits<int8_t>::min();
    const int32_t q_max = std::numeric_limits<int8_t>::max();
    const float scaling_factor = new_quantization->scale[0];
    const int32_t zp = new_quantization->zero_point[0];
    const auto& rescale = [&scaling_factor, &zp, &q_min,
                           &q_max](float f) -> uint8_t {
      const float scaling_factor_inv =
          (scaling_factor == 0) ? 0 : 1.0 / scaling_factor;
      int32_t q_i32 = TfLiteRound(f * scaling_factor_inv) + zp;
      int8_t q = std::min(std::max(q_i32, q_min), q_max);
      return *(reinterpret_cast<uint8_t*>(&q));
    };
    std::transform(float_values.begin(), float_values.end(),
                   new_buffer_data.begin(), rescale);
    return kTfLiteOk;
  }
  return kTfLiteError;
}

// Apply constraints to ops if they have any.
// We have made the restriction that for int8 quantized concat, minimum, and
// maximum, the inputs and outputs must have the same scale and zero point.
// The other ones with constraints are handled in QuantizeWeightsAndInput.
TfLiteStatus ApplyConstraints(
    ModelT* model, const std::unordered_set<string>& operator_names,
    const std::unordered_set<string>& real_value_op_set,
    TensorType activations_type, ErrorReporter* error_reporter) {
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Iterate backward to avoid messing with index.
    for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      if (op->outputs.empty()) {
        continue;
      }
      const string operator_name = subgraph->tensors[op->outputs[0]]->name;
      operator_property::OperatorProperty property =
          GetOperatorProperty(operator_names, model, subgraph_idx, op_idx,
                              operator_name, activations_type);
      if (!property.quantizable ||
          !IsRealValueOp(real_value_op_set, operator_name)) {
        continue;
      }
      TensorT* output_tensor = subgraph->tensors[op->outputs[0]].get();
      if (!property.arbitrary_inputs ||
          !property.restrict_same_input_output_scale(output_tensor->type)) {
        continue;
      }
      // If ApplyConstraints and requant is needed, use the min of min and max
      // of max, which means using the scale and zero point of output.
      if (!utils::QuantizationParametersExist(output_tensor)) {
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "Unable to get scale or zero point from the tensor at %d.",
            op->outputs[0]);
        return kTfLiteError;
      }
      const float output_scale = output_tensor->quantization->scale[0];
      const float output_zp = output_tensor->quantization->zero_point[0];
      for (size_t input_idx = 0; input_idx < op->inputs.size(); ++input_idx) {
        TensorT* input_tensor = subgraph->tensors[op->inputs[input_idx]].get();
        if (!utils::QuantizationParametersExist(input_tensor)) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "Unable to get scale or zero point from tensor at %d.",
              op->inputs[input_idx]);
          return kTfLiteError;
        }
        if (input_tensor->quantization->scale[0] == output_scale &&
            input_tensor->quantization->zero_point[0] == output_zp) {
          // This input does not need to be requantized.
          continue;
        }

        if (IsConstantWithOneUse(model, subgraph, op->inputs[input_idx])) {
          auto quantization = std::make_unique<QuantizationParametersT>();
          quantization->scale.push_back(output_scale);
          quantization->zero_point.push_back(output_zp);
          const std::vector<uint8_t>& buffer_data =
              model->buffers[input_tensor->buffer]->data;
          std::vector<uint8_t> new_buffer_data;
          TfLiteStatus requant_status = kTfLiteError;
          if (input_tensor->type == TensorType_INT8) {
            requant_status = RequantizeConstant<int8_t>(
                buffer_data, input_tensor, quantization, new_buffer_data);
          } else if (input_tensor->type == TensorType_INT16) {
            requant_status = RequantizeConstant<int16_t>(
                buffer_data, input_tensor, quantization, new_buffer_data);
          }
          if (requant_status == kTfLiteOk) {
            model->buffers[input_tensor->buffer]->data = new_buffer_data;
            input_tensor->quantization = std::move(quantization);
            continue;
          } else {
            quantization.release();
          }
        }

        std::unique_ptr<TensorT> additional_tensor;
        const string requant_tensor_name = input_tensor->name + "_requantized";
        utils::MakeTensorWithQuantParam(
            requant_tensor_name, input_tensor->shape,
            input_tensor->shape_signature, activations_type, output_scale,
            output_zp, &additional_tensor);
        const int32_t additional_tensor_idx = subgraph->tensors.size();
        subgraph->tensors.push_back(std::move(additional_tensor));

        // Add requant op before this input.
        // There are better ways to handle this, which is to try to push the
        // rescale upwards recursively and hope all upstream ops can absort
        // this rescale.and only add requant when there is no other way.
        std::unique_ptr<OperatorT> requant_op;
        utils::MakeQuantizeOperator(model, &requant_op, op->inputs[input_idx],
                                    additional_tensor_idx);
        op->inputs[input_idx] = additional_tensor_idx;

        subgraph->operators.insert(subgraph->operators.begin() + op_idx,
                                   std::move(requant_op));
      }
    }
  }
  return kTfLiteOk;
}

// In case of int16 activations, there are two implementations of kernels for
// ADD/SUB operators. We set the builtin option pot_scale_int16
// during quantization so that from now only the general case implementation is
// used.
void SetOperatorPropertyADDSUBOperator(ModelT* model,
                                       const TensorType& activations_type) {
  if (activations_type != TensorType_INT16) {
    // This is needed only in case of int16 activations.
    return;
  }

  for (int subgraph_idx = 0, end = model->subgraphs.size(); subgraph_idx < end;
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Iterate backward to avoid messing with index.
    for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      OperatorCodeT* op_code = model->operator_codes[op->opcode_index].get();
      if (op_code && op_code->builtin_code == BuiltinOperator_ADD) {
        {
          auto* options = op->builtin_options.AsAddOptions();
          if (options) {
            options->pot_scale_int16 = false;
          }
        }
      }
      if (op_code && op_code->builtin_code == BuiltinOperator_SUB) {
        {
          auto* options = op->builtin_options.AsSubOptions();
          if (options) {
            options->pot_scale_int16 = false;
          }
        }
      }
    }
  }
}

std::vector<std::pair<int, operator_property::TensorProperty>> GetInputs(
    const OperatorT* op, operator_property::OperatorProperty property) {
  std::vector<std::pair<int, operator_property::TensorProperty>> inputs;
  if (property.arbitrary_inputs || !property.quantizable) {
    for (int i = 0; i < op->inputs.size(); ++i) {
      inputs.push_back({i, {}});
    }
  } else {
    inputs = property.inputs;
  }
  return inputs;
}

std::vector<std::pair<int, operator_property::TensorProperty>> GetOutputs(
    const OperatorT* op, operator_property::OperatorProperty property) {
  std::vector<std::pair<int, operator_property::TensorProperty>> outputs;
  if (property.arbitrary_outputs) {
    for (int i = 0; i < op->outputs.size(); ++i) {
      outputs.push_back({i, {}});
    }
  } else {
    outputs = property.outputs;
  }
  return outputs;
}

bool ShouldRestrictSameInputOutputScale(
    operator_property::OperatorProperty property, TensorType tensor_type) {
  // Ops with multiple inputs (i.e. concat, max and min) gets restricted in
  // ApplyConstraints.
  return (!property.arbitrary_inputs &&
          property.restrict_same_input_output_scale(tensor_type));
}

bool IsSubgraphInput(SubGraphT* subgraph, int32_t index) {
  for (const int32_t input_idx : subgraph->inputs) {
    if (index == input_idx) {
      return true;
    }
  }
  return false;
}

// Quantize the op input. Will increment op_idx if ops are added.
TfLiteStatus QuantizeOpInput(
    ModelT* model, int32_t subgraph_idx, size_t* op_idx,
    operator_property::OperatorProperty property,
    const std::pair<int32_t, operator_property::TensorProperty>& input,
    const TensorType& activations_type, ErrorReporter* error_reporter) {
  int32_t input_idx = input.first;
  operator_property::TensorProperty tensor_property = input.second;
  SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
  OperatorT* op = subgraph->operators[*op_idx].get();
  const BuiltinOperator op_code =
      GetBuiltinCode(model->operator_codes[op->opcode_index].get());
  if (input_idx >= op->inputs.size()) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Required input index %d is larger than the input length of op "
        "%s at index %d in subgraph %d",
        input_idx, op->inputs.size(), EnumNameBuiltinOperator(op_code), *op_idx,
        subgraph_idx);
    return kTfLiteError;
  }
  const int32_t tensor_idx = op->inputs[input_idx];
  if (tensor_idx == -1) {
    // Skip optional tensor.
    return kTfLiteOk;
  }
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  // Assumes if tensor is quantized, then it is a weight and quantized to 8 bit.
  const bool is_input_quantized = utils::QuantizationParametersExist(tensor);
  if (property.quantizable && !is_input_quantized) {
    // The operation is quantizable, but the input isn't yet quantized.
    if (utils::HasBuffer(model, subgraph, tensor_idx)) {
      // TODO(suharshs): Look at consumers, throw error if one consumer is
      // per-channel and one per-layer.
      bool quantize_const_input = false;
      if (activations_type == TensorType_INT16 &&
          (property.restrict_same_input_output_scale(tensor->type) ||
           property.quantize_input_as_activations)) {
        quantize_const_input = true;
      }
      if (tensor_property.number_of_bits == 8 && !quantize_const_input) {
        if (tensor_property.use_derived_scale) {
          // Currently 8bit tensors in input do not accept derived scale.
          return kTfLiteError;
        }
        if (utils::QuantizeWeight(model, tensor, tensor_property.per_axis,
                                  tensor_property.per_axis_index,
                                  error_reporter) != kTfLiteOk) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "Unable to quantize buffer or min/max value for input %d "
              "in op %s in subgraph %d, node: %d",
              input_idx, EnumNameBuiltinOperator(op_code), subgraph_idx,
              *op_idx);
          return kTfLiteError;
        }
      } else if (tensor_property.number_of_bits == 16 || quantize_const_input) {
        if (tensor_property.use_derived_scale) {
          // Currently 16bit tensors in input do not accept derived scale.
          return kTfLiteError;
        }
        TensorT* tensor = subgraph->tensors[tensor_idx].get();
        int total_size = 1;
        for (int i = 0; i < tensor->shape.size(); ++i) {
          total_size *= tensor->shape[i];
        }
        BufferT* buffer = model->buffers[tensor->buffer].get();
        float* float_data = reinterpret_cast<float*>(buffer->data.data());
        auto minmax = std::minmax_element(float_data, float_data + total_size);
        const float min = *minmax.first;
        const float max = *minmax.second;
        const float range = std::max(std::abs(min), std::abs(max));
        // The narrow range quantized value for int16.
        const float quantize_range = 32767.0;
        const float scale = range / quantize_range;
        return utils::SymmetricQuantizeFloatsToInt16(model, tensor, scale,
                                                     error_reporter);
      } else if (tensor_property.number_of_bits == 32) {
        if (!tensor_property.use_derived_scale) {
          // Currently 32 bit tensors in input only accept derived scale.
          return kTfLiteError;
        }
        TensorT* tensor = subgraph->tensors[tensor_idx].get();
        const float scale = utils::GetEffectiveScale(
            model, subgraph, *op_idx,
            tensor_property.derived_scale.input_tensors,
            tensor_property.derived_scale.intermediate_tensors,
            tensor_property.derived_scale.factors);
        return utils::SymmetricPerLayerBiasQuantize<std::int32_t>(
            model, tensor, scale, error_reporter);

      } else if (tensor_property.number_of_bits == 10) {
        // When the number of bits is 10 (instead of 16), quantize the tensor to
        // [-512, 512], instead of [-32767, 32767].
        TensorT* tensor = subgraph->tensors[tensor_idx].get();
        int total_size = 1;
        for (int i = 0; i < tensor->shape.size(); ++i) {
          total_size *= tensor->shape[i];
        }
        BufferT* buffer = model->buffers[tensor->buffer].get();
        float* buffer_data = reinterpret_cast<float*>(buffer->data.data());
        auto minmax =
            std::minmax_element(buffer_data, buffer_data + total_size);
        const float range =
            std::max(std::abs(*minmax.first), std::abs(*minmax.second));
        const float quantized_range = 512.0;
        const float scale = range / quantized_range;
        return utils::SymmetricQuantizeFloatsToInt16(model, tensor, scale,
                                                     error_reporter);
      } else {
        // Currently supports only 8, 16, 32, 10 bits.
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "Unable to quantize buffer or min/max value for input %d "
            "in op %s in subgraph %d, node: %d",
            input_idx, EnumNameBuiltinOperator(op_code), subgraph_idx, *op_idx);
        return kTfLiteError;
      }
    } else if (utils::HasMinMax(tensor)) {
      if (IsSubgraphInput(subgraph, tensor_idx) ||
          tensor_property.state_tensor) {
        if (tensor_property.number_of_bits == 8) {
          if (tensor_property.use_derived_scale) {
            // Currently 8bit tensors in input do not accept derived scale.
            return kTfLiteError;
          }
          TF_LITE_ENSURE_STATUS(utils::QuantizeActivation(
              tensor, activations_type, error_reporter));
        } else if (tensor_property.number_of_bits == 16) {
          TensorT* tensor = subgraph->tensors[tensor_idx].get();
          float quantized_range = 32767.0;
          float range = std::max(std::abs(tensor->quantization->min[0]),
                                 std::abs(tensor->quantization->max[0]));
          if (tensor_property.extend_to_power_of_two) {
            const int power_of_two_scale = utils::GetPowerOfTwoScale(
                tensor->quantization->min[0], tensor->quantization->max[0]);
            range = std::pow(2, power_of_two_scale);  // NOLINT
            quantized_range = 32768.0;
          }
          const float scale = range / quantized_range;
          utils::QuantizeActivationToInt16(tensor, scale);
        }
      } else {
        // If the tensor is not a model input, we need to add a Quantize
        // operation since the preceding op may require a float output.
        std::string type_string =
            activations_type == TensorType_INT16 ? "int16" : "int8";
        std::unique_ptr<TensorT> op_output;
        utils::MakeTensor(tensor->name + "_" + type_string, tensor->shape,
                          tensor->shape_signature, activations_type,
                          &op_output);
        op_output->quantization = std::make_unique<QuantizationParametersT>();
        op_output->quantization->min.push_back(tensor->quantization->min[0]);
        op_output->quantization->max.push_back(tensor->quantization->max[0]);
        TF_LITE_ENSURE_STATUS(utils::QuantizeActivation(
            op_output.get(), activations_type, error_reporter));
        const int32_t quant_op_output_idx = subgraph->tensors.size();
        subgraph->tensors.push_back(std::move(op_output));
        std::unique_ptr<OperatorT> quant_op;
        utils::MakeQuantizeOperator(model, &quant_op, tensor_idx,
                                    quant_op_output_idx);
        subgraph->operators.insert(subgraph->operators.begin() + *op_idx,
                                   std::move(quant_op));
        op->inputs[input_idx] = quant_op_output_idx;
        *op_idx += 1;
      }
    } else {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Unable to find buffer or min/max value for input "
                           "%d in %s in subgraph %d, node: %d",
                           input_idx, EnumNameBuiltinOperator(op_code),
                           subgraph_idx, *op_idx);
      return kTfLiteError;
    }
  } else if (!property.quantizable && is_input_quantized) {
    // If the tensor is quantized, we have to add a Dequantize op after
    // since this op is not quantizable.
    std::unique_ptr<TensorT> op_output;
    utils::MakeTensor(tensor->name + "_float", tensor->shape,
                      tensor->shape_signature, TensorType_FLOAT32, &op_output);
    const int32_t dequant_op_output_idx = subgraph->tensors.size();
    subgraph->tensors.push_back(std::move(op_output));
    std::unique_ptr<OperatorT> dequant_op;
    utils::MakeDequantizeOperator(model, &dequant_op, tensor_idx,
                                  dequant_op_output_idx);
    subgraph->operators.insert(subgraph->operators.begin() + *op_idx,
                               std::move(dequant_op));
    op->inputs[input_idx] = dequant_op_output_idx;
    *op_idx += 1;
  }
  return kTfLiteOk;
}

// Quantize the op output.
TfLiteStatus QuantizeOpOutput(
    ModelT* model, int32_t subgraph_idx, int32_t op_idx,
    operator_property::OperatorProperty property,
    const std::pair<int32_t, operator_property::TensorProperty>& output,
    TensorType activations_type, ErrorReporter* error_reporter) {
  int32_t output_idx = output.first;
  operator_property::TensorProperty tensor_property = output.second;
  // If the operator is not quantizable, we don't need to do anything for the
  // output.
  if (!property.quantizable) {
    return kTfLiteOk;
  }
  SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
  OperatorT* op = subgraph->operators[op_idx].get();
  const BuiltinOperator op_code =
      GetBuiltinCode(model->operator_codes[op->opcode_index].get());
  if (output_idx >= op->outputs.size()) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "Required output index %d is larger than the output length of "
        "op %s at index %d in subgraph %d",
        output_idx, op->outputs.size(), EnumNameBuiltinOperator(op_code),
        op_idx, subgraph_idx);
    return kTfLiteError;
  }

  TensorT* output_tensor = subgraph->tensors[op->outputs[output_idx]].get();
  if (utils::QuantizationParametersExist(output_tensor)) {
    // Skip output if it has been quantized.
    return kTfLiteOk;
  }
  if (ShouldRestrictSameInputOutputScale(property, output_tensor->type)) {
    // Copy quantization parameter. For average pool, max pool, etc
    // min/max can be different but we want them to be the same.
    // Get scale and zero point of input.
    if (property.inputs[0].first >= op->inputs.size()) {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Required input index %d is larger than the input length of "
          "op %s at index %d in subgraph %d",
          property.inputs[0].first, op->inputs.size(),
          EnumNameBuiltinOperator(op_code), op_idx, subgraph_idx);
      return kTfLiteError;
    }
    const int input_tensor_idx = op->inputs[property.inputs[0].first];
    TensorT* input_tensor = subgraph->tensors[input_tensor_idx].get();
    if (input_tensor->quantization->scale.size() != 1 ||
        input_tensor->quantization->zero_point.size() != 1) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Invalid quantization params for op %s at index %d "
                           "in subgraph %d",
                           EnumNameBuiltinOperator(op_code), op_idx,
                           subgraph_idx);
      return kTfLiteError;
    }

    const float input_scale = input_tensor->quantization->scale[0];
    const int32_t input_zero_point = input_tensor->quantization->zero_point[0];

    // Apply to output.
    output_tensor->quantization = std::make_unique<QuantizationParametersT>();
    output_tensor->quantization->scale.push_back(input_scale);
    output_tensor->quantization->zero_point.push_back(input_zero_point);
    if (!input_tensor->quantization->min.empty()) {
      const float min = input_tensor->quantization->min[0];
      output_tensor->quantization->min = {min};
    }
    if (!input_tensor->quantization->max.empty()) {
      const float max = input_tensor->quantization->max[0];
      output_tensor->quantization->max = {max};
    }
    output_tensor->type = activations_type;
  } else if (tensor_property.restriction) {
    const auto scale_and_zp = activations_type == TensorType_INT16
                                  ? tensor_property.restricted_value_int16
                                  : tensor_property.restricted_value_int8;

    // Apply to output.
    output_tensor->quantization = std::make_unique<QuantizationParametersT>();
    output_tensor->quantization->scale.push_back(scale_and_zp.first);
    output_tensor->quantization->zero_point.push_back(scale_and_zp.second);
    output_tensor->type = activations_type;
  } else {
    // Process regular output that doesn't have any restrictions.
    if (utils::HasMinMax(output_tensor)) {
      utils::QuantizeActivation(output_tensor, activations_type,
                                error_reporter);
    } else {
      TF_LITE_REPORT_ERROR(
          error_reporter,
          "Unable to find min/max value for output %d in %s in "
          "subgraph %d, node: %d",
          output_idx, EnumNameBuiltinOperator(op_code), subgraph_idx, op_idx);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus QuantizeIntermediateTensors(ModelT* model,
                                         TensorType activations_type,
                                         ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      if (!property.intermediates.empty()) {
        OperatorT* op = subgraph->operators[op_idx].get();
        const BuiltinOperator op_code =
            GetBuiltinCode(model->operator_codes[op->opcode_index].get());
        for (const std::pair<int, operator_property::TensorProperty>& input :
             property.intermediates) {
          const int index_local = input.first;
          const int index_global = op->intermediates[index_local];
          if (index_global == -1) {
            // Skip optional tensor.
            continue;
          }
          if (input.second.number_of_bits == 8 &&
              input.second.symmetric == false) {
            TensorT* tensor = subgraph->tensors[index_global].get();
            if (tensor->quantization == nullptr) {
              continue;
            }
            if (utils::HasMinMax(tensor)) {
              utils::QuantizeActivation(tensor, activations_type,
                                        error_reporter);
            } else {
              TF_LITE_REPORT_ERROR(error_reporter,
                                   "Unable to find min/max value for "
                                   "intermediate tensor %d in %s in "
                                   "subgraph %d, node: %d",
                                   index_local,
                                   EnumNameBuiltinOperator(op_code),
                                   subgraph_idx, op_idx);
              return kTfLiteError;
            }
          } else if (input.second.number_of_bits == 16 &&
                     input.second.symmetric == true) {
            TensorT* tensor = subgraph->tensors[index_global].get();
            if (tensor->quantization == nullptr) {
              continue;
            }
            const float min = tensor->quantization->min[0];
            const float max = tensor->quantization->max[0];
            const float range = std::max(std::abs(min), std::abs(max));
            if (range < 1e-8) {
              return kTfLiteError;
            }

            // Get scale and zero point.
            const float quantized_range = 32767.0;
            const float scale = range / quantized_range;
            utils::QuantizeActivationToInt16(tensor, scale);
          } else {
            return kTfLiteError;
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

// Quantize tensors that have shared range. For example, in LSTM, the output
// tensor and input state tensor should share the same range because they are
// using the same scale and zero point.
// We have to model this explicitly because the output is modeled as an extra
// tensor in LSTM. In calibrator, state tensors are logged both before and after
// the inference so the range is fully captured. But output, although it is
// identical to activation, is not a state tensor the input value (range) of the
// very first inference is not captured.
TfLiteStatus QuantizeSharedRange(ModelT* model, ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      if (!property.intermediates.empty()) {
        OperatorT* op = subgraph->operators[op_idx].get();
        for (const std::vector<int>& input : property.restrict_scale) {
          if (input.empty()) {
            continue;
          }
          // Currently only support two values. The first one for input and
          // the second one for output.
          if (input.size() != 2) {
            return kTfLiteError;
          }
          const int index_1 = input[0];
          const int index_2 = input[1];
          TensorT* tensor_1 = subgraph->tensors[op->inputs[index_1]].get();
          TensorT* tensor_2 = subgraph->tensors[op->outputs[index_2]].get();
          const float min_of_min = std::min(tensor_1->quantization->min[0],
                                            tensor_2->quantization->min[0]);
          const float max_of_max = std::max(tensor_1->quantization->max[0],
                                            tensor_2->quantization->max[0]);
          if (min_of_min == 0.0 && max_of_max == 0.0) {
            return kTfLiteError;
          }

          // Asmmetric quantization to 8 bit.
          auto quantization_params =
              std::make_unique<QuantizationParametersT>();
          utils::GetAsymmetricQuantizationParams(
              min_of_min, max_of_max, -128, 127, quantization_params.get());

          // Populate both tensors with the same parameters.
          const float scale = quantization_params->scale[0];
          const int32 zero_point = quantization_params->zero_point[0];
          for (TensorT* tensor : {tensor_1, tensor_2}) {
            tensor->quantization = std::make_unique<QuantizationParametersT>();
            tensor->quantization->scale.push_back(scale);
            tensor->quantization->zero_point.push_back(zero_point);
            tensor->type = TensorType_INT8;
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

// Quantize a constant based on min/max quantization parameters for
// resource assignments during initialization. Constant buffers should
// have the same quantization parameters as assignments.
TfLiteStatus QuantizeConstantVariable(ModelT* model,
                                      const TensorType& activations_type,
                                      TensorT* var_tensor,
                                      ErrorReporter* error_reporter) {
  if (activations_type == TensorType_INT16) {
    const float min = var_tensor->quantization->min[0];
    const float max = var_tensor->quantization->max[0];
    const float range = std::max(std::abs(min), std::abs(max));
    const float quantize_range = 32767.0;
    const float scale = range / quantize_range;
    return utils::SymmetricQuantizeFloatsToInt16(model, var_tensor, scale,
                                                 error_reporter);
  } else if (activations_type == TensorType_INT8) {
    TF_LITE_ENSURE_STATUS(utils::QuantizeActivation(
        var_tensor, activations_type, error_reporter));
    QuantizationParametersT* quantization_params =
        var_tensor->quantization.get();
    const float scaling_factor = quantization_params->scale[0];
    const int zero_point = quantization_params->zero_point[0];
    const BufferT* buffer = model->buffers[var_tensor->buffer].get();
    const float* float_data =
        reinterpret_cast<const float*>(buffer->data.data());
    uint64_t num_elements;
    TF_LITE_ENSURE_STATUS(utils::NumElements(*var_tensor, &num_elements));
    const float scaling_factor_inv =
        (scaling_factor == 0) ? 0 : 1.0 / scaling_factor;
    std::vector<int8_t> quantized(num_elements);
    const int32_t kMinScale = std::numeric_limits<int8_t>::min();
    const int32_t kMaxScale = std::numeric_limits<int8_t>::max();
    for (size_t i = 0; i < num_elements; i++) {
      const int32_t quantized_value = static_cast<int32_t>(
          TfLiteRound(float_data[i] * scaling_factor_inv) + zero_point);
      quantized[i] = std::min(kMaxScale, std::max(kMinScale, quantized_value));
    }
    uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(quantized.data());
    const size_t buffer_size = num_elements * sizeof(int8_t);
    model->buffers[var_tensor->buffer]->data.assign(uint8_buffer,
                                                    uint8_buffer + buffer_size);
    return kTfLiteOk;
  }
  return kTfLiteError;
}

using TensorResourceMap = std::map<std::pair<int, int>, std::string>;
using ResourceMinMaxMap = std::map<std::string, std::pair<float, float>>;
// Find min of mins, max of maxes for each variable read or assignment.
void PopulateResourceMinMaxMap(ModelT* model,
                               TensorResourceMap& tensor_resource_map,
                               ResourceMinMaxMap& resource_min_max_map) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      if (op_code == BuiltinOperator_VAR_HANDLE) {
        const std::string& name =
            op->builtin_options.AsVarHandleOptions()->shared_name;
        resource_min_max_map.insert({name, {0.0, 0.0}});
        tensor_resource_map.insert({{subgraph_idx, op->outputs[0]}, name});
      }
      if ((op_code == BuiltinOperator_ASSIGN_VARIABLE) ||
          (op_code == BuiltinOperator_READ_VARIABLE)) {
        if (tensor_resource_map.find({subgraph_idx, op->inputs[0]}) ==
            tensor_resource_map.end()) {
          continue;
        }
        const std::string& name =
            tensor_resource_map[{subgraph_idx, op->inputs[0]}];
        TensorT* var_tensor;
        if (op_code == BuiltinOperator_ASSIGN_VARIABLE) {
          var_tensor = subgraph->tensors[op->inputs[1]].get();
        } else if (op_code == BuiltinOperator_READ_VARIABLE) {
          var_tensor = subgraph->tensors[op->outputs[0]].get();
        } else {
          continue;
        }
        if (!var_tensor->quantization ||
            var_tensor->quantization->min.empty() ||
            var_tensor->quantization->max.empty()) {
          continue;
        }
        // resources are quantized per tensor.
        const float current_min = var_tensor->quantization->min[0];
        const float current_max = var_tensor->quantization->max[0];
        auto inserted =
            resource_min_max_map.insert({name, {current_min, current_max}});
        if (!inserted.second) {
          resource_min_max_map[name] = {
              std::min(inserted.first->second.first, current_min),
              std::max(inserted.first->second.second, current_max)};
        }
      }
    }
  }
}

// Quantize resource variables. Each resource read and assign should have
// identical quantization parameters.
TfLiteStatus QuantizeResources(ModelT* model,
                               const TensorType& activations_type,
                               ErrorReporter* error_reporter) {
  // Shared name is only stored in the var handle operator, use resoure name map
  // to map tensors to resource names.
  TensorResourceMap tensor_resource_map;
  ResourceMinMaxMap resource_min_max_map;
  PopulateResourceMinMaxMap(model, tensor_resource_map, resource_min_max_map);
  if (resource_min_max_map.empty()) {
    // No resources found, so this is OK.
    return kTfLiteOk;
  }
  // Update quantization parameters.
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      if (op_code == BuiltinOperator_ASSIGN_VARIABLE ||
          op_code == BuiltinOperator_READ_VARIABLE) {
        if (tensor_resource_map.find({subgraph_idx, op->inputs[0]}) ==
            tensor_resource_map.end()) {
          continue;
        }
        const std::string& name =
            tensor_resource_map[{subgraph_idx, op->inputs[0]}];
        TensorT* var_tensor = nullptr;
        bool is_constant_assign = false;
        if (op_code == BuiltinOperator_ASSIGN_VARIABLE) {
          var_tensor = subgraph->tensors[op->inputs[1]].get();
          is_constant_assign = utils::HasBuffer(model, subgraph, op->inputs[1]);
        } else if (op_code == BuiltinOperator_READ_VARIABLE) {
          var_tensor = subgraph->tensors[op->outputs[0]].get();
        } else {
          continue;
        }
        if (resource_min_max_map.find(name) == resource_min_max_map.end()) {
          continue;
        }
        if (!var_tensor->quantization) {
          var_tensor->quantization =
              std::make_unique<QuantizationParametersT>();
          var_tensor->quantization->min.push_back(
              resource_min_max_map[name].first);
          var_tensor->quantization->max.push_back(
              resource_min_max_map[name].second);
        } else {
          var_tensor->quantization->min[0] = resource_min_max_map[name].first;
          var_tensor->quantization->max[0] = resource_min_max_map[name].second;
        }
        if (!is_constant_assign) {
          continue;
        }
        if (QuantizeConstantVariable(model, activations_type, var_tensor,
                                     error_reporter) != kTfLiteOk) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "Unable to quantize buffer or min/max value for assignment "
              "in op %s in subgraph %d, node: %d",
              EnumNameBuiltinOperator(op_code), subgraph_idx, op_idx);
          return kTfLiteError;
        }
      }
    }
  }
  return kTfLiteOk;
}

// Quantize inputs and weights.
// Because of ops such as lstm, still need to do per op, instead of weights.
TfLiteStatus QuantizeWeightsInputOutput(
    ModelT* model, bool allow_float,
    const std::unordered_set<string>& operator_names,
    const std::unordered_set<string>& real_value_op_set,
    const TensorType& activations_type, bool disable_per_channel,
    ErrorReporter* error_reporter) {
  // Flag to track unsupported ops.
  bool quantization_not_supported = false;

  // Loop over the graph and quantize ops.
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      if (op->outputs.empty() && op_code != BuiltinOperator_ASSIGN_VARIABLE) {
        continue;
      }
      const string operator_name = op_code != BuiltinOperator_ASSIGN_VARIABLE
                                       ? subgraph->tensors[op->outputs[0]]->name
                                       : subgraph->tensors[op->inputs[0]]->name;
      operator_property::OperatorProperty property = GetOperatorProperty(
          operator_names, model, subgraph_idx, op_idx, operator_name,
          activations_type, disable_per_channel);
      if (!IsRealValueOp(real_value_op_set, operator_name)) {
        continue;
      }

      if (activations_type == TensorType_INT16 && !property.quantizable &&
          !allow_float) {
        TF_LITE_REPORT_ERROR(
            error_reporter,
            "Quantization to 16x8-bit not yet supported for op: '%s'.\n",
            EnumNameBuiltinOperator(op_code));
        quantization_not_supported = true;
      } else if (!property.quantizable && !allow_float) {
        if (op_code == BuiltinOperator_DEQUANTIZE &&
            std::find(subgraph->outputs.begin(), subgraph->outputs.end(),
                      op->outputs[0]) != subgraph->outputs.end()) {
          continue;
        }
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Quantization not yet supported for op: '%s'.\n",
                             EnumNameBuiltinOperator(op_code));
        quantization_not_supported = true;
      }

      // Quantize operator inputs/weights.
      for (const std::pair<int, operator_property::TensorProperty>& input :
           GetInputs(op, property)) {
        TF_LITE_ENSURE_STATUS(QuantizeOpInput(model, subgraph_idx, &op_idx,
                                              property, input, activations_type,
                                              error_reporter));
      }

      // Quantize operator outputs.
      for (const std::pair<int, operator_property::TensorProperty>& output :
           GetOutputs(op, property)) {
        TF_LITE_ENSURE_STATUS(
            QuantizeOpOutput(model, subgraph_idx, op_idx, property, output,
                             activations_type, error_reporter));
      }
    }
  }

  // Return; emit errors if there are any.
  if (quantization_not_supported) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

// Quantize bias.
TfLiteStatus QuantizeBiases(ModelT* model,
                            const std::unordered_set<string>& operator_names,
                            const std::unordered_set<string>& real_value_op_set,
                            const TensorType& activations_type,
                            const TensorType& bias_type,
                            bool disable_per_channel,
                            ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          GetBuiltinCode(model->operator_codes[op->opcode_index].get());
      if (op->outputs.empty()) {
        continue;
      }
      const string operator_name = subgraph->tensors[op->outputs[0]]->name;
      operator_property::OperatorProperty property = GetOperatorProperty(
          operator_names, model, subgraph_idx, op_idx, operator_name,
          activations_type, disable_per_channel);
      if (!property.quantizable ||
          !IsRealValueOp(real_value_op_set, operator_name)) {
        continue;
      }
      for (const int bias_idx : property.biases) {
        if (bias_idx >= op->inputs.size() ||
            op->inputs[bias_idx] == kTfLiteOptionalTensor) {
          continue;
        }
        // Quantize if it is not quantized already as the
        // output of another op or input of another op.
        TensorT* bias_tensor = subgraph->tensors[op->inputs[bias_idx]].get();
        if (!utils::QuantizationParametersExist(bias_tensor)) {
          if (utils::HasBuffer(model, subgraph, op->inputs[bias_idx])) {
            if (property.inputs.size() != 2) {
              TF_LITE_REPORT_ERROR(error_reporter,
                                   "Expect the input length of "
                                   "op %s at index %d in subgraph %d to be 2",
                                   bias_idx, op->inputs.size(),
                                   EnumNameBuiltinOperator(op_code), op_idx,
                                   subgraph_idx);
              return kTfLiteError;
            }
            TensorT* input_tensor =
                subgraph->tensors[op->inputs[property.inputs[0].first]].get();
            TensorT* weight_tensor =
                subgraph->tensors[op->inputs[property.inputs[1].first]].get();
            operator_property::TensorProperty weight_property =
                property.inputs[1].second;
            TF_LITE_ENSURE_STATUS(QuantizeBias(
                model, input_tensor, weight_tensor, bias_tensor,
                weight_property.per_axis, weight_property.per_axis_index,
                bias_type, error_reporter));
          }
        } else {
          // If bias is already quantized, make sure it is quantized to 32 bit.
          if (bias_tensor->type != TensorType_INT32) {
            TF_LITE_REPORT_ERROR(
                error_reporter,
                "Bias (\"%s\" at global index %d) of op \"%s\" at op_index %d "
                "in subgraph %d is expected to be quantized to INT32 but it is "
                "already quantized to %s.\n",
                bias_tensor->name.c_str(), op->inputs[bias_idx],
                operator_name.c_str(), op_idx, subgraph_idx,
                EnumNameTensorType(bias_tensor->type));
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

std::unordered_set<string> GetAllOperatorOutputs(ModelT* model) {
  std::unordered_set<string> operator_names;
  for (int32_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (int32_t tensor_idx = 0; tensor_idx < subgraph->tensors.size();
         tensor_idx++) {
      operator_names.insert(subgraph->tensors[tensor_idx]->name);
    }
  }
  return operator_names;
}
// Populate the quantization parameters max and min for input tensors.
// Assumes that dynamic tensors already have stored min, max values and throw
// an error if a tensor does not have min, max quantization parameter or a
// buffer.
// If any static tensors are not inputs to an operation, their max, min values
// will not be filled by this function.
TfLiteStatus FillQuantizationParams(
    ModelT* model, const std::unordered_set<string>& operator_names,
    const std::unordered_set<string>& real_value_op_set,
    const TensorType& activations_type, bool disable_per_channel,
    ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(model, subgraph_idx, op_idx);
      if (!property.quantizable) {
        continue;
      }
      if (!op->outputs.empty()) {
        const string operator_name = subgraph->tensors[op->outputs[0]]->name;
        property = GetOperatorProperty(operator_names, model, subgraph_idx,
                                       op_idx, operator_name, activations_type,
                                       disable_per_channel);
        if (!IsRealValueOp(real_value_op_set, operator_name)) {
          continue;
        }
      }

      // Populate max, min for each input tensor.
      for (const std::pair<int, operator_property::TensorProperty>& input :
           property.inputs) {
        // Get tensor.
        const int32_t input_idx = input.first;
        const int32_t tensor_idx = op->inputs[input_idx];
        if (tensor_idx == -1) {
          // Skip optional tensor.
          continue;
        }
        TensorT* tensor = subgraph->tensors[tensor_idx].get();

        // Static tensor.
        if (!utils::HasMinMax(tensor) &&
            utils::HasBuffer(model, subgraph, tensor_idx)) {
          // Get input float data and tensor dimensions.
          const BufferT* buffer = model->buffers[tensor->buffer].get();
          const float* float_input_data =
              reinterpret_cast<const float*>(buffer->data.data());

          if (tensor->quantization == nullptr) {
            tensor->quantization = std::make_unique<QuantizationParametersT>();
          }

          // Fill per channel max and min with respect to channel_dim_index.
          if (input.second.per_axis) {
            if (tensor->shape.size() == 4) {
              int32_t channel_dim_index = input.second.per_axis_index;
              TF_LITE_ENSURE_STATUS(utils::FillPerChannelMinMax(
                  float_input_data, tensor->shape, channel_dim_index,
                  tensor->quantization.get(), error_reporter));
            } else {
              TF_LITE_REPORT_ERROR(
                  error_reporter,
                  "Could not fill max min for tensor as the dimension is %d "
                  "and not 4 as expected.",
                  tensor->shape.size());
              return kTfLiteError;
            }

            // Fill per layer max and min.
          } else if (!utils::HasMinMax(tensor) && !input.second.per_axis &&
                     utils::HasBuffer(model, subgraph, tensor_idx)) {
            uint64_t input_size;
            TF_LITE_ENSURE_STATUS(utils::NumElements(*tensor, &input_size));
            utils::FillSingleMinMax(float_input_data, input_size,
                                    tensor->quantization.get());
          }
          if (tensor->quantization->quantized_dimension !=
              input.second.per_axis_index) {
            TF_LITE_REPORT_ERROR(
                error_reporter,
                "Quantized dimension for tensor property and quantization "
                "parameters do not match. Got %d and %d respectively.",
                input.second.per_axis_index,
                tensor->quantization->quantized_dimension);
            return kTfLiteError;
          }

          // Dynamic tensor.
        } else if (!utils::HasMinMax(tensor) &&
                   !utils::HasBuffer(model, subgraph, tensor_idx)) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "Max and min for dynamic tensors should be"
              " recorded during calibration: Failed for tensor %s\n",
              tensor->name.c_str());
          if (tensor->quantization == nullptr) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                 "No quantization params for tensor %s",
                                 tensor->name.c_str());
          } else if (tensor->quantization->min.empty() ||
                     tensor->quantization->max.empty()) {
            TF_LITE_REPORT_ERROR(error_reporter, "Empty min/max for tensor %s",
                                 tensor->name.c_str());
          }
          return kTfLiteError;
        }

        if (utils::QuantizationParametersExist(tensor)) {
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "Scale and zero points should not be recorded before "
              "quantization.");
          return kTfLiteError;
        }
      }  // loop over op inputs
    }    // loop over ops
  }      // loop over subgraphs
  return kTfLiteOk;
}

// Check compatibility of activation, weight and bias scales. Adjust if needed.
TfLiteStatus EnsureBiasScaleCompatibility(
    ModelT* model, const std::unordered_set<string>& operator_names,
    const std::unordered_set<string>& real_value_op_set,
    const TensorType& activations_type, bool disable_per_channel,
    ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      if (op->outputs.empty()) {
        continue;
      }
      const string operator_name = subgraph->tensors[op->outputs[0]]->name;
      operator_property::OperatorProperty property = GetOperatorProperty(
          operator_names, model, subgraph_idx, op_idx, operator_name,
          activations_type, disable_per_channel);
      if (!IsRealValueOp(real_value_op_set, operator_name)) {
        continue;
      }

      // Loop over all bias tensors.
      for (const int bias_idx : property.biases) {
        if (bias_idx >= op->inputs.size() ||
            op->inputs[bias_idx] == kTfLiteOptionalTensor) {
          continue;
        }
        TensorT* bias_tensor = subgraph->tensors[op->inputs[bias_idx]].get();
        int32_t channel_dim_size = bias_tensor->shape[0];
        if (bias_tensor->shape.size() != 1) {
          TF_LITE_REPORT_ERROR(error_reporter,
                               "Expected bias tensor to be a vector.");
          return kTfLiteError;
        }

        if (property.inputs.size() != 2) {  // Only works for two input tensors.
          TF_LITE_REPORT_ERROR(
              error_reporter,
              "Expect %d inputs for op %s at index %d in subgraph %d to be 2",
              property.inputs.size(), op_idx, subgraph_idx);
          return kTfLiteError;
        }

        if (!property.arbitrary_inputs && property.quantizable) {
          // Get input and weight tensors.
          TensorT* input_tensor =
              subgraph->tensors[op->inputs[property.inputs[0].first]].get();
          TensorT* weight_tensor =
              subgraph->tensors[op->inputs[property.inputs[1].first]].get();
          operator_property::TensorProperty weight_property =
              property.inputs[1].second;
          TF_LITE_ENSURE(error_reporter, input_tensor->quantization);

          // Check quantization parameters exist for input.
          if (!utils::HasMinMax(input_tensor)) {
            TF_LITE_REPORT_ERROR(
                error_reporter,
                "Input tensor missing quantization information. Should be "
                "populated during calibration.");
            return kTfLiteError;
          }

          // Get input scale for asymmetric quantization.
          QuantizationParametersT temp_quant_params = QuantizationParametersT();
          TF_LITE_ENSURE_STATUS(
              utils::GetQuantizationParams(input_tensor, activations_type,
                                           &temp_quant_params, error_reporter));
          if (temp_quant_params.scale.size() != 1) {
            TF_LITE_REPORT_ERROR(error_reporter,
                                 "Unexpected input quantization scale size.");
            return kTfLiteError;
          }
          float input_scale = temp_quant_params.scale[0];

          // Check that max/min values have been filled for weights.
          if (!utils::HasMinMax(weight_tensor)) {
            TF_LITE_REPORT_ERROR(
                error_reporter,
                "Min and/or max values have not been recorded for weight "
                "tensor. This should have happened in FillQuantizationParams.");
            return kTfLiteError;
          }

          // Ensure the tensor dimensions are compatible.
          if (weight_property.per_axis) {
            if (bias_tensor->shape[0] !=
                weight_tensor->shape[weight_property.per_axis_index]) {
              TF_LITE_REPORT_ERROR(
                  error_reporter,
                  "Channel mismatch between bias and weight tensors %d vs %d",
                  bias_tensor->shape[0],
                  weight_tensor->shape[weight_property.per_axis_index]);
              return kTfLiteError;
            }
            // Ensure that the number of max/mins matches the channel_dim_size.
            if (weight_tensor->quantization->max.size() != channel_dim_size) {
              TF_LITE_REPORT_ERROR(
                  error_reporter,
                  "Mismatch between number of weight maxs and channels: %d vs "
                  "%d",
                  weight_tensor->quantization->max.size(), channel_dim_size);
              return kTfLiteError;
            }
            if (weight_tensor->quantization->min.size() != channel_dim_size) {
              TF_LITE_REPORT_ERROR(
                  error_reporter,
                  "Mismatch between number of weight mins and channels: %d",
                  weight_tensor->quantization->min.size());
              return kTfLiteError;
            }
          }

          // Get data and size of bias tensor.
          const BufferT* buffer = model->buffers[bias_tensor->buffer].get();
          const float* bias_data =
              reinterpret_cast<const float*>(buffer->data.data());
          uint64_t bias_size;
          TF_LITE_ENSURE_STATUS(utils::NumElements(*bias_tensor, &bias_size));

          // Adjust weight scales if needed.
          TF_LITE_ENSURE_STATUS(utils::AdjustWeightsForBiasScale(
              weight_tensor->quantization.get(), bias_data, bias_size,
              input_scale, error_reporter));

          if (utils::QuantizationParametersExist(weight_tensor)) {
            TF_LITE_REPORT_ERROR(
                error_reporter,
                "Scale and zero points should not be recorded for the weight "
                "tensor before quantization.");
            return kTfLiteError;
          }
          if (utils::QuantizationParametersExist(input_tensor)) {
            TF_LITE_REPORT_ERROR(
                error_reporter,
                "Scale and zero points should not be recorded for the input "
                "tensor before quantization.");
            return kTfLiteError;
          }
        }
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace

// Assumes that the operators in the model have been topologically sorted.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           const std::unordered_set<string>& operator_names,
                           const TensorType& activations_type,
                           const TensorType& bias_type,
                           bool disable_per_channel,
                           ErrorReporter* error_reporter) {
  auto real_value_op_set =
      PopulateRealValueOpSet(model, operator_names, activations_type);
  TF_LITE_ENSURE_STATUS(DuplicateBiasesWithMultipleUses(model, error_reporter));
  TF_LITE_ENSURE_STATUS(FillQuantizationParams(
      model, operator_names, real_value_op_set, activations_type,
      disable_per_channel, error_reporter));
  TF_LITE_ENSURE_STATUS(EnsureBiasScaleCompatibility(
      model, operator_names, real_value_op_set, activations_type,
      disable_per_channel, error_reporter));
  TF_LITE_ENSURE_STATUS(
      QuantizeIntermediateTensors(model, activations_type, error_reporter));
  TF_LITE_ENSURE_STATUS(QuantizeSharedRange(model, error_reporter));
  TF_LITE_ENSURE_STATUS(
      QuantizeResources(model, activations_type, error_reporter));
  TF_LITE_ENSURE_STATUS(QuantizeWeightsInputOutput(
      model, allow_float, operator_names, real_value_op_set, activations_type,
      disable_per_channel, error_reporter));
  TF_LITE_ENSURE_STATUS(ApplyConstraints(model, operator_names,
                                         real_value_op_set, activations_type,
                                         error_reporter));
  SetOperatorPropertyBiasType(model, bias_type);
  TF_LITE_ENSURE_STATUS(QuantizeBiases(model, operator_names, real_value_op_set,
                                       activations_type, bias_type,
                                       disable_per_channel, error_reporter));
  utils::SetOperatorCodeVersion(model);
  TF_LITE_ENSURE_STATUS(SetInputAndOutputTypes(
      model, input_type, output_type, activations_type, error_reporter));
  SetOperatorPropertyADDSUBOperator(model, activations_type);
  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model);
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
}

// Assumes that the operators in the model have been topologically sorted.
TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           const std::unordered_set<string>& operator_names,
                           const TensorType& activations_type,
                           const TensorType& bias_type,
                           ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       operator_names, activations_type,
                       /*bias_type=*/bias_type,
                       /*disable_per_channel=*/false, error_reporter);
}

TfLiteStatus QuantizeModelAllOperators(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const TensorType& activations_type,
    const TensorType& bias_type, ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       GetAllOperatorOutputs(model), activations_type,
                       bias_type,
                       /*disable_per_channel=*/false, error_reporter);
}

TfLiteStatus QuantizeModelAllOperators(
    flatbuffers::FlatBufferBuilder* builder, ModelT* model,
    const TensorType& input_type, const TensorType& output_type,
    bool allow_float, const TensorType& activations_type,
    const TensorType& bias_type, bool disable_per_channel,
    ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       GetAllOperatorOutputs(model), activations_type,
                       bias_type, disable_per_channel, error_reporter);
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type, bool allow_float,
                           ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       GetAllOperatorOutputs(model),
                       /*activations_type=*/TensorType_INT8,
                       /*bias_type=*/TensorType_INT32, error_reporter);
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, const TensorType& input_type,
                           const TensorType& output_type,
                           ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type,
                       /*allow_float=*/false, error_reporter);
}

TfLiteStatus QuantizeModel(flatbuffers::FlatBufferBuilder* builder,
                           ModelT* model, ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, TensorType_FLOAT32, TensorType_FLOAT32,
                       /*allow_float=*/false, error_reporter);
}

}  // namespace optimize
}  // namespace tflite
