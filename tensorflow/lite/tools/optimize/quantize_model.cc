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
#include <vector>

#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/model_utils.h"
#include "tensorflow/lite/tools/optimize/operator_property.h"
#include "tensorflow/lite/tools/optimize/quantization_utils.h"

namespace tflite {
namespace optimize {

namespace {

TfLiteStatus QuantizeBias(ModelT* model, const TensorT* input_tensor,
                          const TensorT* weight_tensor, TensorT* bias_tensor,
                          bool is_per_channel, int channel_dim_index,
                          ErrorReporter* error_reporter) {
  if (bias_tensor->shape.size() != 1) {
    error_reporter->Report("Expected bias tensor shape to be 1.");
    return kTfLiteError;
  }

  int32_t channel_dim_size = bias_tensor->shape[0];
  TF_LITE_ENSURE(error_reporter, weight_tensor->quantization);
  std::vector<float> weight_scales = weight_tensor->quantization->scale;

  if (is_per_channel) {
    if (bias_tensor->shape[0] != weight_tensor->shape[channel_dim_index]) {
      error_reporter->Report(
          "Channel mismatch between bias and weight tensors %d vs %d",
          bias_tensor->shape[0], weight_tensor->shape[channel_dim_index]);
      return kTfLiteError;
    }
    if (!input_tensor->quantization ||
        input_tensor->quantization->scale.size() != 1) {
      error_reporter->Report("Input tensor missing quantization information");
      return kTfLiteError;
    }

    if (weight_scales.size() != channel_dim_size) {
      error_reporter->Report("Mismatch weight scale dimension: %d",
                             weight_scales.size());
      return kTfLiteError;
    }
    return utils::SymmetricPerChannelBiasQuantize(
        model, bias_tensor, input_tensor->quantization->scale[0],
        weight_scales.data(), channel_dim_size, channel_dim_index);
  } else {
    if (weight_scales.size() != 1) {
      error_reporter->Report(
          "Expected per-layer weight scale dimension size 1, got %d",
          weight_scales.size());
      return kTfLiteError;
    }
    return utils::SymmetricPerLayerBiasQuantize(
        model, bias_tensor, input_tensor->quantization->scale[0],
        weight_scales[0]);
  }
  return kTfLiteError;
}

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
TfLiteStatus SetInputAndOutputTypes(ModelT* model, const TensorType& input_type,
                                    const TensorType& output_type,
                                    ErrorReporter* error_reporter) {
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();

    for (int i = 0; i < subgraph->inputs.size(); ++i) {
      TensorT* tensor = subgraph->tensors[subgraph->inputs[i]].get();
      // TODO(suharshs): Add support for this case if it ever comes up.
      if (tensor->type == TensorType_FLOAT32 && input_type != tensor->type) {
        error_reporter->Report(
            "Unsupported input type %s for input tensor %d of type %s.",
            EnumNameTensorType(input_type), subgraph->inputs[i],
            EnumNameTensorType(tensor->type));
        return kTfLiteError;
      }
      const int32_t input_idx =
          SetInputType(model, subgraph, subgraph->inputs[i], input_type);
      if (input_idx < 0) {
        continue;
      }
      subgraph->inputs[i] = input_idx;
    }
    for (int i = 0; i < subgraph->outputs.size(); ++i) {
      TensorT* tensor = subgraph->tensors[subgraph->outputs[i]].get();
      // TODO(suharshs): Add support for this case if it ever comes up.
      if (tensor->type == TensorType_FLOAT32 && output_type != tensor->type) {
        error_reporter->Report(
            "Unsupported output type %s for output tensor %d of type %s.",
            EnumNameTensorType(output_type), subgraph->outputs[i],
            EnumNameTensorType(tensor->type));
        return kTfLiteError;
      }
      const int32_t output_idx =
          SetOutputType(model, subgraph, subgraph->outputs[i], output_type);
      if (output_idx < 0) {
        continue;
      }
      subgraph->outputs[i] = output_idx;
    }
  }
  return kTfLiteOk;
}

// Apply constraints to ops if they have any.
// We have made the restriction that for int8 quantized concat, the inputs and
// outpus must have the same scale and zero point. The other ones with
// constraints(averagepool, maxpool, gather, softmax, tanh etc) are handled in
// QuantizeWeightsAndInput.
TfLiteStatus ApplyConstraints(ModelT* model, ErrorReporter* error_reporter) {
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Iterate backward to avoid messing with index.
    for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          model->operator_codes[op->opcode_index]->builtin_code;
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(op_code);
      if (!property.quantizable) {
        continue;
      }
      // Basically only Concat passes this check.
      if (!property.restrict_same_input_output_scale ||
          (property.input_indexes.size() == 1 &&
           property.output_indexes.size() == 1 && property.biases.empty())) {
        continue;
      }
      // If ApplyConstraintsnd requant is needed, use the min of min and max of
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
        utils::MakeTensorWithQuantParam(
            requant_tensor_name, input_tensor->shape, TensorType_INT8,
            output_scale, output_zp, &additional_tensor);
        const int32_t additional_tensor_idx = subgraph->tensors.size();
        subgraph->tensors.push_back(std::move(additional_tensor));

        // Add requant op before this input.
        // There are better ways to handle this, which is to try to push the
        // rescale upwards recurrsively and hope all upstream ops can absort
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

std::vector<int> GetInputIndexes(const OperatorT* op,
                                 operator_property::OperatorProperty property) {
  std::vector<int> input_indexes;
  if (property.arbitrary_inputs || !property.quantizable) {
    for (int i = 0; i < op->inputs.size(); ++i) {
      input_indexes.push_back(i);
    }
  } else {
    input_indexes = property.input_indexes;
  }
  return input_indexes;
}

bool ShouldRestrictSameInputOutputScale(
    operator_property::OperatorProperty property) {
  return (property.input_indexes.size() == 1 &&
          property.output_indexes.size() == 1 && property.biases.empty() &&
          property.restrict_same_input_output_scale);
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
TfLiteStatus QuantizeOpInput(ModelT* model, int32_t subgraph_idx,
                             size_t* op_idx,
                             operator_property::OperatorProperty property,
                             int32_t input_idx, ErrorReporter* error_reporter) {
  SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
  OperatorT* op = subgraph->operators[*op_idx].get();
  const BuiltinOperator op_code =
      model->operator_codes[op->opcode_index]->builtin_code;
  const int32_t tensor_idx = op->inputs[input_idx];
  TensorT* tensor = subgraph->tensors[tensor_idx].get();
  const bool is_input_quantized = utils::IsQuantized(subgraph, tensor_idx);
  if (input_idx >= op->inputs.size()) {
    error_reporter->Report(
        "Required input index %d is larger than the input length of op "
        "%s at index %d in subgraph %d",
        input_idx, op->inputs.size(), EnumNameBuiltinOperator(op_code), *op_idx,
        subgraph_idx);
    return kTfLiteError;
  }
  if (property.quantizable && !is_input_quantized) {
    // The operation is quantizable, but the input isn't yet quantized.
    if (utils::HasBuffer(model, subgraph, tensor_idx)) {
      if (utils::QuantizeWeight(model, tensor, property.per_axis,
                                property.per_axis_index) == kTfLiteError) {
        error_reporter->Report(
            "Unable to quantize buffer or min/max value for input %d "
            "in op %s in subgraph %d, node: %d",
            input_idx, EnumNameBuiltinOperator(op_code), subgraph_idx, *op_idx);
        return kTfLiteError;
      }
    } else if (utils::HasMinMax(tensor)) {
      if (IsSubgraphInput(subgraph, tensor_idx)) {
        utils::QuantizeActivation(tensor);
      } else {
        // If the tensor is not a model input, we need to add a Quantize
        // operation since the preceding op may require a float output.
        std::unique_ptr<TensorT> op_output;
        utils::MakeTensor(tensor->name + "_int8", tensor->shape,
                          TensorType_INT8, &op_output);
        op_output->quantization = absl::make_unique<QuantizationParametersT>();
        op_output->quantization->min.push_back(tensor->quantization->min[0]);
        op_output->quantization->max.push_back(tensor->quantization->max[0]);
        utils::QuantizeActivation(op_output.get());
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
      error_reporter->Report(
          "Unable to find buffer or min/max value for input activation "
          "%d "
          "in %s in subgraph %d, node: %d",
          input_idx, EnumNameBuiltinOperator(op_code), subgraph_idx, *op_idx);
      return kTfLiteError;
    }
  } else if (!property.quantizable && is_input_quantized) {
    // If the tensor is quantized, we have to add a Dequantize op after
    // since this op is not quantizable.
    std::unique_ptr<TensorT> op_output;
    utils::MakeTensor(tensor->name + "_float", tensor->shape,
                      TensorType_FLOAT32, &op_output);
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
TfLiteStatus QuantizeOpOutput(ModelT* model, int32_t subgraph_idx,
                              int32_t op_idx,
                              operator_property::OperatorProperty property,
                              int32_t output_idx,
                              ErrorReporter* error_reporter) {
  // If the operator is not quantizable, we don't need to do anything for the
  // output.
  if (!property.quantizable) {
    return kTfLiteOk;
  }
  SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
  OperatorT* op = subgraph->operators[op_idx].get();
  const BuiltinOperator op_code =
      model->operator_codes[op->opcode_index]->builtin_code;
  if (output_idx >= op->outputs.size()) {
    error_reporter->Report(
        "Required output index %d is larger than the output length of "
        "op %s at index %d in subgraph %d",
        output_idx, op->outputs.size(), EnumNameBuiltinOperator(op_code),
        op_idx, subgraph_idx);
    return kTfLiteError;
  }

  TensorT* output_tensor = subgraph->tensors[op->outputs[output_idx]].get();
  if (ShouldRestrictSameInputOutputScale(property)) {
    // Copy quantization parameter. For average pool, max pool, etc
    // min/max can be different but we want them to be the same.
    // Get scale and zero point of input.
    if (property.input_indexes[0] >= op->inputs.size()) {
      error_reporter->Report(
          "Required input index %d is larger than the input length of "
          "op  %s at index %d in subgraph %d",
          property.input_indexes[0], op->inputs.size(),
          EnumNameBuiltinOperator(op_code), op_idx, subgraph_idx);
      return kTfLiteError;
    }
    const int input_index = op->inputs[property.input_indexes[0]];
    TensorT* input_tensor = subgraph->tensors[input_index].get();
    if (input_tensor->quantization->scale.size() != 1 ||
        input_tensor->quantization->zero_point.size() != 1 ||
        input_tensor->quantization->min.size() != 1 ||
        input_tensor->quantization->max.size() != 1) {
      error_reporter->Report(
          "Invalid quantization params for op %s at index %d "
          "in subgraph %d",
          EnumNameBuiltinOperator(op_code), op_idx, subgraph_idx);
      return kTfLiteError;
    }

    const float input_scale = input_tensor->quantization->scale[0];
    const int32_t input_zero_point = input_tensor->quantization->zero_point[0];

    const float min = input_tensor->quantization->min[0];
    const float max = input_tensor->quantization->max[0];
    if (utils::HasMinMax(output_tensor)) {
      if (output_tensor->quantization->min[0] != min ||
          output_tensor->quantization->max[0] != max) {
        printf(
            "Note the output min/max is different from the input min/max "
            "for op %s at index %d in subgraph %d. This is legal but "
            "should happens rarely.",
            EnumNameBuiltinOperator(op_code), op_idx, subgraph_idx);
      }
    }

    // Apply to output.
    output_tensor->quantization = absl::make_unique<QuantizationParametersT>();
    output_tensor->quantization->scale.push_back(input_scale);
    output_tensor->quantization->zero_point.push_back(input_zero_point);
    output_tensor->quantization->min.push_back(min);
    output_tensor->quantization->max.push_back(max);
    output_tensor->type = TensorType_INT8;
  } else if (property.restriction_on_output) {
    const auto scale_and_zp = property.restricted_value_on_output;
    // Apply to output.
    output_tensor->quantization = absl::make_unique<QuantizationParametersT>();
    output_tensor->quantization->scale.push_back(scale_and_zp.first);
    output_tensor->quantization->zero_point.push_back(scale_and_zp.second);
    output_tensor->type = TensorType_INT8;
  } else {
    // Process regular output that doesn't have any restrictions.
    if (utils::HasMinMax(output_tensor)) {
      utils::QuantizeActivation(output_tensor);
    } else {
      error_reporter->Report(
          "Unable to find min/max value for output %d in %s in "
          "subgraph %d, node: %d",
          output_idx, EnumNameBuiltinOperator(op_code), subgraph_idx, op_idx);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

// Quantize inputs and weights.
// Because of ops such as lstm, still need to do per op, instead of weights.
TfLiteStatus QuantizeWeightsInputOutput(ModelT* model, bool allow_float,
                                        ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          model->operator_codes[op->opcode_index]->builtin_code;
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(op_code);

      if (!property.quantizable && !allow_float) {
        error_reporter->Report("Quantization not yet supported for op: %s",
                               EnumNameBuiltinOperator(op_code));
        return kTfLiteError;
      }

      // Quantize operator inputs/weights.
      for (const int input_idx : GetInputIndexes(op, property)) {
        TF_LITE_ENSURE_STATUS(QuantizeOpInput(
            model, subgraph_idx, &op_idx, property, input_idx, error_reporter));
      }

      // Quantize operator outputs.
      for (const int output_idx : property.output_indexes) {
        TF_LITE_ENSURE_STATUS(QuantizeOpOutput(
            model, subgraph_idx, op_idx, property, output_idx, error_reporter));
      }
    }
  }
  return kTfLiteOk;
}

// Quantize bias.
TfLiteStatus QuantizeBiases(ModelT* model, ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          model->operator_codes[op->opcode_index]->builtin_code;
      operator_property::OperatorProperty property =
          operator_property::GetOperatorProperty(op_code);
      if (!property.quantizable) {
        continue;
      }
      for (const int bias_idx : property.biases) {
        if (bias_idx >= op->inputs.size()) {
          error_reporter->Report(
              "Required input index %d is larger than the input length of "
              "op  %s at index %d in subgraph %d",
              bias_idx, op->inputs.size(), EnumNameBuiltinOperator(op_code),
              op_idx, subgraph_idx);
          return kTfLiteError;
        }
        // Quantize if it is not quantized already as the
        // output of another op or input of another op.
        if (!utils::IsQuantized(subgraph, op->inputs[bias_idx])) {
          if (utils::HasBuffer(model, subgraph, op->inputs[bias_idx])) {
            TensorT* bias_tensor =
                subgraph->tensors[op->inputs[bias_idx]].get();
            if (property.input_indexes.size() != 2) {
              error_reporter->Report(
                  "Expect the input length of "
                  "op %s at index %d in subgraph %d to be 2",
                  bias_idx, op->inputs.size(), EnumNameBuiltinOperator(op_code),
                  op_idx, subgraph_idx);
              return kTfLiteError;
            }
            TensorT* input_tensor =
                subgraph->tensors[op->inputs[property.input_indexes[0]]].get();
            TensorT* weight_tensor =
                subgraph->tensors[op->inputs[property.input_indexes[1]]].get();
            TF_LITE_ENSURE_STATUS(QuantizeBias(
                model, input_tensor, weight_tensor, bias_tensor,
                property.per_axis, property.per_axis_index, error_reporter));
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
                           ErrorReporter* error_reporter) {
  TF_LITE_ENSURE_STATUS(
      QuantizeWeightsInputOutput(model, allow_float, error_reporter));
  TF_LITE_ENSURE_STATUS(ApplyConstraints(model, error_reporter));
  TF_LITE_ENSURE_STATUS(QuantizeBiases(model, error_reporter));
  utils::SetOperatorCodeVersion(model);
  TF_LITE_ENSURE_STATUS(
      SetInputAndOutputTypes(model, input_type, output_type, error_reporter));

  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(*builder, model);
  FinishModelBuffer(*builder, output_model_location);

  return kTfLiteOk;
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
