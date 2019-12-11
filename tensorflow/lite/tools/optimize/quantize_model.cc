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

// Gets the operator property from the operator_property list and additionally
// modifies the quantizable parameter based on the user's specified
// operator_names.
operator_property::OperatorProperty GetOperatorProperty(
    const std::unordered_set<string>& operator_names, const ModelT* model,
    int subgraph_index, int op_idx, const string& operator_name) {
  operator_property::OperatorProperty property =
      operator_property::GetOperatorProperty(model, subgraph_index, op_idx);
  const OperatorT* op =
      model->subgraphs[subgraph_index]->operators[op_idx].get();
  const BuiltinOperator op_code =
      model->operator_codes[op->opcode_index]->builtin_code;
  // The algorithm adds Dequantize and Quantize, so we don't require them to be
  // in the operator_names.
  if (op_code != BuiltinOperator_DEQUANTIZE &&
      op_code != BuiltinOperator_QUANTIZE) {
    property.quantizable =
        property.quantizable &&
        (operator_names.find(operator_name) != operator_names.end());
  }
  return property;
}

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
        weight_scales.data(), channel_dim_size, error_reporter);
  } else {
    if (weight_scales.size() != 1) {
      error_reporter->Report(
          "Expected per-layer weight scale dimension size 1, got %d",
          weight_scales.size());
      return kTfLiteError;
    }
    return utils::SymmetricPerLayerBiasQuantize(
        model, bias_tensor,
        input_tensor->quantization->scale[0] * weight_scales[0],
        error_reporter);
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
            "Unsupported output type %s for output tensor '%s' of type %s.",
            EnumNameTensorType(output_type), tensor->name.c_str(),
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
// We have made the restriction that for int8 quantized concat, minimum, and
// maximum, the inputs and outputs must have the same scale and zero point.
// The other ones with constraints are handled in QuantizeWeightsAndInput.
TfLiteStatus ApplyConstraints(ModelT* model,
                              const std::unordered_set<string>& operator_names,
                              ErrorReporter* error_reporter) {
  for (int subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    // Iterate backward to avoid messing with index.
    for (int op_idx = subgraph->operators.size() - 1; op_idx >= 0; op_idx--) {
      OperatorT* op = subgraph->operators[op_idx].get();
      operator_property::OperatorProperty property =
          GetOperatorProperty(operator_names, model, subgraph_idx, op_idx,
                              subgraph->tensors[op->outputs[0]]->name);
      if (!property.quantizable) {
        continue;
      }
      if (!property.arbitrary_inputs ||
          !property.restrict_same_input_output_scale) {
        continue;
      }
      // If ApplyConstraints and requant is needed, use the min of min and max
      // of max, which means using the scale and zero point of output.
      TensorT* output_tensor = subgraph->tensors[op->outputs[0]].get();
      if (!utils::QuantizationParametersExist(output_tensor)) {
        error_reporter->Report(
            "Unable to get scale or zero point from the tensor at %d.",
            op->outputs[0]);
        return kTfLiteError;
      }
      const float output_scale = output_tensor->quantization->scale[0];
      const float output_zp = output_tensor->quantization->zero_point[0];
      for (size_t input_idx = 0; input_idx < op->inputs.size(); ++input_idx) {
        TensorT* input_tensor = subgraph->tensors[op->inputs[input_idx]].get();
        if (!utils::QuantizationParametersExist(input_tensor)) {
          error_reporter->Report(
              "Unable to get scale or zero point from tensor at %d.",
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
    operator_property::OperatorProperty property) {
  // Ops with multiple inputs (i.e. concat) gets restricted in ApplyConstraints.
  return (!property.arbitrary_inputs &&
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
TfLiteStatus QuantizeOpInput(
    ModelT* model, int32_t subgraph_idx, size_t* op_idx,
    operator_property::OperatorProperty property,
    const std::pair<int32_t, operator_property::TensorProperty>& input,
    ErrorReporter* error_reporter) {
  int32_t input_idx = input.first;
  operator_property::TensorProperty tensor_property = input.second;
  SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
  OperatorT* op = subgraph->operators[*op_idx].get();
  const BuiltinOperator op_code =
      model->operator_codes[op->opcode_index]->builtin_code;
  if (input_idx >= op->inputs.size()) {
    error_reporter->Report(
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
  // Assumes op is quantized to int8.
  const bool is_input_quantized = utils::QuantizationParametersExist(tensor);
  if (property.quantizable && !is_input_quantized) {
    // The operation is quantizable, but the input isn't yet quantized.
    if (utils::HasBuffer(model, subgraph, tensor_idx)) {
      // TODO(suharshs): Look at consumers, throw error if one consumer is
      // per-channel and one per-layer.
      if (tensor_property.number_of_bits == 8) {
        if (tensor_property.use_derived_scale) {
          // Currently 8bit tensors in input do not accept derived scale.
          return kTfLiteError;
        }
        if (utils::QuantizeWeight(model, tensor, tensor_property.per_axis,
                                  tensor_property.per_axis_index,
                                  error_reporter) == kTfLiteError) {
          error_reporter->Report(
              "Unable to quantize buffer or min/max value for input %d "
              "in op %s in subgraph %d, node: %d",
              input_idx, EnumNameBuiltinOperator(op_code), subgraph_idx,
              *op_idx);
          return kTfLiteError;
        }
      } else if (tensor_property.number_of_bits == 16) {
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
        return utils::SymmetricPerLayerBiasQuantize(model, tensor, scale,
                                                    error_reporter);

      } else {
        // Only 8, 16, 32 are supported.
        // TODO(jianlijianli): extend this to support arbitrary bits.
        error_reporter->Report(
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
          utils::QuantizeActivation(tensor);
        } else if (tensor_property.number_of_bits == 16) {
          TensorT* tensor = subgraph->tensors[tensor_idx].get();
          float range = std::max(std::abs(tensor->quantization->min[0]),
                                 std::abs(tensor->quantization->max[0]));
          if (tensor_property.extend_to_power_of_two) {
            const int power_of_two_scale = utils::GetPowerOfTwoScale(
                tensor->quantization->min[0], tensor->quantization->max[0]);
            range = std::pow(2, power_of_two_scale);
          }
          const float quantized_range = 32768.0;
          const float scale = range / quantized_range;
          utils::QuantizeActivationToInt16(tensor, scale);
        }
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
          "%d in %s in subgraph %d, node: %d",
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
TfLiteStatus QuantizeOpOutput(
    ModelT* model, int32_t subgraph_idx, int32_t op_idx,
    operator_property::OperatorProperty property,
    const std::pair<int32_t, operator_property::TensorProperty>& output,
    ErrorReporter* error_reporter) {
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
  if (utils::QuantizationParametersExist(output_tensor)) {
    // Skip output if it has been quantized.
    return kTfLiteOk;
  }
  if (ShouldRestrictSameInputOutputScale(property)) {
    // Copy quantization parameter. For average pool, max pool, etc
    // min/max can be different but we want them to be the same.
    // Get scale and zero point of input.
    if (property.inputs[0].first >= op->inputs.size()) {
      error_reporter->Report(
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
      error_reporter->Report(
          "Invalid quantization params for op %s at index %d "
          "in subgraph %d",
          EnumNameBuiltinOperator(op_code), op_idx, subgraph_idx);
      return kTfLiteError;
    }

    const float input_scale = input_tensor->quantization->scale[0];
    const int32_t input_zero_point = input_tensor->quantization->zero_point[0];

    // Apply to output.
    output_tensor->quantization = absl::make_unique<QuantizationParametersT>();
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
    output_tensor->type = TensorType_INT8;
  } else if (tensor_property.restriction) {
    const auto scale_and_zp = tensor_property.restricted_value;
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

TfLiteStatus QuantizeIntemediateTensors(ModelT* model,
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
            model->operator_codes[op->opcode_index]->builtin_code;
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
            if (utils::HasMinMax(tensor)) {
              utils::QuantizeActivation(tensor);
            } else {
              error_reporter->Report(
                  "Unable to find min/max value for output %d in %s in "
                  "subgraph %d, node: %d",
                  tensor, EnumNameBuiltinOperator(op_code), subgraph_idx,
                  op_idx);
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

// Quantize tensros that have shared range. For example, in LSTM, the output
// tensor and input state tensor should share the same range because they are
// using the same scale and zero point.
// We have to model this explicitely because the output is modeled as an extra
// tensor in LSTM. In calibrator, state tensors are logged both before and after
// the inferece so the range is fully captured. But output, although it is
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
          // Currently only support pair of twos.
          // TODO(jianlijianli): extend to arbitrary number of tensors.
          if (input.size() != 2) {
            return kTfLiteError;
          }
          const int index_1 = input[0];
          const int index_2 = input[1];
          // TODO(jianlijianli): model input/output.
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
              absl::make_unique<QuantizationParametersT>();
          utils::GetAsymmetricQuantizationParams(
              min_of_min, max_of_max, -128, 127, quantization_params.get());

          // Populate both tensors with the same parameters.
          const float scale = quantization_params->scale[0];
          const int32 zero_point = quantization_params->zero_point[0];
          for (TensorT* tensor : {tensor_1, tensor_2}) {
            tensor->quantization = absl::make_unique<QuantizationParametersT>();
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

// Quantize inputs and weights.
// Because of ops such as lstm, still need to do per op, instead of weights.
TfLiteStatus QuantizeWeightsInputOutput(
    ModelT* model, bool allow_float,
    const std::unordered_set<string>& operator_names,
    ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          model->operator_codes[op->opcode_index]->builtin_code;
      operator_property::OperatorProperty property =
          GetOperatorProperty(operator_names, model, subgraph_idx, op_idx,
                              subgraph->tensors[op->outputs[0]]->name);

      if (!property.quantizable && !allow_float) {
        error_reporter->Report("Quantization not yet supported for op: %s",
                               EnumNameBuiltinOperator(op_code));
        return kTfLiteError;
      }

      // Quantize operator inputs/weights.
      for (const std::pair<int, operator_property::TensorProperty>& input :
           GetInputs(op, property)) {
        TF_LITE_ENSURE_STATUS(QuantizeOpInput(model, subgraph_idx, &op_idx,
                                              property, input, error_reporter));
      }

      // Quantize operator outputs.
      for (const std::pair<int, operator_property::TensorProperty>& output :
           GetOutputs(op, property)) {
        TF_LITE_ENSURE_STATUS(QuantizeOpOutput(
            model, subgraph_idx, op_idx, property, output, error_reporter));
      }
    }
  }
  return kTfLiteOk;
}

// Quantize bias.
TfLiteStatus QuantizeBiases(ModelT* model,
                            const std::unordered_set<string>& operator_names,
                            ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      const BuiltinOperator op_code =
          model->operator_codes[op->opcode_index]->builtin_code;
      operator_property::OperatorProperty property =
          GetOperatorProperty(operator_names, model, subgraph_idx, op_idx,
                              subgraph->tensors[op->outputs[0]]->name);
      if (!property.quantizable) {
        continue;
      }
      for (const int bias_idx : property.biases) {
        if (op->inputs[bias_idx] == kTfLiteOptionalTensor) {
          continue;
        }
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
        TensorT* bias_tensor = subgraph->tensors[op->inputs[bias_idx]].get();
        if (!utils::QuantizationParametersExist(bias_tensor)) {
          if (utils::HasBuffer(model, subgraph, op->inputs[bias_idx])) {
            if (property.inputs.size() != 2) {
              error_reporter->Report(
                  "Expect the input length of "
                  "op %s at index %d in subgraph %d to be 2",
                  bias_idx, op->inputs.size(), EnumNameBuiltinOperator(op_code),
                  op_idx, subgraph_idx);
              return kTfLiteError;
            }
            TensorT* input_tensor =
                subgraph->tensors[op->inputs[property.inputs[0].first]].get();
            TensorT* weight_tensor =
                subgraph->tensors[op->inputs[property.inputs[1].first]].get();
            operator_property::TensorProperty weight_property =
                property.inputs[1].second;
            TF_LITE_ENSURE_STATUS(
                QuantizeBias(model, input_tensor, weight_tensor, bias_tensor,
                             weight_property.per_axis,
                             weight_property.per_axis_index, error_reporter));
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
    ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      operator_property::OperatorProperty property =
          GetOperatorProperty(operator_names, model, subgraph_idx, op_idx,
                              subgraph->tensors[op->outputs[0]]->name);

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
            tensor->quantization = absl::make_unique<QuantizationParametersT>();
          }

          // Fill per channel max and min with respect to channel_dim_index.
          if (input.second.per_axis) {
            if (tensor->shape.size() == 4) {
              int32_t channel_dim_index = input.second.per_axis_index;
              TF_LITE_ENSURE_STATUS(utils::FillPerChannelMinMax(
                  float_input_data, tensor->shape, channel_dim_index,
                  tensor->quantization.get(), error_reporter));
            } else {
              error_reporter->Report(
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
            error_reporter->Report(
                "Quantized dimension for tensor property and quantization "
                "parameters do not match. Got %d and %d respectively.",
                input.second.per_axis_index,
                tensor->quantization->quantized_dimension);
            return kTfLiteError;
          }

          // Dynamic tensor.
        } else if (!utils::HasMinMax(tensor) &&
                   !utils::HasBuffer(model, subgraph, tensor_idx)) {
          error_reporter->Report(
              "Max and min for dynamic tensors should be"
              " recorded during calibration");
          return kTfLiteError;
        }

        if (utils::QuantizationParametersExist(tensor)) {
          error_reporter->Report(
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
    ErrorReporter* error_reporter) {
  for (size_t subgraph_idx = 0; subgraph_idx < model->subgraphs.size();
       subgraph_idx++) {
    SubGraphT* subgraph = model->subgraphs.at(subgraph_idx).get();
    for (size_t op_idx = 0; op_idx < subgraph->operators.size(); op_idx++) {
      OperatorT* op = subgraph->operators[op_idx].get();
      operator_property::OperatorProperty property =
          GetOperatorProperty(operator_names, model, subgraph_idx, op_idx,
                              subgraph->tensors[op->outputs[0]]->name);

      // Loop over all bias tensors.
      for (const int bias_idx : property.biases) {
        if (op->inputs[bias_idx] == kTfLiteOptionalTensor) {
          continue;
        }
        TensorT* bias_tensor = subgraph->tensors[op->inputs[bias_idx]].get();
        int32_t channel_dim_size = bias_tensor->shape[0];
        if (bias_tensor->shape.size() != 1) {
          error_reporter->Report("Expected bias tensor to be a vector.");
          return kTfLiteError;
        }

        if (property.inputs.size() != 2) {  // Only works for two input tensors.
          error_reporter->Report(
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
            error_reporter->Report(
                "Input tensor missing quantization information. Should be "
                "populated during calibration.");
            return kTfLiteError;
          }

          // Get input scale for assymmetric quantization.
          QuantizationParametersT temp_quant_params = QuantizationParametersT();
          utils::GetAsymmetricQuantizationParams(
              input_tensor->quantization->min[0],
              input_tensor->quantization->max[0],
              std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max(), &temp_quant_params);
          if (temp_quant_params.scale.size() != 1) {
            error_reporter->Report("Unexpected input quantization scale size.");
            return kTfLiteError;
          }
          float input_scale = temp_quant_params.scale[0];

          // Check that max/min values have been filled for weights.
          if (!utils::HasMinMax(weight_tensor)) {
            error_reporter->Report(
                "Min and/or max values have not been recorded for weight "
                "tensor. This should have happened in FillQuantizationParams.");
            return kTfLiteError;
          }

          // Ensure the tensor dimensions are compatible.
          if (weight_property.per_axis) {
            if (bias_tensor->shape[0] !=
                weight_tensor->shape[weight_property.per_axis_index]) {
              error_reporter->Report(
                  "Channel mismatch between bias and weight tensors %d vs %d",
                  bias_tensor->shape[0],
                  weight_tensor->shape[weight_property.per_axis_index]);
              return kTfLiteError;
            }
            // Ensure that the number of max/mins matches the channel_dim_size.
            if (weight_tensor->quantization->max.size() != channel_dim_size) {
              error_reporter->Report(
                  "Mismatch between number of weight maxs and channels: %d vs "
                  "%d",
                  weight_tensor->quantization->max.size(), channel_dim_size);
              return kTfLiteError;
            }
            if (weight_tensor->quantization->min.size() != channel_dim_size) {
              error_reporter->Report(
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
            error_reporter->Report(
                "Scale and zero points should not be recorded for the weight "
                "tensor before quantization.");
            return kTfLiteError;
          }
          if (utils::QuantizationParametersExist(input_tensor)) {
            error_reporter->Report(
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
                           ErrorReporter* error_reporter) {
  TF_LITE_ENSURE_STATUS(
      FillQuantizationParams(model, operator_names, error_reporter));
  TF_LITE_ENSURE_STATUS(
      EnsureBiasScaleCompatibility(model, operator_names, error_reporter));
  TF_LITE_ENSURE_STATUS(QuantizeIntemediateTensors(model, error_reporter));
  TF_LITE_ENSURE_STATUS(QuantizeSharedRange(model, error_reporter));
  TF_LITE_ENSURE_STATUS(QuantizeWeightsInputOutput(
      model, allow_float, operator_names, error_reporter));
  TF_LITE_ENSURE_STATUS(
      ApplyConstraints(model, operator_names, error_reporter));
  TF_LITE_ENSURE_STATUS(QuantizeBiases(model, operator_names, error_reporter));
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
                           const TensorType& output_type, bool allow_float,
                           ErrorReporter* error_reporter) {
  return QuantizeModel(builder, model, input_type, output_type, allow_float,
                       GetAllOperatorOutputs(model), error_reporter);
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
