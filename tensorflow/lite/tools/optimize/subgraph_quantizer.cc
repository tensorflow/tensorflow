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
#include "tensorflow/lite/tools/optimize/subgraph_quantizer.h"

#include <algorithm>
#include <limits>

#include "flatbuffers/flexbuffers.h"
#include "absl/memory/memory.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/quantization_utils.h"
#include "tensorflow/lite/tools/optimize/symmetric_per_channel_params.h"

namespace tflite {
namespace optimize {
namespace internal {

namespace {
const int8_t kMinQuantizedValue = -127;
const int8_t kMaxQuantizedValue = 127;

TfLiteStatus AddQuantizationParams(const SymmetricPerChannelParams& params,
                                   const uint8_t* buffer_data,
                                   size_t buffer_size, TensorType output_type,
                                   ModelT* model, TensorT* tensor) {
  params.AddToTensor(tensor);
  model->buffers[tensor->buffer]->data.assign(buffer_data,
                                              buffer_data + buffer_size);
  // Update the tensor type.
  tensor->type = output_type;
  return kTfLiteOk;
}

bool OpHasOptionalBiasTensor(BuiltinOperator op_code) {
  return op_code == BuiltinOperator_CONV_2D ||
         op_code == BuiltinOperator_DEPTHWISE_CONV_2D;
}

struct OpWithBiasTensors {
  int activation_input_index;
  int bias_input_index;
  int weights_input_index;
  int index_for_channel_in_weights;
};

const OpWithBiasTensors* GetInfoForOpWithBiasTensor(BuiltinOperator op_code) {
  if (op_code == BuiltinOperator_CONV_2D) {
    static OpWithBiasTensors op_info = {.activation_input_index = 0,
                                        .weights_input_index = 1,
                                        .bias_input_index = 2,
                                        .index_for_channel_in_weights = 0};
    return &op_info;
  }
  if (op_code == BuiltinOperator_DEPTHWISE_CONV_2D) {
    static OpWithBiasTensors op_info = {.activation_input_index = 0,
                                        .weights_input_index = 1,
                                        .bias_input_index = 2,
                                        .index_for_channel_in_weights = 3};
    return &op_info;
  }

  return nullptr;
}

// Symmetrically Quantizes the given tensor as int8 values.
TfLiteStatus SymmetricPerChannelQuantizeTensor(ModelT* model, TensorT* tensor,
                                               int32_t channel_dim_index,
                                               ErrorReporter* error_reporter) {
  int32_t channel_dim_size = tensor->shape[channel_dim_index];
  if (tensor->shape.size() != 4) {
    error_reporter->Report("Only dims=4 is supported, tensor dims: %d",
                           tensor->shape.size());
    return kTfLiteError;
  }

  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(utils::NumElements(*tensor, &num_elements));
  const uint64_t num_elements_per_channel = num_elements / channel_dim_size;

  if (tensor->quantization == nullptr) {
    tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }

  std::vector<float> min_vals(channel_dim_size);
  std::vector<float> max_vals(channel_dim_size);
  std::vector<bool> has_min_max_value(channel_dim_size, false);
  int indices[4];
  RuntimeShape tensor_dims{tensor->shape[0], tensor->shape[1], tensor->shape[2],
                           tensor->shape[3]};
  BufferT* buffer = model->buffers[tensor->buffer].get();
  float* float_data = reinterpret_cast<float*>(buffer->data.data());

  // Compute min max ranges per channel
  for (indices[0] = 0; indices[0] < tensor->shape[0]; indices[0]++) {
    for (indices[1] = 0; indices[1] < tensor->shape[1]; indices[1]++) {
      for (indices[2] = 0; indices[2] < tensor->shape[2]; indices[2]++) {
        for (indices[3] = 0; indices[3] < tensor->shape[3]; indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          const float val = float_data[Offset(tensor_dims, indices)];
          if (has_min_max_value[channel_idx]) {
            if (min_vals[channel_idx] > val) {
              min_vals[channel_idx] = val;
            } else if (max_vals[channel_idx] < val) {
              max_vals[channel_idx] = val;
            }
          } else {
            min_vals[channel_idx] = val;
            max_vals[channel_idx] = val;
            has_min_max_value[channel_idx] = true;
          }
        }
      }
    }
  }

  // Calculate scales per channel
  std::vector<float> scales(channel_dim_size);
  std::vector<float> scale_invs(channel_dim_size);
  const float half_scale = kMaxQuantizedValue;
  for (size_t channel_idx = 0; channel_idx < channel_dim_size; channel_idx++) {
    const float half_range = std::max(std::abs(min_vals[channel_idx]),
                                      std::abs(max_vals[channel_idx]));
    scales[channel_idx] = half_range / half_scale;
    if (half_range == 0) {
      scale_invs[channel_idx] = 0;
    } else {
      scale_invs[channel_idx] = half_scale / half_range;
    }
  }

  // Quantize the values.
  std::vector<int8_t> quantized_buffer(num_elements_per_channel);
  std::vector<int8_t> final_buffer(num_elements);
  memset(indices, 0, 4 * sizeof(int));
  for (indices[0] = 0; indices[0] < tensor->shape[0]; indices[0]++) {
    for (indices[1] = 0; indices[1] < tensor->shape[1]; indices[1]++) {
      for (indices[2] = 0; indices[2] < tensor->shape[2]; indices[2]++) {
        for (indices[3] = 0; indices[3] < tensor->shape[3]; indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          int index = Offset(tensor_dims, indices);
          const float val = float_data[index];
          const int32_t quantized_value =
              static_cast<int32_t>(TfLiteRound(val * scale_invs[channel_idx]));
          final_buffer[index] = std::min<int8_t>(
              kMaxQuantizedValue,
              std::max<int8_t>(kMinQuantizedValue, quantized_value));
        }
      }
    }
  }

  // Set the buffers and output type.
  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(final_buffer.data());
  size_t buffer_size = num_elements * sizeof(int8_t);
  SymmetricPerChannelParams symmetric_params(scales, channel_dim_index);
  return AddQuantizationParams(symmetric_params, uint8_buffer, buffer_size,
                               TensorType_INT8, model, tensor);
}

// Symmetrically quantizes the bias for ops like Conv and DepthwiseConv.
// The scale of bias if weight_per_channel_scale[channel] * input_scale
TfLiteStatus SymmetricPerChannelBiasQuantize(const TensorT* input_tensor,
                                             const TensorT* weight_tensor,
                                             int channel_dim_index,
                                             ModelT* model, TensorT* tensor,
                                             ErrorReporter* error_reporter) {
  if (tensor->shape.size() != 1) {
    error_reporter->Report("Expected bias tensor shape to be 1.");
    return kTfLiteError;
  }

  if (tensor->type != TensorType_FLOAT32) {
    return kTfLiteOk;
  }

  // TODO(shashishekhar): Make this support scalar biases.
  if (tensor->shape[0] != weight_tensor->shape[channel_dim_index]) {
    error_reporter->Report(
        "Channel mismatch between bias and weight tensors %d vs %d",
        tensor->shape[0], weight_tensor->shape[channel_dim_index]);
    return kTfLiteError;
  }
  int32_t channel_dim_size = tensor->shape[0];
  if (!input_tensor->quantization ||
      input_tensor->quantization->scale.size() != 1) {
    error_reporter->Report("Input tensor missing quantization information");
    return kTfLiteError;
  }
  std::unique_ptr<SymmetricPerChannelParams> weight_params;
  TF_LITE_ENSURE_STATUS(SymmetricPerChannelParams::ReadFromTensor(
      *weight_tensor, &weight_params));
  const auto& weight_scales = weight_params->scales();

  if (weight_scales.size() != channel_dim_size) {
    error_reporter->Report("Mismatch weight scale dimension: %d",
                           weight_scales.size());
    return kTfLiteError;
  }

  // Compute scales.
  std::vector<float> scales(channel_dim_size);
  for (size_t i = 0; i < channel_dim_size; i++) {
    scales[i] = input_tensor->quantization->scale[0] * weight_scales[i];
  }

  BufferT* buffer = model->buffers[tensor->buffer].get();
  float* float_data = reinterpret_cast<float*>(buffer->data.data());
  uint64_t num_elements;
  TF_LITE_ENSURE_STATUS(utils::NumElements(*tensor, &num_elements));

  if (tensor->quantization == nullptr) {
    tensor->quantization = absl::make_unique<QuantizationParametersT>();
  }
  std::vector<int32_t> final_buffer(num_elements);
  const int32_t kScale = std::numeric_limits<int32_t>::max();

  for (int32_t channel_idx = 0; channel_idx < channel_dim_size; channel_idx++) {
    float scaling_factor = scales[channel_idx];
    float scaling_factor_inv = (scaling_factor == 0) ? 0 : 1.0 / scaling_factor;
    const int32_t quantized_value = static_cast<int32_t>(
        TfLiteRound(float_data[channel_idx] * scaling_factor_inv));
    final_buffer[channel_idx] =
        std::min(kScale, std::max(-kScale, quantized_value));
  }

  // Set the buffers and output type.
  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(final_buffer.data());
  size_t buffer_size = num_elements * sizeof(int32_t);
  SymmetricPerChannelParams symmetric_params(scales, channel_dim_index);
  return AddQuantizationParams(symmetric_params, uint8_buffer, buffer_size,
                               TensorType_INT32, model, tensor);
}
}  // namespace

TfLiteStatus SubgraphQuantizer::AsymmetricQuantizeTensor(
    BuiltinOperator op_code, int32_t tensor_idx) {
  TensorT* tensor = subgraph_->tensors[tensor_idx].get();
  if (tensor->type != TensorType_FLOAT32) {
    return kTfLiteOk;
  }

  if (model_->buffers[tensor->buffer]->data.data() != nullptr) {
    return kTfLiteError;
  }
  if (!tensor->quantization || tensor->quantization->min.empty() ||
      tensor->quantization->max.empty()) {
    error_reporter_->Report(
        "Missing required min/max information for tensor_idx %d of operation: "
        "%s",
        tensor_idx, EnumNameBuiltinOperator(op_code));
    return kTfLiteError;
  }
  utils::GetAsymmetricQuantizationParams(
      tensor->quantization->min[0], tensor->quantization->max[0],
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
      tensor->quantization.get());
  tensor->type = TensorType_INT8;
  return kTfLiteOk;
}

TfLiteStatus SubgraphQuantizer::QuantizeOpWithBias(BuiltinOperator op_code,
                                                   OperatorT* op) {
  auto op_tensor_info = GetInfoForOpWithBiasTensor(op_code);
  if (!op_tensor_info) {
    error_reporter_->Report("Cannot quantize op: %s",
                            EnumNameBuiltinOperator(op_code));
    return kTfLiteError;
  }

  // Conv/Depthwise conv have 2 inputs when there is no bias, 3 otherwise.
  if (op->inputs.size() != 2 && op->inputs.size() != 3) {
    return kTfLiteError;
  }
  auto input_tensor_idx = op->inputs[op_tensor_info->activation_input_index];
  if (IsSubgraphInput(input_tensor_idx)) {
    TF_LITE_ENSURE_STATUS(AsymmetricQuantizeTensor(op_code, input_tensor_idx));
  }
  auto weights_tensor_idx = op->inputs[op_tensor_info->weights_input_index];

  TensorT* weights_tensor = subgraph_->tensors[weights_tensor_idx].get();
  int weights_channel_index = op_tensor_info->index_for_channel_in_weights;

  auto status = SymmetricPerChannelQuantizeTensor(
      model_, weights_tensor, weights_channel_index, error_reporter_);
  TF_LITE_ENSURE_STATUS(status);

  // If there is bias, quantize it.
  if (op->inputs.size() == 3) {
    auto bias_tensor_idx = op->inputs[op_tensor_info->bias_input_index];
    const TensorT* input_tensor = subgraph_->tensors[input_tensor_idx].get();
    TensorT* bias_tensor = subgraph_->tensors[bias_tensor_idx].get();
    TF_LITE_ENSURE_STATUS(SymmetricPerChannelBiasQuantize(
        input_tensor, weights_tensor, weights_channel_index, model_,
        bias_tensor, error_reporter_));
  }

  if (op->outputs.size() != 1) {
    return kTfLiteError;
  }
  auto output_tensor_idx = op->outputs[0];
  TF_LITE_ENSURE_STATUS(AsymmetricQuantizeTensor(op_code, output_tensor_idx));

  return kTfLiteOk;
}

TfLiteStatus SubgraphQuantizer::PropagateMinMaxForAvgAndMaxPool(
    BuiltinOperator op_code, OperatorT* op) {
  TF_LITE_ENSURE_EQ(this->error_reporter_, op->inputs.size(), 1);

  if (IsSubgraphInput(op->inputs[0])) {
    TF_LITE_ENSURE_STATUS(AsymmetricQuantizeTensor(op_code, op->inputs[0]));
  }

  auto output_tensor = subgraph_->tensors[op->outputs[0]].get();
  if (output_tensor->type != TensorType_FLOAT32) {
    return kTfLiteOk;
  }
  auto input_tensor = subgraph_->tensors[op->inputs[0]].get();
  if (!input_tensor->quantization) {
    error_reporter_->Report(
        "Missing required min/max information for input of operation: %s",
        EnumNameBuiltinOperator(op_code));
    return kTfLiteError;
  }
  if (input_tensor->quantization->min.size() != 1 ||
      input_tensor->quantization->max.size() != 1 ||
      input_tensor->quantization->scale.size() != 1 ||
      input_tensor->quantization->zero_point.size() != 1) {
    error_reporter_->Report(
        "Invalid quantization information for Op: %s, tensor: %s",
        EnumNameBuiltinOperator(op_code), input_tensor->name.c_str());
    return kTfLiteError;
  }
  auto quant_params = absl::make_unique<QuantizationParametersT>();
  // Nudge min, max to include the floating point zero.
  const float min = std::min(0.f, input_tensor->quantization->min[0]);
  const float max = std::max(0.f, input_tensor->quantization->max[0]);
  quant_params->min.push_back(min);
  quant_params->max.push_back(max);
  quant_params->scale.push_back(input_tensor->quantization->scale[0]);
  quant_params->zero_point.push_back(input_tensor->quantization->zero_point[0]);
  // TODO(shashishekhar): Log a warning here if overriding existing
  // min/max/scales differ from input scales.
  output_tensor->quantization = std::move(quant_params);
  output_tensor->type = TensorType_INT8;
  return kTfLiteOk;
}

TfLiteStatus SubgraphQuantizer::AsymmetricQuantizeSingleInputOutputOp(
    BuiltinOperator op_code, OperatorT* op) {
  TF_LITE_ENSURE_EQ(this->error_reporter_, op->inputs.size(), 1);
  TF_LITE_ENSURE_EQ(this->error_reporter_, op->outputs.size(), 1);

  if (IsSubgraphInput(op->inputs[0])) {
    TF_LITE_ENSURE_STATUS(AsymmetricQuantizeTensor(op_code, op->inputs[0]));
  }

  auto output_tensor = subgraph_->tensors[op->outputs[0]].get();
  if (output_tensor->type != TensorType_FLOAT32) {
    return kTfLiteOk;
  }
  auto quant_params = absl::make_unique<QuantizationParametersT>();
  TF_LITE_ENSURE_STATUS(AsymmetricQuantizeTensor(op_code, op->outputs[0]));
  return kTfLiteOk;
}

bool SubgraphQuantizer::IsSubgraphInput(int32_t tensor_idx) const {
  return std::find(subgraph_->inputs.begin(), subgraph_->inputs.end(),
                   tensor_idx) != subgraph_->inputs.end();
}

TfLiteStatus SubgraphQuantizer::QuantizeOperator(int op_idx) {
  OperatorT* op = subgraph_->operators[op_idx].get();
  const BuiltinOperator op_code =
      model_->operator_codes[op->opcode_index]->builtin_code;
  if (OpHasOptionalBiasTensor(op_code)) {
    return QuantizeOpWithBias(op_code, op);
  }
  switch (op_code) {
    case BuiltinOperator_AVERAGE_POOL_2D:
    case BuiltinOperator_MAX_POOL_2D:
      return PropagateMinMaxForAvgAndMaxPool(op_code, op);
    case BuiltinOperator_SQUEEZE:
    case BuiltinOperator_SOFTMAX:
      return AsymmetricQuantizeSingleInputOutputOp(op_code, op);
    default:
      return kTfLiteError;
  }

  return kTfLiteError;
}

}  // namespace internal
}  // namespace optimize
}  // namespace tflite
