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
// This file is the MLIR copy of part of
// third_party/tensorflow/lite/tools/optimize/quantization_utils.cc as part of
// the effort to decouple TFLite from MLIR.

#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/quantization_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/compiler/mlir/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/model_utils.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy/portable_tensor_utils.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {
namespace toco_legacy {

namespace {

// LINT.IfChange(QuantizationUtilsConstants)
const int8_t kMinQuantizedValue8bit = -127;
const int8_t kMaxQuantizedValue8bit = 127;

const int8_t kMinQuantizedValue4bit = -7;
const int8_t kMaxQuantizedValue4bit = 7;

// The maximum number of dimensions supported in per-channel quantization.
constexpr int kPerChannelMaxDim = 4;
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:QuantizationUtilsConstants)
}  // namespace

using absl::InternalError;
using mlir::RuntimeShape;
using tflite::BufferT;
using tflite::QuantizationParametersT;
using tflite::TensorT;
using tflite::TensorType;
using tflite::TensorType_INT8;

// LINT.IfChange(NumElements)
absl::Status NumElements(const TensorT& tensor, uint64_t* num_elements) {
  *num_elements = 1;
  for (const int64_t dim : tensor.shape) {
    if (dim <= 0 || *num_elements > UINT64_MAX / static_cast<uint64_t>(dim)) {
      return InternalError("Invalid tensor shape.");
    }
    *num_elements *= dim;
  }
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:NumElements)

// LINT.IfChange(FillPerChannelMinMax)
absl::Status FillPerChannelMinMax(
    const float* const input, const std::vector<int32_t>& dimension,
    int32_t channel_dim_index, QuantizationParametersT* quantization_params) {
  if (!quantization_params->min.empty() || !quantization_params->max.empty()) {
    return absl::InvalidArgumentError(
        "Min or max already present in tensor quantization params.");
  }

  if (dimension.size() > kPerChannelMaxDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected tensor with less than %d dimensions, but got %d.",
        kPerChannelMaxDim + 1, dimension.size()));
  }
  if (channel_dim_index >= dimension.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected channel_dim_index to be less than %d, but got %d.",
        dimension.size(), channel_dim_index));
  }

  const int32_t channel_dim_size = dimension[channel_dim_index];
  quantization_params->quantized_dimension = channel_dim_index;
  quantization_params->min = std::vector<float>(channel_dim_size);
  quantization_params->max = std::vector<float>(channel_dim_size);
  std::vector<bool> has_min_max_value(channel_dim_size, false);
  int indices[kPerChannelMaxDim];
  RuntimeShape unextended_tensor_dims(dimension.size(), dimension.data());
  RuntimeShape tensor_dims =
      RuntimeShape::ExtendedShape(kPerChannelMaxDim, unextended_tensor_dims);
  channel_dim_index +=
      kPerChannelMaxDim - unextended_tensor_dims.DimensionsCount();

  // Compute min max ranges per channel
  for (indices[0] = 0; indices[0] < tensor_dims.Dims(0); indices[0]++) {
    for (indices[1] = 0; indices[1] < tensor_dims.Dims(1); indices[1]++) {
      for (indices[2] = 0; indices[2] < tensor_dims.Dims(2); indices[2]++) {
        for (indices[3] = 0; indices[3] < tensor_dims.Dims(3); indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          const float val = input[Offset(tensor_dims, indices)];
          if (has_min_max_value[channel_idx]) {
            if (quantization_params->min[channel_idx] > val) {
              quantization_params->min[channel_idx] = val;
            } else if (quantization_params->max[channel_idx] < val) {
              quantization_params->max[channel_idx] = val;
            }
          } else {
            quantization_params->min[channel_idx] = val;
            quantization_params->max[channel_idx] = val;
            has_min_max_value[channel_idx] = true;
          }
        }
      }
    }
  }
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:FillPerChannelMinMax)

// LINT.IfChange(SymmetricPerChannelQuantization)
// Per-channel quantize a tensor at the given index and fills both scales and
// quantized values.
absl::Status SymmetricPerChannelQuantization(
    TensorT* tensor, const float* const input, int32_t channel_dim_index,
    std::vector<float>* output_scales, std::vector<int8_t>* output_value) {
  if (tensor == nullptr) {
    return absl::InvalidArgumentError("Cannot quantize. Tensor is null.");
  }
  const int32_t channel_dim_size = tensor->shape[channel_dim_index];
  // Fill per channel max and min values if needed
  if (tensor->quantization == nullptr) {
    tensor->quantization = std::make_unique<QuantizationParametersT>();
  }
  if (!HasMinMax(tensor)) {
    absl::Status status = FillPerChannelMinMax(
        input, tensor->shape, channel_dim_index, tensor->quantization.get());
    if (!status.ok()) {
      return status;
    }
  }

  // Calculate scales per channel using max and min values from tensor.
  std::vector<float> scale_invs(channel_dim_size);
  const float half_scale = kMaxQuantizedValue8bit;
  for (int channel_idx = 0; channel_idx < channel_dim_size; channel_idx++) {
    const float half_range =
        std::max(std::abs(tensor->quantization->min[channel_idx]),
                 std::abs(tensor->quantization->max[channel_idx]));
    output_scales->at(channel_idx) = half_range / half_scale;
    if (half_range == 0) {
      scale_invs[channel_idx] = 0;
    } else {
      scale_invs[channel_idx] = half_scale / half_range;
    }
  }

  // Quantize the input values.
  SymmetricPerChannelQuantizeValues(input, scale_invs, tensor->shape,
                                    channel_dim_index, output_value);
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:SymmetricPerChannelQuantization)

// LINT.IfChange(SymmetricPerChannelQuantizeValues)
void SymmetricPerChannelQuantizeValues(const float* const input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int32_t>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value) {
  // Quantize the values.
  int indices[kPerChannelMaxDim];
  RuntimeShape unextended_tensor_dims(dimension.size(), dimension.data());
  RuntimeShape tensor_dims =
      RuntimeShape::ExtendedShape(kPerChannelMaxDim, unextended_tensor_dims);
  channel_dim_index +=
      kPerChannelMaxDim - unextended_tensor_dims.DimensionsCount();
  for (indices[0] = 0; indices[0] < tensor_dims.Dims(0); indices[0]++) {
    for (indices[1] = 0; indices[1] < tensor_dims.Dims(1); indices[1]++) {
      for (indices[2] = 0; indices[2] < tensor_dims.Dims(2); indices[2]++) {
        for (indices[3] = 0; indices[3] < tensor_dims.Dims(3); indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          int index = Offset(tensor_dims, indices);
          const float val = input[index];
          const int32_t quantized_value =
              static_cast<int32_t>(round(val * scales_inv[channel_idx]));
          output_value->at(index) = std::min<int8_t>(
              kMaxQuantizedValue8bit,
              std::max<int8_t>(kMinQuantizedValue8bit, quantized_value));
        }
      }
    }
  }
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:SymmetricPerChannelQuantizeValues)

// LINT.IfChange(SymmetricQuantizeTensor)
absl::Status SymmetricQuantizeTensor(ModelT* model, TensorT* tensor) {
  if (model == nullptr || tensor == nullptr) {
    return absl::InvalidArgumentError("No tensor to quantize.");
  }

  BufferT* buffer = model->buffers[tensor->buffer].get();
  if (buffer == nullptr) {
    return absl::InvalidArgumentError("Missing buffer.");
  }
  const float* float_data = reinterpret_cast<const float*>(buffer->data.data());
  uint64_t num_elements;
  absl::Status status = NumElements(*tensor, &num_elements);
  if (!status.ok()) {
    return status;
  }

  std::vector<int8_t> quantized_buffer;
  quantized_buffer.resize(num_elements);

  float min_value, max_value, scaling_factor;
  mlir::lite::toco_legacy::PortableSymmetricQuantizeFloats(
      float_data, num_elements, quantized_buffer.data(), &min_value, &max_value,
      &scaling_factor);

  if (tensor->quantization == nullptr) {
    tensor->quantization = std::make_unique<QuantizationParametersT>();
  }
  tensor->quantization->scale = std::vector<float>(1, scaling_factor);
  tensor->quantization->zero_point = std::vector<int64_t>(1, 0);

  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(quantized_buffer.data());
  model->buffers[tensor->buffer]->data.assign(uint8_buffer,
                                              uint8_buffer + num_elements);

  // Update the tensor type.
  tensor->type = TensorType_INT8;

  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:SymmetricQuantizeTensor)

// LINT.IfChange(QuantizeTensorFloat16)
absl::Status QuantizeTensorFloat16(ModelT* model, TensorT* tensor) {
  if (model == nullptr || tensor == nullptr) {
    return absl::InvalidArgumentError("No tensor to quantize.");
  }

  BufferT* buffer = model->buffers[tensor->buffer].get();
  if (buffer == nullptr) {
    return absl::InvalidArgumentError("Missing buffer.");
  }

  uint64_t num_elements;
  absl::Status status = NumElements(*tensor, &num_elements);
  if (!status.ok()) {
    return status;
  }

  // Copy single byte buffer data to float vector to guard against misalignment.
  std::vector<float> float_vector(num_elements);
  uint8_t* first = buffer->data.data();
  std::copy(first, first + buffer->data.size(),
            reinterpret_cast<uint8_t*>(float_vector.data()));

  // Transform float data to float16.
  std::vector<Eigen::half> quantized_buffer;
  quantized_buffer.resize(num_elements);
  constexpr float kMaxFloat16Value = 65504.f;
  constexpr float kMinFloat16Value = -65504.f;
  std::transform(float_vector.begin(), float_vector.end(),
                 quantized_buffer.begin(), [=](float a) {
                   float clamped = std::min(std::max(a, kMinFloat16Value),
                                            kMaxFloat16Value);
                   return static_cast<Eigen::half>(clamped);
                 });

  char* half_buffer = reinterpret_cast<char*>(quantized_buffer.data());
  model->buffers[tensor->buffer]->data.assign(
      half_buffer, half_buffer + sizeof(Eigen::half) * num_elements);

  // Update the tensor type.
  tensor->type = tflite::TensorType_FLOAT16;

  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:QuantizeTensorFloat16)

// LINT.IfChange(AddQuantizationParams)
absl::Status AddQuantizationParams(const std::vector<float>& scales,
                                   const std::vector<int64_t>& zero_point,
                                   int quantized_dimension,
                                   const uint8_t* buffer_data,
                                   size_t buffer_size, TensorType output_type,
                                   ModelT* model, TensorT* tensor) {
  if (tensor->quantization == nullptr) {
    tensor->quantization = std::make_unique<QuantizationParametersT>();
  }
  tensor->quantization->scale.assign(scales.begin(), scales.end());
  if (zero_point.size() != scales.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Received zero_point of size %d and scales of size %d. "
                     "These sizes should match.",
                     zero_point.size(), scales.size()));
  }
  tensor->quantization->zero_point.assign(zero_point.begin(), zero_point.end());
  tensor->quantization->quantized_dimension = quantized_dimension;
  model->buffers[tensor->buffer]->data.assign(buffer_data,
                                              buffer_data + buffer_size);
  // Update the tensor type.
  tensor->type = output_type;
  return absl::OkStatus();
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:AddQuantizationParams)

// LINT.IfChange(SymmetricQuantizeTensorPerChannel)
absl::Status SymmetricQuantizeTensorPerChannel(ModelT* model, TensorT* tensor,
                                               int32_t channel_dim_index) {
  if (tensor->shape.size() > kPerChannelMaxDim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "SymmetricQuantizeTensorPerChannel requires tensor with less than %d "
        "dimensions, but got %d dimension(s).",
        kPerChannelMaxDim + 1, tensor->shape.size()));
  }

  // Get dimensions.
  uint64_t num_elements;
  absl::Status status = NumElements(*tensor, &num_elements);
  if (!status.ok()) {
    return status;
  }
  const int32_t channel_dim_size = tensor->shape[channel_dim_index];

  // Get input float data.
  const BufferT* buffer = model->buffers[tensor->buffer].get();
  const float* float_input_data =
      reinterpret_cast<const float*>(buffer->data.data());

  // Create container for output scale and output data.
  std::vector<float> scales(channel_dim_size);
  std::vector<int8_t> final_buffer(num_elements);

  // Quantize the input data with respect to channel_dim_index.
  status = SymmetricPerChannelQuantization(
      tensor, float_input_data, channel_dim_index, &scales, &final_buffer);
  if (!status.ok()) {
    return status;
  }

  // Set the buffers and output type.
  uint8_t* uint8_buffer = reinterpret_cast<uint8_t*>(final_buffer.data());
  const size_t buffer_size = num_elements * sizeof(int8_t);
  std::vector<int64_t> zero_point(scales.size(), 0);
  return AddQuantizationParams(scales, zero_point, channel_dim_index,
                               uint8_buffer, buffer_size, TensorType_INT8,
                               model, tensor);
}
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.cc:SymmetricQuantizeTensorPerChannel)

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir
