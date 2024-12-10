// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/model/flatbuffer_to_litert.h"

#include <cstdint>
#include <utility>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

LiteRtStatus IsOpSupported(const tflite::OperatorT& op) {
  // TODO: b/365299994 - Check for supported options.

  if (!op.intermediates.empty()) {
    // TODO: b/365299994 - Support intermediates.
    LITERT_LOG(LITERT_ERROR, "Intermediate tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (op.large_custom_options_size != 0) {
    // TODO: b/365299994 - Support large custom options.
    LITERT_LOG(LITERT_ERROR, "Large custom options not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  for (auto m_input : op.mutating_variable_inputs) {
    if (m_input) {
      // TODO: b/365299994 - Support mutating variable inputs.
      LITERT_LOG(LITERT_ERROR, "Mutating variable inputs not yet supported.");
      return kLiteRtStatusErrorUnsupported;
    }
  }

  return kLiteRtStatusOk;
}

LiteRtStatus IsBufferSupported(const tflite::BufferT& buffer) {
  if (buffer.offset != 0) {
    // TODO: b/365299994 - Support buffer with offset.
    LITERT_LOG(LITERT_ERROR, "Buffers with offset not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtStatus IsTensorSupported(const TflTensor& tensor) {
  if (tensor.is_variable) {
    // TODO: b/365299994 - Support variable tensors.
    LITERT_LOG(LITERT_ERROR, "Variable tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (!tensor.variant_tensors.empty()) {
    // TODO: b/365299994 - Support variant tensors.
    LITERT_LOG(LITERT_ERROR, "Variant tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tensor.sparsity) {
    // TODO: b/365299994 - Support sparsity tensors.
    LITERT_LOG(LITERT_ERROR, "Sparsity tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}

LiteRtElementType MapElementType(TflElementType type) {
  switch (type) {
    case tflite::TensorType_FLOAT32:
      return kLiteRtElementTypeFloat32;
    case tflite::TensorType_FLOAT16:
      return kLiteRtElementTypeFloat16;
    case tflite::TensorType_INT32:
      return kLiteRtElementTypeInt32;
    case tflite::TensorType_BOOL:
      return kLiteRtElementTypeBool;
    case tflite::TensorType_INT16:
      return kLiteRtElementTypeInt16;
    case tflite::TensorType_INT8:
      return kLiteRtElementTypeInt8;
    default:
      return kLiteRtElementTypeNone;
  }
}

Expected<TensorType> MapTensorType(const TflTensorType& tfl_tensor_type) {
  const auto& [element_type, shape] = tfl_tensor_type;
  auto ranked_shape = AsDynamicShape(shape);
  if (!ranked_shape) {
    LITERT_LOG(LITERT_ERROR, "Only ranked tensors currently supported");
    return Error(kLiteRtStatusErrorUnsupported);
  }

  auto litert_element_type = MapElementType(element_type);
  if (litert_element_type == kLiteRtElementTypeNone) {
    LITERT_LOG(LITERT_ERROR, "Element type not currently supported");
    return Error(kLiteRtStatusErrorUnsupported);
  }

  LiteRtTypeDetail detail;
  detail.ranked_tensor_type.element_type = litert_element_type;
  detail.ranked_tensor_type.layout = BuildLayout(*ranked_shape);

  return std::make_pair(kLiteRtRankedTensorType, detail);
}

Expected<Quantization> MapQuantization(
    const TflQuantization* tfl_quantization) {
  if (!IsQuantized(tfl_quantization)) {
    return std::make_pair(kLiteRtQuantizationNone,
                          LiteRtQuantizationTypeDetail());
  }

  LiteRtQuantizationTypeId quantization_type;
  LiteRtQuantizationTypeDetail qparams;

  if (IsPerTensorQuantized(tfl_quantization)) {
    quantization_type = kLiteRtQuantizationPerTensor;
    auto per_tensor_qparams = AsPerTensorQparams(tfl_quantization);
    if (!per_tensor_qparams) {
      LITERT_LOG(LITERT_ERROR, "Per-tensor quantization parameters not found.");
      return Error(kLiteRtStatusErrorNotFound);
    }
    auto [zero_point, scale] = *per_tensor_qparams;
    qparams.per_tensor.scale = scale;
    qparams.per_tensor.zero_point = zero_point;
  }
  if (IsPerChannelQuantized(tfl_quantization)) {
    quantization_type = kLiteRtQuantizationPerChannel;
    auto per_channel_qparams = AsPerChannelQparams(tfl_quantization);
    if (!per_channel_qparams) {
      LITERT_LOG(LITERT_ERROR,
                 "Per-channel quantization parameters not found.");
      return Error(kLiteRtStatusErrorNotFound);
    }
    auto [quantized_dimension, num_channels, zero_points, scales] =
        *per_channel_qparams;
    qparams.per_channel.scales = const_cast<float*>(scales->data());
    qparams.per_channel.zero_points = const_cast<int64_t*>(zero_points->data());
    qparams.per_channel.quantized_dimension = quantized_dimension;
    qparams.per_channel.num_channels = num_channels;
  }

  return std::make_pair(quantization_type, qparams);
}
}  // namespace litert::internal
