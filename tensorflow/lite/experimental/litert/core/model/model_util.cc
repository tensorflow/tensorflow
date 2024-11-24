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

#include "tensorflow/lite/experimental/litert/core/model/model_util.h"

#include <utility>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
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

LiteRtStatus SetDefaultOptions(tflite::BuiltinOptionsUnion& opts,
                               LiteRtOpCode code) {
  switch (code) {
    case kLiteRtOpCodeTflMul:
      opts.Set(tflite::MulOptionsT());
      return kLiteRtStatusOk;
    case kLiteRtOpCodeTflAdd:
      opts.Set(tflite::AddOptionsT());
      return kLiteRtStatusOk;
    case kLiteRtOpCodeTflCustom:
      return kLiteRtStatusOk;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
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
    default:
      return kLiteRtElementTypeNone;
  }
}

Expected<TensorType> MapTensorType(const TflTensorType& tfl_tensor_type) {
  const auto& [element_type, shape] = tfl_tensor_type;
  if (!IsStaticTensorType(shape)) {
    LITERT_LOG(LITERT_ERROR, "Only static shaped tensors currently supported");
    return Error(kLiteRtStatusErrorUnsupported);
  }

  auto litert_element_type = MapElementType(element_type);
  if (litert_element_type == kLiteRtElementTypeNone) {
    LITERT_LOG(LITERT_ERROR, "Element type not currently supported");
    return Error(kLiteRtStatusErrorUnsupported);
  }

  LiteRtTypeDetail detail;
  detail.ranked_tensor_type.element_type = litert_element_type;
  detail.ranked_tensor_type.layout.rank = shape.shape.size();
  detail.ranked_tensor_type.layout.dimensions = shape.shape.data();
  // TFL tensors don't support strides yet.
  detail.ranked_tensor_type.layout.strides = nullptr;

  return std::make_pair(kLiteRtRankedTensorType, detail);
}

Expected<Quantization> MapQuantization(
    const TflQuantization* tfl_quantization) {
  if (!IsQuantized(tfl_quantization)) {
    return std::make_pair(kLiteRtQuantizationNone,
                          LiteRtQuantizationTypeDetail());
  }

  auto per_tensor_qparams = AsPerTensorQparams(tfl_quantization);
  if (!per_tensor_qparams) {
    LITERT_LOG(LITERT_ERROR,
               "Only per tensor quantization currently supported");
    return Error(kLiteRtStatusErrorUnsupported);
  }
  auto [zero_point, scale] = *per_tensor_qparams;

  LiteRtQuantizationTypeDetail detail;
  detail.per_tensor.scale = scale;
  detail.per_tensor.zero_point = zero_point;

  return std::make_pair(kLiteRtQuantizationPerTensor, detail);
}

}  // namespace litert::internal
