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

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
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

LiteRtStatus IsTensorSupported(const tflite::TensorT& tensor) {
  if (!tensor.has_rank) {
    // TODO: b/365299994 - Support unranked tensors.
    LITERT_LOG(LITERT_ERROR, "Unranked tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

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

  if (!tensor.shape_signature.empty()) {
    // TODO: b/365299994 - Support shape signature.
    LITERT_LOG(LITERT_ERROR, "Shape signature not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tensor.sparsity) {
    // TODO: b/365299994 - Support sparsity tensors.
    LITERT_LOG(LITERT_ERROR, "Sparsity tensors not yet supported.");
    return kLiteRtStatusErrorUnsupported;
  }

  if (tensor.type != tflite::TensorType_FLOAT32 &&
      tensor.type != tflite::TensorType_INT32 &&
      tensor.type != tflite::TensorType_BOOL) {
    // TODO: b/365299994 - Support all element types.
    LITERT_LOG(LITERT_ERROR, "Only f32, i32 and bool currently supported.");
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

LiteRtElementType MapElementType(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_FLOAT32:
      return kLiteRtElementTypeFloat32;
    case tflite::TensorType_FLOAT16:
      return kLiteRtElementTypeFloat16;
    case tflite::TensorType_INT32:
      return kLiteRtElementTypeInt32;
    case tflite::TensorType_BOOL:
      return kLiteRtElementTypeBool;
    default:
      return kLiteRtElementTypeNone;
  }
}

}  // namespace litert::internal
