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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"

#include <cstdint>

#include "absl/log/absl_check.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"

namespace litert::qnn {

namespace {

LiteRtStatus LegalizeShapeInfo(const litert::Layout& src, Qnn_Tensor_t& dest) {
  LITERT_ENSURE_SUPPORTED(!src.HasStrides(), "Strides not yet supported");

  dest.v2.rank = src.Rank();
  // Ad-hoc fix: rank 0 tensor needs to be single element 1D tensor in QNN.
  if (dest.v2.rank == 0) {
    LITERT_LOG(LITERT_INFO, "Setting rank 0 tensor to single element tensor");
    dest.v2.rank = 1;
    dest.v2.dimensions = new uint32_t[1];
    dest.v2.dimensions[0] = 1;
    return kLiteRtStatusOk;
  }

  dest.v2.dimensions = new uint32_t[dest.v2.rank];
  for (int i = 0; i < dest.v2.rank; ++i) {
    const auto src_dim = src.Dimensions()[i];
    LITERT_ENSURE(src_dim >= 1, kLiteRtStatusErrorInvalidArgument,
                  "Cannot pass dim < 1 to QNN Tensor.");

    dest.v2.dimensions[i] = src.Dimensions()[i];
  }
  return kLiteRtStatusOk;
}

void FreeTensorDims(Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_2 &&
      tensor.v2.dimensions != nullptr) {
    delete[] tensor.v2.dimensions;
    tensor.v2.dimensions = nullptr;
    tensor.v2.rank = 0;
  }
}

void FreePerChannelQuantization(Qnn_Tensor_t& tensor) {
  if (tensor.v2.quantizeParams.quantizationEncoding ==
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    delete[] tensor.v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset;
    tensor.v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset = nullptr;
    tensor.v2.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets = 0;
  }
}

}  // namespace

void SetInputTensorAttrs(Qnn_Tensor_t& tensor) {
  ABSL_DCHECK(tensor.version == QNN_TENSOR_VERSION_2);
  tensor.v2.type = QNN_TENSOR_TYPE_APP_WRITE;
  tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  tensor.v2.clientBuf = QNN_CLIENT_BUFFER_INIT;
}

void SetOutputTensorAttrs(Qnn_Tensor_t& tensor) {
  ABSL_DCHECK(tensor.version == QNN_TENSOR_VERSION_2);
  tensor.v2.type = QNN_TENSOR_TYPE_APP_READ;
}

void SetResultTensorAttrs(Qnn_Tensor_t& tensor) {
  ABSL_DCHECK(tensor.version == QNN_TENSOR_VERSION_2);
  tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  tensor.v2.type = QNN_TENSOR_TYPE_NATIVE;
}

void ResetTensor(Qnn_Tensor_t& tensor) {
  FreeTensorDims(tensor);
  FreePerChannelQuantization(tensor);
  tensor = QNN_TENSOR_INIT;
  tensor.version = QNN_TENSOR_VERSION_2;
  tensor.v2 = QNN_TENSOR_V2_INIT;
  tensor.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
  tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
}

Qnn_Tensor_t BuildDefaultTensor(uint32_t id) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  ResetTensor(tensor);
  tensor.v2.id = id;
  return tensor;
}

Qnn_Tensor_t BuildDefaultTensor() { return BuildDefaultTensor(0); }

Qnn_Tensor_t BuildInputTensor() {
  auto tensor = BuildDefaultTensor();
  SetInputTensorAttrs(tensor);
  return tensor;
}

Qnn_ClientBuffer_t BuildDefaultClientBuffer() {
  Qnn_ClientBuffer_t client_buf = QNN_CLIENT_BUFFER_INIT;
  client_buf.data = nullptr;
  client_buf.dataSize = 0;
  return client_buf;
}

Qnn_Tensor_t BuildOutputTensor() {
  Qnn_Tensor_t tensor = BuildDefaultTensor();
  SetOutputTensorAttrs(tensor);
  return tensor;
}

uint32_t MoveToId(Qnn_Tensor_t& tensor) {
  const auto id = tensor.v2.id;
  ResetTensor(tensor);
  tensor.v2.id = id;
  return id;
}

void SetPerChannelQuantization(
    Qnn_Tensor_t& tensor,
    const LiteRtQuantizationPerChannel& lite_rt_quantization_per_channel) {
  tensor.v2.quantizeParams.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;

  tensor.v2.quantizeParams.axisScaleOffsetEncoding = QNN_AXIS_SCALE_OFFSET_INIT;
  tensor.v2.quantizeParams.axisScaleOffsetEncoding.axis =
      lite_rt_quantization_per_channel.quantized_dimension;
  tensor.v2.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets =
      lite_rt_quantization_per_channel.num_channels;

  // Allocates memory for scaleOffset array.
  tensor.v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset =
      new Qnn_ScaleOffset_t[lite_rt_quantization_per_channel.num_channels];

  for (int i = 0; i < lite_rt_quantization_per_channel.num_channels; ++i) {
    tensor.v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset[i].scale =
        lite_rt_quantization_per_channel.scales[i];
    tensor.v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset[i].offset =
        lite_rt_quantization_per_channel.zero_points[i];
  }
}

void SetPerTensorQuantization(
    Qnn_Tensor_t& tensor,
    const LiteRtQuantizationPerTensor& lite_rt_quantization_per_tensor) {
  tensor.v2.quantizeParams.quantizationEncoding =
      QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  tensor.v2.quantizeParams.scaleOffsetEncoding.scale =
      lite_rt_quantization_per_tensor.scale;
  tensor.v2.quantizeParams.scaleOffsetEncoding.offset =
      lite_rt_quantization_per_tensor.zero_point;
}

LiteRtStatus LegalizeQuntizationParameter(const litert::Tensor& src,
                                          Qnn_Tensor_t& dest) {
  LiteRtQuantizationTypeId lite_rt_quantization_type_id = src.QTypeId();
  switch (lite_rt_quantization_type_id) {
    case kLiteRtQuantizationPerTensor:
      SetPerTensorQuantization(dest, src.PerTensorQuantization());
      return kLiteRtStatusOk;
    case kLiteRtQuantizationPerChannel:
      SetPerChannelQuantization(dest, src.PerChannelQuantization());
      return kLiteRtStatusOk;
    default:
      LITERT_LOG(LITERT_ERROR, "Unsupported quantization type.");
      return kLiteRtStatusErrorInvalidArgument;
  }
}

LiteRtStatus LegalizeTensor(const litert::Tensor& src, Qnn_Tensor_t& dest) {
  if (src.TypeId() != kLiteRtRankedTensorType) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  ResetTensor(dest);

  if (src.HasQuantization()) {
    LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeQuntizationParameter(src, dest));
  }

  auto src_ranked_tensor_type = src.RankedTensorType();
  if (!src_ranked_tensor_type) {
    LITERT_LOG(LITERT_ERROR, "%s",
               src_ranked_tensor_type.Error().Message().data());
    return src_ranked_tensor_type.Error().Status();
  }

  Qnn_DataType_t* qnn_data_type = &dest.v2.dataType;
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeElementType(
      src_ranked_tensor_type->ElementType(), qnn_data_type));

  LITERT_RETURN_STATUS_IF_NOT_OK(
      LegalizeShapeInfo(src_ranked_tensor_type->Layout(), dest));

  const bool is_subgraph_in = src.IsSubgraphInput();
  const bool is_subgraph_out = src.IsSubgraphOutput();
  const bool is_constant = src.IsConstant();

  LITERT_ENSURE(!(is_subgraph_in && is_subgraph_out),
                kLiteRtStatusErrorInvalidArgument,
                "Malformed tensor, cannot be both subgraph in and out.");
  if (is_constant) {
    LITERT_LOG(LITERT_INFO, "Adding constant tensor %s to qnn graph",
               dest.v2.name);
    LITERT_ENSURE(src.HasWeights(), kLiteRtStatusErrorInvalidLegalization,
                  "Empty weights for constant tensor.");
    Qnn_ClientBuffer_t client_buf = BuildDefaultClientBuffer();
    client_buf.data = (void*)src.Weights().Bytes().data();
    client_buf.dataSize = src.Weights().Bytes().size();
    dest.v2.clientBuf = client_buf;
    dest.v2.memType = QNN_TENSORMEMTYPE_RAW;
    dest.v2.type = QNN_TENSOR_TYPE_STATIC;
    dest.v2.isDynamicDimensions = nullptr;
  }

  if (is_subgraph_in) {
    LITERT_LOG(LITERT_INFO, "Adding subgraph input tensor to qnn graph");
    SetInputTensorAttrs(dest);
  }
  if (is_subgraph_out) {
    LITERT_LOG(LITERT_INFO, "Adding subgraph output tensor to qnn graph");
    SetOutputTensorAttrs(dest);
  }
  if (!is_constant && !is_subgraph_in && !is_subgraph_out) {
    LITERT_LOG(LITERT_INFO, "Adding result tensor to qnn graph");
    SetResultTensorAttrs(dest);
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
