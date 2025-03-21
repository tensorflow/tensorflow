// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

namespace qnn {

std::size_t GetDataTypeSize(const Qnn_DataType_t data_type) {
  std::size_t bytes = 0;
  switch (data_type) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      bytes = 1;
      break;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      bytes = 2;
      break;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      bytes = 4;
      break;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      bytes = 8;
      break;
    case QNN_DATATYPE_UNDEFINED:
    case QNN_DATATYPE_SFIXED_POINT_4:
    case QNN_DATATYPE_UFIXED_POINT_4:
    default:
      bytes = 0;
      break;
  }
  return bytes;
}

TensorWrapper::TensorWrapper() = default;

TensorWrapper::TensorWrapper(
    std::uint32_t id, Qnn_TensorType_t tensor_type, Qnn_DataType_t data_type,
    const QuantizeParamsWrapperVariant& quantize_params,
    const std::vector<std::uint32_t>& dimentions)
    : name_{std::to_string(id)},
      dimentions_{dimentions},
      quantize_params_{quantize_params} {
  qnn_tensor_.v2.name = name_.c_str();
  qnn_tensor_.v2.type = tensor_type;
  qnn_tensor_.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  qnn_tensor_.v2.dataType = data_type;
  std::visit(
      [this](auto&& quantize_params) -> void {
        quantize_params.CloneTo(qnn_tensor_.v2.quantizeParams);
      },
      quantize_params_);
  qnn_tensor_.v2.rank = dimentions_.size();
  qnn_tensor_.v2.dimensions = dimentions_.data();
  qnn_tensor_.v2.memType = QNN_TENSORMEMTYPE_RAW;
}

TensorWrapper::TensorWrapper(
    std::uint32_t id, Qnn_TensorType_t tensor_type, Qnn_DataType_t data_type,
    const QuantizeParamsWrapperVariant& quantize_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
    const void* data)
    : TensorWrapper(id, tensor_type, data_type, quantize_params, dimentions) {
  SetDataBy(bytes, data);
}

TensorWrapper::TensorWrapper(const TensorWrapper& other)
    : qnn_tensor_{other.qnn_tensor_},
      name_{other.name_},
      dimentions_{other.dimentions_},
      quantize_params_{other.quantize_params_},
      owned_data_{other.owned_data_} {
  qnn_tensor_.v2.name = name_.c_str();
  qnn_tensor_.v2.dimensions = dimentions_.data();
  qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  std::visit(
      [this](auto&& quant_params) -> void {
        quant_params.CloneTo(qnn_tensor_.v2.quantizeParams);
      },
      quantize_params_);
}

TensorWrapper::TensorWrapper(TensorWrapper&& other)
    : qnn_tensor_{other.qnn_tensor_},
      name_{std::move(other.name_)},
      dimentions_{std::move(other.dimentions_)},
      quantize_params_{std::move(other.quantize_params_)},
      owned_data_{std::move(other.owned_data_)} {
  qnn_tensor_.v2.name = name_.c_str();
  qnn_tensor_.v2.dimensions = dimentions_.data();
  qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  std::visit(
      [this](auto&& quant_params) -> void {
        quant_params.CloneTo(qnn_tensor_.v2.quantizeParams);
      },
      quantize_params_);
}

TensorWrapper::~TensorWrapper() = default;

std::uint32_t TensorWrapper::GetDim(size_t index) const {
  return dimentions_[index];
}

Qnn_DataType_t TensorWrapper::GetDataType() const {
  return qnn_tensor_.v2.dataType;
}

void TensorWrapper::CloneTo(Qnn_Tensor_t& dst) const { dst = qnn_tensor_; }

std::uint32_t TensorWrapper::GetRank() const { return qnn_tensor_.v2.rank; }

Qnn_TensorType_t TensorWrapper::GetTensorType() const {
  return qnn_tensor_.v2.type;
}

std::uint32_t TensorWrapper::GetTensorNumElements() const {
  return GetDims().empty() ? 0
                           : std::accumulate(GetDims().begin(), GetDims().end(),
                                             1, std::multiplies<>());
}

size_t TensorWrapper::GetTensorBytes() const {
  return GetDataTypeSize(GetDataType()) * GetTensorNumElements();
}

void TensorWrapper::SetDataType(Qnn_DataType_t data_type) {
  qnn_tensor_.v2.dataType = data_type;
}

bool TensorWrapper::IsPerTensorQuantWithOffsetDiff(
    const TensorWrapper& rhs) const {
  const auto& lhs_quant = qnn_tensor_.v2.quantizeParams;
  const auto& rhs_quant = rhs.qnn_tensor_.v2.quantizeParams;

  if (lhs_quant.encodingDefinition != QNN_DEFINITION_DEFINED ||
      rhs_quant.encodingDefinition != QNN_DEFINITION_DEFINED) {
    return false;
  }

  if (lhs_quant.quantizationEncoding !=
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET ||
      rhs_quant.quantizationEncoding !=
          QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    return false;
  }

  const auto lhs_scale = lhs_quant.scaleOffsetEncoding.scale;
  const auto lhs_offset = lhs_quant.scaleOffsetEncoding.offset;
  const auto rhs_scale = rhs_quant.scaleOffsetEncoding.scale;
  const auto rhs_offset = rhs_quant.scaleOffsetEncoding.offset;
  if ((GetDataType() == QNN_DATATYPE_SFIXED_POINT_8 &&
       rhs.GetDataType() == QNN_DATATYPE_UFIXED_POINT_8) ||
      (GetDataType() == QNN_DATATYPE_UFIXED_POINT_8 &&
       rhs.GetDataType() == QNN_DATATYPE_SFIXED_POINT_8)) {
    constexpr int kSUFixed8OffsetDiff = 128;
    if (std::fabs(lhs_scale - rhs_scale) <
            std::numeric_limits<float>::epsilon() &&
        std::abs(lhs_offset - rhs_offset) == kSUFixed8OffsetDiff) {
      return true;
    }
  } else if ((GetDataType() == QNN_DATATYPE_SFIXED_POINT_16 &&
              rhs.GetDataType() == QNN_DATATYPE_UFIXED_POINT_16) ||
             (GetDataType() == QNN_DATATYPE_UFIXED_POINT_16 &&
              rhs.GetDataType() == QNN_DATATYPE_SFIXED_POINT_16)) {
    constexpr int kSUFixed16OffsetDiff = 32768;
    if (std::fabs(lhs_scale - rhs_scale) <
            std::numeric_limits<float>::epsilon() &&
        std::abs(lhs_offset - rhs_offset) == kSUFixed16OffsetDiff) {
      return true;
    }
  }
  return false;
}

void TensorWrapper::SetDataBy(std::uint32_t bytes, const void* data) {
  if (bytes != GetTensorBytes()) {
    QNN_LOG_WARNING(
        "Bytes: %d != GetTensorBytes(): %d, use GetTensorBytes() instead.",
        bytes, GetTensorBytes());
    bytes = GetTensorBytes();
  }
  owned_data_.resize(bytes);
  std::memcpy(owned_data_.data(), reinterpret_cast<const char*>(data), bytes);
  qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
  qnn_tensor_.v2.clientBuf.data = owned_data_.data();
}

}  // namespace qnn
