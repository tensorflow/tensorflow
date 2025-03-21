// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/miscs.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

namespace qnn {

// Get the Qnn_DataType_t associated with given C++ type.
template <typename T>
inline constexpr Qnn_DataType_t GetQnnDataType(const bool is_quant) {
  if constexpr (std::is_same_v<T, bool>) {
    return QNN_DATATYPE_BOOL_8;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return is_quant ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_UINT_8;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return is_quant ? QNN_DATATYPE_SFIXED_POINT_8 : QNN_DATATYPE_INT_8;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return is_quant ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UINT_16;
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return is_quant ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_INT_16;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return is_quant ? QNN_DATATYPE_UFIXED_POINT_32 : QNN_DATATYPE_UINT_32;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return is_quant ? QNN_DATATYPE_SFIXED_POINT_32 : QNN_DATATYPE_INT_32;
  } else if constexpr (std::is_same_v<T, float>) {
    return QNN_DATATYPE_FLOAT_32;
  } else {
    static_assert(always_false<T>, "Uknown C++ type");
  }
  return QNN_DATATYPE_UNDEFINED;
}

std::size_t GetDataTypeSize(const Qnn_DataType_t data_type);

template <typename T>
void TransposeFromOHWIToHWIO(absl::Span<const T> weight_data,
                             const std::vector<uint32_t>& weight_dims,
                             std::vector<T>& weight_data_transpose) {
  weight_data_transpose.resize(weight_data.size());
  uint32_t output = weight_dims[0];
  uint32_t height = weight_dims[1];
  uint32_t width = weight_dims[2];
  uint32_t input = weight_dims[3];
  // OHWI->HWIO
  uint32_t map_o = 0;
  uint32_t map_w = 0;
  uint32_t map_h = 0;
  for (uint32_t index_o = 0; index_o < output; index_o++) {
    map_o = index_o * height * width * input;
    for (uint32_t index_h = 0; index_h < height; index_h++) {
      map_h = index_h * width * input;
      for (uint32_t index_w = 0; index_w < width; index_w++) {
        map_w = index_w * input;
        for (uint32_t index_i = 0; index_i < input; index_i++) {
          T inval = weight_data[map_o + map_h + map_w + index_i];
          uint32_t index_transpose = index_h * width * input * output +
                                     index_w * input * output +
                                     index_i * output + index_o;
          weight_data_transpose[index_transpose] = inval;
        }
      }
    }
  }
}

class TensorWrapper final {
  friend class TensorPool;

 public:
  explicit TensorWrapper();

  explicit TensorWrapper(std::uint32_t id, Qnn_TensorType_t tensor_type,
                         Qnn_DataType_t data_type,
                         const QuantizeParamsWrapperVariant& quantize_params,
                         const std::vector<std::uint32_t>& dimentions);

  explicit TensorWrapper(std::uint32_t id, Qnn_TensorType_t tensor_type,
                         Qnn_DataType_t data_type,
                         const QuantizeParamsWrapperVariant& quantize_params,
                         const std::vector<std::uint32_t>& dimentions,
                         std::uint32_t bytes, const void* data);

  TensorWrapper(const TensorWrapper& other);

  TensorWrapper(TensorWrapper&& other);

  ~TensorWrapper();

  void CloneTo(Qnn_Tensor_t& dst) const;

  Qnn_Tensor_t& GetQnnTensor() { return qnn_tensor_; }

  std::uint32_t GetRank() const;

  std::uint32_t GetDim(size_t index) const;

  const std::vector<std::uint32_t>& GetDims() const { return dimentions_; };

  std::uint32_t GetTensorNumElements() const;

  const QuantizeParamsWrapperVariant& GetQuantParams() const {
    return quantize_params_;
  };

  QuantizeParamsWrapperVariant& GetQuantParams() { return quantize_params_; };

  const bool IsQuant() const {
    return !std::holds_alternative<UndefinedQuantizeParamsWrapper>(
        quantize_params_);
  };

  bool IsPerTensorQuantWithOffsetDiff(const TensorWrapper& rhs) const;

  bool IsQuant8() const {
    return GetDataType() == QNN_DATATYPE_SFIXED_POINT_8 ||
           GetDataType() == QNN_DATATYPE_UFIXED_POINT_8;
  }

  bool IsQuant16() const {
    return GetDataType() == QNN_DATATYPE_SFIXED_POINT_16 ||
           GetDataType() == QNN_DATATYPE_UFIXED_POINT_16;
  }

  bool IsF32() const { return GetDataType() == QNN_DATATYPE_FLOAT_32; }
  bool IsF16() const { return GetDataType() == QNN_DATATYPE_FLOAT_16; }

  Qnn_DataType_t GetDataType() const;

  void SetDataType(Qnn_DataType_t data_type);

  bool IsSubgraphInput() const {
    return GetTensorType() == QNN_TENSOR_TYPE_APP_WRITE;
  }

  bool IsSubgraphOutput() const {
    return GetTensorType() == QNN_TENSOR_TYPE_APP_READ;
  }

  bool IsTensorStatic() const {
    return GetTensorType() == QNN_TENSOR_TYPE_STATIC;
  }

  template <typename T>
  bool SetTensorData(absl::Span<const T> data) {
    if (!IsSubgraphInput() && !IsTensorStatic()) {
      QNN_LOG_ERROR(
          "Cannot set tensor data of tensor type other than "
          "QNN_TENSOR_TYPE_APP_WRITE or QNN_TENSOR_TYPE_STATIC.");
      return false;
    }

    size_t num_elements = GetTensorNumElements();
    if (!num_elements) {
      QNN_LOG_ERROR("Cannot set tensor data, number of elements = 0");
      return false;
    }

    size_t data_bytes = sizeof(T) * data.size();
    size_t tensor_bytes = GetTensorBytes();
    if (tensor_bytes > data_bytes) {
      QNN_LOG_ERROR(
          "Tensor bytes: %d > given data bytes: %d, SetTensorData failed.",
          tensor_bytes, data_bytes);
      return false;
    }
    if (tensor_bytes < data_bytes) {
      QNN_LOG_WARNING(
          "Tensor bytes : %d < given data bytes: %d, using only %d.",
          tensor_bytes, data_bytes, tensor_bytes);
    }

    if constexpr (std::is_same_v<T, float>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_FLOAT_32) {
        QNN_LOG_ERROR(
            "Cannot set tensor data, setting float data on QNN data type %d.",
            qnn_tensor_.v2.dataType);
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_INT_8 &&
          qnn_tensor_.v2.dataType != QNN_DATATYPE_SFIXED_POINT_8) {
        QNN_LOG_ERROR(
            "Cannot set tensor data, setting std::int8_t data on QNN data type "
            "%d.",
            qnn_tensor_.v2.dataType);
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_UINT_8 &&
          qnn_tensor_.v2.dataType != QNN_DATATYPE_UFIXED_POINT_8) {
        QNN_LOG_ERROR(
            "Cannot set tensor data, setting std::uint8_t data on QNN data "
            "type %d.",
            qnn_tensor_.v2.dataType);
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_INT_16 &&
          qnn_tensor_.v2.dataType != QNN_DATATYPE_SFIXED_POINT_16) {
        QNN_LOG_ERROR(
            "Cannot set tensor data, setting std::int16_t data on QNN data "
            "type %d.",
            qnn_tensor_.v2.dataType);
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_UINT_16 &&
          qnn_tensor_.v2.dataType != QNN_DATATYPE_UFIXED_POINT_16) {
        QNN_LOG_ERROR(
            "Cannot set tensor data, setting std::uint16_t data on QNN data "
            "type %d.",
            qnn_tensor_.v2.dataType);
        return false;
      }

    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_INT_32 &&
          qnn_tensor_.v2.dataType != QNN_DATATYPE_SFIXED_POINT_32) {
        QNN_LOG_ERROR(
            "Cannot set tensor data, setting std::int32_t data on QNN data "
            "type %d.",
            qnn_tensor_.v2.dataType);
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_UINT_32 &&
          qnn_tensor_.v2.dataType != QNN_DATATYPE_UFIXED_POINT_32) {
        QNN_LOG_ERROR(
            "Cannot set tensor data, setting std::uint32_t data on QNN data "
            "type %d.",
            qnn_tensor_.v2.dataType);
        return false;
      }
    } else {
      QNN_LOG_ERROR("Cannot set tensor data, unknown data type.");
      return false;
    }

    owned_data_.resize(tensor_bytes);
    std::memcpy(owned_data_.data(), reinterpret_cast<const char*>(data.data()),
                tensor_bytes);
    qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
    qnn_tensor_.v2.clientBuf.data = owned_data_.data();
    return true;
  }

  // Allocate memory on owned_data_ for output tensors
  void AllocateOutputTensorBuffer() {
    owned_data_.resize(GetTensorBytes());
    qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
    qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  }

  template <typename T>
  std::optional<absl::Span<const T>> GetStaticTensorData() const;

  void ConvertAxisScaleOffsetToScaleOffset() {
    if (!std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
            quantize_params_)) {
      return;
    }

    quantize_params_.emplace<ScaleOffsetQuantizeParamsWrapper>(0.0, 0);
  }

  size_t GetTensorBytes() const;

 private:
  Qnn_TensorType_t GetTensorType() const;

  void SetDataBy(std::uint32_t bytes, const void* data);

  Qnn_Tensor_t qnn_tensor_{.version = QNN_TENSOR_VERSION_2,
                           .v2 = QNN_TENSOR_V2_INIT};
  std::string name_{};
  std::vector<std::uint32_t> dimentions_{};
  QuantizeParamsWrapperVariant quantize_params_{};
  std::vector<std::byte> owned_data_{};
};

using TensorWrapperRef = std::reference_wrapper<TensorWrapper>;

template <typename T>
std::optional<absl::Span<const T>> TensorWrapper::GetStaticTensorData() const {
  if (!IsTensorStatic()) {
    QNN_LOG_ERROR(
        "Cannot GetStaticTensorData() on a non-static tensor, tensor type %d.",
        GetTensorType());
    return std::nullopt;
  }

  if (GetDataType() != GetQnnDataType<T>(IsQuant())) {
    QNN_LOG_ERROR("GetStaticTensorData() with incorrect template type.");
    return std::nullopt;
  }

  if (qnn_tensor_.v2.clientBuf.dataSize == 0 ||
      qnn_tensor_.v2.clientBuf.data == nullptr) {
    QNN_LOG_ERROR("Empty StaticTensorData.");
    return std::nullopt;
  }

  if (qnn_tensor_.v2.clientBuf.dataSize != GetTensorBytes()) {
    QNN_LOG_ERROR("Tensor bytes != stored data bytes.");
    return std::nullopt;
  }

  uint32_t num_elements = qnn_tensor_.v2.clientBuf.dataSize / sizeof(T);
  if (!num_elements) {
    QNN_LOG_ERROR("No element in this tensor.");
    return std::nullopt;
  }

  return absl::MakeConstSpan(
      reinterpret_cast<const T*>(qnn_tensor_.v2.clientBuf.data), num_elements);
}
}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_
