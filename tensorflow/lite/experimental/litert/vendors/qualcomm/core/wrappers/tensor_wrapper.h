// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"

namespace qnn {

std::size_t GetDataTypeSize(const Qnn_DataType_t data_type);

template <typename T>
void TransposeOp(const T* weight_data, const uint32_t* weight_dims,
                         T* weight_data_transpose) {
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

  const QuantizeParamsWrapperVariant& GetQuantParams() const {
    return quantize_params_;
  };

  QuantizeParamsWrapperVariant& GetQuantParams() { return quantize_params_; };

  bool IsPerTensorQuantWithOffsetDiff(const TensorWrapper& rhs) const;

  bool IsQuant8() const {
    return GetDataType() == QNN_DATATYPE_SFIXED_POINT_8 ||
           GetDataType() == QNN_DATATYPE_UFIXED_POINT_8;
  }

  bool IsQuant16() const {
    return GetDataType() == QNN_DATATYPE_SFIXED_POINT_16 ||
           GetDataType() == QNN_DATATYPE_UFIXED_POINT_16;
  }

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

  void SetTensorData(std::uint32_t bytes, const void* data);

  const std::vector<std::byte>& GetTensorData() const { return owned_data_; }

  // Allocate memory on owned_data_ for output tensors
  void AllocateOutputTensorBuffer() {
    owned_data_.resize(GetTensorSize());
    qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
    qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  }

  const void* GetStaticTensorData() const {
    return qnn_tensor_.v2.clientBuf.data;
  };

  void ConvertAxisScaleOffsetToScaleOffset() {
    if (!std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
            quantize_params_)) {
      return;
    }

    quantize_params_.emplace<ScaleOffsetQuantizeParamsWrapper>(0.0, 0);
  }

  size_t GetTensorSize() const;

 private:
  Qnn_TensorType_t GetTensorType() const;

  Qnn_Tensor_t qnn_tensor_{.version = QNN_TENSOR_VERSION_2,
                           .v2 = QNN_TENSOR_V2_INIT};
  std::string name_{};
  std::vector<std::uint32_t> dimentions_{};
  QuantizeParamsWrapperVariant quantize_params_{};
  std::vector<std::byte> owned_data_{};
};

using TensorWrapperRef = std::reference_wrapper<TensorWrapper>;

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_
