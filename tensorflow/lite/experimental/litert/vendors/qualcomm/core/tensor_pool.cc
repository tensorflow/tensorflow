//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"

namespace qnn {

TensorPool::TensorPool() = default;

TensorPool::TensorPool(std::function<void(TensorWrapper&)> tensor_callback)
    : tensor_callback_{tensor_callback}, tensor_wrappers_{} {}

TensorWrapper& TensorPool::CreateInputTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  auto& back = tensor_wrappers_.emplace_back(
      id, QNN_TENSOR_TYPE_APP_WRITE, data_type, quant_params, dimentions);

  if (tensor_callback_) {
    tensor_callback_(back);
  }

  return back;
}

TensorWrapper& TensorPool::CreateOutpuTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  auto& back = tensor_wrappers_.emplace_back(
      id, QNN_TENSOR_TYPE_APP_READ, data_type, quant_params, dimentions);

  if (tensor_callback_) {
    tensor_callback_(back);
  }

  return back;
}

TensorWrapper& TensorPool::CreateNativeTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  auto& back = tensor_wrappers_.emplace_back(
      id, QNN_TENSOR_TYPE_NATIVE, data_type, quant_params, dimentions);

  if (tensor_callback_) {
    tensor_callback_(back);
  }

  return back;
}

TensorWrapper& TensorPool::CreateStaticTensor(
    Qnn_DataType_t data_type, const QuantizeParamsWrapperVariant& quant_params,
    const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
    const void* data) {
  const auto id = tensor_wrappers_.size();
  auto& back =
      tensor_wrappers_.emplace_back(id, QNN_TENSOR_TYPE_STATIC, data_type,
                                    quant_params, dimentions, bytes, data);

  if (tensor_callback_) {
    tensor_callback_(back);
  }

  return back;
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(const TensorWrapper& src) {
  const auto id = tensor_wrappers_.size();
  auto& back = tensor_wrappers_.emplace_back(
      id, QNN_TENSOR_TYPE_NATIVE, src.GetDataType(), src.quantize_params_,
      src.dimentions_);

  if (tensor_callback_) {
    tensor_callback_(back);
  }

  return back;
}

TensorWrapper& TensorPool::CloneNativeTensorFrom(
    const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions) {
  const auto id = tensor_wrappers_.size();
  auto& back = tensor_wrappers_.emplace_back(id, QNN_TENSOR_TYPE_NATIVE,
                                             src.GetDataType(),
                                             src.quantize_params_, dimentions);

  if (tensor_callback_) {
    tensor_callback_(back);
  }

  return back;
}

TensorWrapper& TensorPool::CloneStaticTensorFrom(const TensorWrapper& src,
                                                 Qnn_DataType_t data_type) {
  const auto id = tensor_wrappers_.size();
  auto& back = tensor_wrappers_.emplace_back(
      id, QNN_TENSOR_TYPE_STATIC, data_type, src.quantize_params_,
      src.dimentions_, src.owned_data_.size(), src.owned_data_.data());

  if (tensor_callback_) {
    tensor_callback_(back);
  }

  return back;
}

}  // namespace qnn