//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_TENSOR_POOL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_TENSOR_POOL_H_

#include <functional>
#include <list>

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

namespace qnn {

class TensorPool {
 public:
  TensorPool();

  TensorPool(std::function<void(TensorWrapper&)> tensor_callback);

  TensorWrapper& CreateInputTensor(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions);

  TensorWrapper& CreateOutpuTensor(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions);

  TensorWrapper& CreateNativeTensor(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions);

  TensorWrapper& CreateStaticTensor(
      Qnn_DataType_t data_type,
      const QuantizeParamsWrapperVariant& quant_params,
      const std::vector<std::uint32_t>& dimentions, std::uint32_t bytes,
      const void* data);

  TensorWrapper& CloneNativeTensorFrom(const TensorWrapper& src);

  TensorWrapper& CloneNativeTensorFrom(
      const TensorWrapper& src, const std::vector<std::uint32_t>& dimentions);

 private:
  std::function<void(TensorWrapper&)> tensor_callback_{};
  std::list<TensorWrapper> tensor_wrappers_{};
};

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_TENSOR_POOL_H_
