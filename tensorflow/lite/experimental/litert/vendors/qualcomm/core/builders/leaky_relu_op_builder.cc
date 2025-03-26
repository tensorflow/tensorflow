// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/leaky_relu_op_builder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;

template <typename T>
TensorWrapper& CreateAlphaTensor(
    TensorPool& tensor_pool, const Qnn_DataType_t data_type,
    const QuantizeParamsWrapperVariant& quant_param, const T alpha) {
  const std::vector<std::uint32_t> alpha_dims{1};
  const std::array<T, 1> alpha_data{alpha};
  return tensor_pool.CreateStaticTensor(data_type, quant_param, alpha_dims,
                                        sizeof(T) * alpha_data.size(),
                                        alpha_data.data());
}

}  // namespace
std::vector<OpWrapper> BuildLeakyReluOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const float alpha) {
  std::vector<OpWrapper> res;

  OpWrapper& leaky_relu_op = CreateOpWrapper(res, QNN_OP_PRELU);
  TensorWrapper& input_tensor = inputs[kInputIndex];
  leaky_relu_op.AddInputTensor(input_tensor);
  leaky_relu_op.AddOutputTensor(outputs[kOutputIndex]);

  if (std::holds_alternative<UndefinedQuantizeParamsWrapper>(
          input_tensor.GetQuantParams())) {
    TensorWrapper& alpha_tensor =
        CreateAlphaTensor<float>(tensor_pool, input_tensor.GetDataType(),
                                 input_tensor.GetQuantParams(), alpha);
    leaky_relu_op.AddInputTensor(alpha_tensor);
  } else if (std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
                 input_tensor.GetQuantParams())) {
    QuantizeParamsWrapperVariant quant_param;
    quant_param.emplace<ScaleOffsetQuantizeParamsWrapper>(std::max(alpha, 0.0f),
                                                          0);

    switch (input_tensor.GetDataType()) {
      case QNN_DATATYPE_UFIXED_POINT_8: {
        TensorWrapper& alpha_tensor = CreateAlphaTensor<std::uint8_t>(
            tensor_pool, input_tensor.GetDataType(), quant_param, 1);
        leaky_relu_op.AddInputTensor(alpha_tensor);
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_8: {
        TensorWrapper& alpha_tensor = CreateAlphaTensor<std::int8_t>(
            tensor_pool, input_tensor.GetDataType(), quant_param, 1);
        leaky_relu_op.AddInputTensor(alpha_tensor);
        break;
      }
      case QNN_DATATYPE_UFIXED_POINT_16: {
        TensorWrapper& alpha_tensor = CreateAlphaTensor<std::uint16_t>(
            tensor_pool, input_tensor.GetDataType(), quant_param, 1);
        leaky_relu_op.AddInputTensor(alpha_tensor);
        break;
      }
      case QNN_DATATYPE_SFIXED_POINT_16: {
        TensorWrapper& alpha_tensor = CreateAlphaTensor<std::int16_t>(
            tensor_pool, input_tensor.GetDataType(), quant_param, 1);
        leaky_relu_op.AddInputTensor(alpha_tensor);
        break;
      }
      default: {
        QNN_LOG_ERROR(
            "Unsupported QNN data type when creating alpha tensor for "
            "per-tensor quantization.");
        break;
      }
    }
  } else {
    QNN_LOG_ERROR("Unsupported quantization type for LeakyRelu op.");
  }

  return res;
}

}  // namespace qnn
