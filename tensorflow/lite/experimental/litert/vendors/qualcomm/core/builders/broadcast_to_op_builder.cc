// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/broadcast_to_op_builder.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
std::vector<std::uint32_t> GetStaticTensorDimention(
    const std::vector<std::uint32_t>& output,
    const std::vector<std::uint32_t>& input) {
  std::vector<std::uint32_t> final_dims(output.size());
  for (size_t i = 0; i < output.size(); ++i) {
    final_dims[i] = (output[i] > input[i]) ? output[i] : 1;
  }
  return final_dims;
}

template <typename T>
TensorWrapper& CreateStaticTensor(TensorPool& tensor_pool,
                                  const Qnn_DataType_t data_type,
                                  qnn::TensorWrapperRef output,
                                  qnn::TensorWrapperRef input) {
  std::vector<T> static_data{0};
  std::vector<std::uint32_t> static_dims =
      GetStaticTensorDimention(output.get().GetDims(), input.get().GetDims());
  std::uint32_t static_size =
      std::accumulate(static_dims.begin(), static_dims.end(), 1,
                      std::multiplies<std::uint32_t>());

  return tensor_pool.CreateStaticTensor(
      data_type, QuantizeParamsWrapperVariant{}, static_dims,
      sizeof(T) * static_size, static_data.data());
}

std::vector<OpWrapper> BuildBroadcastToOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;
  TensorWrapper* broadcast_in = nullptr;

  const char* qnn_op = nullptr;
  if (inputs[0].get().GetDataType() == QNN_DATATYPE_BOOL_8) {
    qnn_op = QNN_OP_ELEMENT_WISE_OR;
  } else {
    qnn_op = QNN_OP_ELEMENT_WISE_ADD;
  }

  auto& broadcast_op = CreateOpWrapper(res, qnn_op);
  broadcast_in = &(inputs[0].get());
  broadcast_op.AddInputTensor(*broadcast_in);

  // TODO: Need handle quant param
  switch (inputs[0].get().GetDataType()) {
    case QNN_DATATYPE_BOOL_8: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::uint8_t>(
          tensor_pool, QNN_DATATYPE_BOOL_8, outputs[0], inputs[0]);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_8: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::uint8_t>(
          tensor_pool, inputs[0].get().GetDataType(), outputs[0], inputs[0]);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_8: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::int8_t>(
          tensor_pool, inputs[0].get().GetDataType(), outputs[0], inputs[0]);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::uint16_t>(
          tensor_pool, inputs[0].get().GetDataType(), outputs[0], inputs[0]);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_SFIXED_POINT_16: {
      TensorWrapper& static_tensor = CreateStaticTensor<std::int16_t>(
          tensor_pool, inputs[0].get().GetDataType(), outputs[0], inputs[0]);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      TensorWrapper& static_tensor = CreateStaticTensor<float>(
          tensor_pool, inputs[0].get().GetDataType(), outputs[0], inputs[0]);
      broadcast_op.AddInputTensor(static_tensor);
      break;
    }
    default: {
      QNN_LOG_ERROR("Unsupported QNN data type when creating static tensor");
      break;
    }
  }

  broadcast_op.AddOutputTensor(outputs[0]);

  // TODO: fused activation
  return res;
}

}  // namespace qnn
