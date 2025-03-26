// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/fully_connected_op_builder_htp.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
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

std::vector<OpWrapper> BuildFullyConnectedOpHtp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_num_dims) {
  std::vector<OpWrapper> res;
  QNN_LOG_INFO("[FullyConnected Optimization] FC -> CONV2D");
  // TFLite FC Input: [1, k, n] and Weight: [m, n]
  // QNN Conv2D Input:
  // [batch, height, width, channel_in]
  // -> [1, 1, k, n]
  // QNN Conv2D Weight:
  // [filter_height, filter_width, channel_in / group, channel_out]
  // -> [1, 1, n, m]
  bool is_supported = inputs[0].get().GetRank() == 3 && inputs.size() == 2 &&
                      inputs[1].get().IsTensorStatic();
  if (!is_supported) {
    QNN_LOG_INFO("[FullyConnected Optimization] FAILURE: Unsupported Input");
    return res;
  }

  // TFLite FC -> QNN CONV2D:
  // Reshape -> Conv2D -> Reshpae
  TensorWrapper& input_tensor = inputs[0];
  TensorWrapper& weight_tensor = inputs[1];
  TensorWrapper& output_tensor = outputs[0];
  // Reshape
  qnn::OpWrapper& reshape_op_1 = CreateOpWrapper(res, QNN_OP_RESHAPE);
  reshape_op_1.AddInputTensor(input_tensor);
  std::vector<uint32_t> conv_input_dims = input_tensor.GetDims();
  conv_input_dims.insert(conv_input_dims.begin() + 1, 1);
  qnn::TensorWrapper& conv_input_tensor =
      tensor_pool.CloneNativeTensorFrom(input_tensor, conv_input_dims);
  reshape_op_1.AddOutputTensor(conv_input_tensor);
  // Conv2D Input, Weight, and Output
  OpWrapper& conv_op = CreateOpWrapper(res, QNN_OP_CONV_2D);
  conv_op.AddInputTensor(conv_input_tensor);
  auto& quant_params = weight_tensor.GetQuantParams();
  if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          quant_params)) {
    auto& axis_quant_param =
        std::get<AxisScaleOffsetQuantizeParamsWrapper>(quant_params);
    axis_quant_param.SetAxis(3);
  }
  std::vector<uint32_t> weight_dims{1, 1, weight_tensor.GetDim(1),
                                    weight_tensor.GetDim(0)};
  size_t weight_bytes = weight_tensor.GetTensorBytes();
  const std::vector<std::uint32_t> transpose_dim{weight_tensor.GetDim(0), 1, 1,
                                                 weight_tensor.GetDim(1)};
  TensorWrapper* weight;
  if (weight_tensor.GetDataType() == QNN_DATATYPE_SFIXED_POINT_8) {
    std::vector<std::int8_t> conv_weight;
    auto fc_weight = weight_tensor.GetStaticTensorData<std::int8_t>();
    TransposeFromOHWIToHWIO(fc_weight.value(), transpose_dim, conv_weight);
    weight = &(tensor_pool.CreateStaticTensor(
        weight_tensor.GetDataType(), quant_params, weight_dims, weight_bytes,
        conv_weight.data()));
  } else if (weight_tensor.GetDataType() == QNN_DATATYPE_SFIXED_POINT_16) {
    std::vector<std::int16_t> conv_weight;
    auto fc_weight = weight_tensor.GetStaticTensorData<std::int16_t>();
    TransposeFromOHWIToHWIO(fc_weight.value(), transpose_dim, conv_weight);
    weight = &(tensor_pool.CreateStaticTensor(
        weight_tensor.GetDataType(), quant_params, weight_dims, weight_bytes,
        conv_weight.data()));
  } else if (weight_tensor.GetDataType() == QNN_DATATYPE_UFIXED_POINT_16) {
    std::vector<std::uint16_t> conv_weight;
    auto fc_weight = weight_tensor.GetStaticTensorData<std::uint16_t>();
    TransposeFromOHWIToHWIO(fc_weight.value(), transpose_dim, conv_weight);
    weight = &(tensor_pool.CreateStaticTensor(
        weight_tensor.GetDataType(), quant_params, weight_dims, weight_bytes,
        conv_weight.data()));
  } else if (weight_tensor.GetDataType() == QNN_DATATYPE_FLOAT_32) {
    std::vector<float> conv_weight;
    auto fc_weight = weight_tensor.GetStaticTensorData<float>();
    TransposeFromOHWIToHWIO(fc_weight.value(), transpose_dim, conv_weight);
    weight = &(tensor_pool.CreateStaticTensor(
        weight_tensor.GetDataType(), quant_params, weight_dims, weight_bytes,
        conv_weight.data()));
  } else {
    QNN_LOG_INFO(
        "[FullyConnected Optimization] FAILURE: Upsupported Weight Datatype");
    return {};
  }
  conv_op.AddInputTensor(*weight);
  qnn::TensorWrapper& conv_out = tensor_pool.CloneNativeTensorFrom(
      output_tensor, {conv_input_dims[0], conv_input_dims[1],
                      conv_input_dims[2], weight_dims[3]});
  conv_op.AddOutputTensor(conv_out);
  // Conv2D Stride
  const std::array<std::uint32_t, 2> stride_data{1, 1};
  const std::vector<std::uint32_t> stride_shape{2};
  auto& stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(std::uint32_t) * stride_data.size(), stride_data.data());
  conv_op.AddTensorParam(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE, stride_tensor);
  // Conv2D Padding
  const std::array<std::uint32_t, 4> padding_data = {0, 0, 0, 0};
  const std::vector<std::uint32_t> padding_shape{2, 2};
  auto& padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(std::uint32_t) * padding_data.size(), padding_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, padding_tensor);

  // Reshape
  qnn::OpWrapper& reshape_op_2 = CreateOpWrapper(res, QNN_OP_RESHAPE);
  reshape_op_2.AddInputTensor(conv_out);
  reshape_op_2.AddOutputTensor(output_tensor);
  return res;
}

}  // namespace qnn
