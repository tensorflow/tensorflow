// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/conv2d_op_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnLog.h"
#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kFilterIndex = 1;
constexpr size_t kBiasIndex = 2;
constexpr size_t kOutputIndex = 0;
constexpr size_t kBatchIndex = 0;
constexpr size_t kHeightIndex = 1;
constexpr size_t kWidthIndex = 2;
constexpr size_t kChannelIndex = 3;

}  // namespace

std::vector<OpWrapper> BuildConv2dOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const std::uint32_t stride_h,
    const std::uint32_t stride_w, const std::uint32_t dilation_h,
    const std::uint32_t dilation_w, const std::uint32_t fused_activation,
    const PaddingType padding_type) {
  std::vector<OpWrapper> res;

  // transpose filter
  TensorWrapper& filter_tensor = inputs[kFilterIndex];
  auto& filter_quant_params = filter_tensor.GetQuantParams();
  if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          filter_quant_params)) {
    auto& axis_quant_params =
        std::get<AxisScaleOffsetQuantizeParamsWrapper>(filter_quant_params);
    const std::array<std::int32_t, 4> new_axis{3, 0, 1, 2};
    axis_quant_params.SetAxis(new_axis[axis_quant_params.GetAxis()]);
  }

  const std::vector<std::uint32_t> permute_dims{
      filter_tensor.GetDim(kHeightIndex), filter_tensor.GetDim(kWidthIndex),
      filter_tensor.GetDim(kChannelIndex), filter_tensor.GetDim(kBatchIndex)};
  auto& transposed_filter_tensor =
      tensor_pool.CloneNativeTensorFrom(filter_tensor, permute_dims);

  const std::vector<std::uint32_t> permute_shape{4};
  const std::array<std::uint32_t, 4> permute_data{kHeightIndex, kWidthIndex,
                                                  kChannelIndex, kBatchIndex};
  auto& permute_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, permute_shape,
      sizeof(decltype(permute_data)::value_type) * permute_data.size(),
      permute_data.data());

  OpWrapper& transpose_op = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  transpose_op.AddInputTensor(filter_tensor);
  transpose_op.AddOutputTensor(transposed_filter_tensor);
  transpose_op.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM, permute_tensor);

  bool is_int8_weight_only_quantized =
      inputs[0].get().IsF32() && inputs[1].get().IsQuant8();

  TensorWrapper* conv2d_output_tensor = nullptr;

  // If Conv2d fused activation is available, create a new tensor for the
  // output of Conv2d, which is also the input of the activation op.
  if (fused_activation != kLiteRtFusedActivationNone) {
    conv2d_output_tensor = &tensor_pool.CreateNativeTensor(
        outputs[0].get().GetDataType(), {}, outputs[0].get().GetDims());
  } else {
    conv2d_output_tensor = &outputs[0].get();
  }

  // Converted input and output are either the original input/output tensor or
  // the output of the Convert Op.
  TensorWrapper* converted_input_tensor = nullptr;
  TensorWrapper* converted_output_tensor = nullptr;

  ConvertFp32ActivationToFp16IfWeightOnlyQuantized(
      res, inputs[0].get(), *conv2d_output_tensor, converted_input_tensor,
      converted_output_tensor, is_int8_weight_only_quantized, tensor_pool);

  // conv
  TensorWrapper* bias_tensor = nullptr;
  if (inputs.size() - 1 >= kBiasIndex) {
    TensorWrapper* converted_bias_tensor = nullptr;
    ConvertFp32ActivationToFp16(res, inputs[2].get(), converted_bias_tensor,
                                is_int8_weight_only_quantized, tensor_pool);

    // TensorWrapper& bias_tensor = inputs[kBiasIndex];
    bias_tensor = converted_bias_tensor;
    // QNN only support per-tensor quant for bias,
    // and the scale and offset are both zero.
    bias_tensor->ConvertAxisScaleOffsetToScaleOffset();
  }
  OpWrapper& conv_op = CreateOpWrapper(res, QNN_OP_CONV_2D);
  TensorWrapper& input_tensor = *converted_input_tensor;
  conv_op.AddInputTensor(input_tensor);
  conv_op.AddInputTensor(transposed_filter_tensor);
  if (bias_tensor != nullptr) {
    conv_op.AddInputTensor(*bias_tensor);
  }

  TensorWrapper& output_tensor = *converted_output_tensor;
  conv_op.AddOutputTensor(output_tensor);
  // TODO: fused activation

  // stride param
  const std::array<std::uint32_t, 2> stride_data{stride_h, stride_w};
  const std::vector<std::uint32_t> stride_shape{2};
  auto& stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(decltype(stride_data)::value_type) * stride_data.size(),
      stride_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_STRIDE, stride_tensor);

  // dilation param
  const std::array<std::uint32_t, 2> dilation_data{dilation_h, dilation_w};
  const std::vector<std::uint32_t> dilation_shape{2};
  auto& dilation_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, dilation_shape,
      sizeof(decltype(dilation_data)::value_type) * dilation_data.size(),
      dilation_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_DILATION, dilation_tensor);

  // padding param
  const auto [padding_before_height, padding_after_height] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kHeightIndex),
                                filter_tensor.GetDim(kHeightIndex), stride_h,
                                dilation_h, padding_type);
  const auto [padding_before_width, padding_after_width] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kWidthIndex),
                                filter_tensor.GetDim(kWidthIndex), stride_w,
                                dilation_w, padding_type);
  const std::array<std::uint32_t, 4> padding_data = {
      padding_before_height, padding_after_height, padding_before_width,
      padding_after_width};
  const std::vector<std::uint32_t> padding_shape{2, 2};
  auto& padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(decltype(padding_data)::value_type) * padding_data.size(),
      padding_data.data());
  conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, padding_tensor);

  // group param
  if ((input_tensor.GetDim(kChannelIndex) %
       filter_tensor.GetDim(kChannelIndex)) != 0) {
    QNN_LOG_WARNING(
        "The channels of the input tensor cannot be evenly divided by the "
        "channels of the filter tensor.");
  }
  if (const std::uint32_t groups = input_tensor.GetDim(kChannelIndex) /
                                   filter_tensor.GetDim(kChannelIndex);
      groups > 1) {
    conv_op.AddScalarParam<std::uint32_t>(QNN_OP_CONV_2D_PARAM_GROUP, groups);
  }

  ConvertFp16ActivationToFp32IfWeightOnlyQuantized(
      res, converted_output_tensor, *conv2d_output_tensor,
      is_int8_weight_only_quantized);

  // Fused activation if available.
  if (fused_activation != kLiteRtFusedActivationNone) {
    AddFusedActivationNode(res, fused_activation, *conv2d_output_tensor,
                           outputs[0].get());
    QNN_LOG_INFO("Adding fused activation node");
  }

  QNN_LOG_INFO("LiteRt Conv_2d Op built with:");
  for (auto& op : res) {
    QNN_LOG_INFO("    Op: %s", op.GetOpConfig().v1.name);
  }
  return res;
}

}  // namespace qnn
