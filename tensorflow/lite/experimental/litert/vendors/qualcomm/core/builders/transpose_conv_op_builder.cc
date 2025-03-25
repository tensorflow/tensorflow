// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/transpose_conv_op_builder.h"

#include <memory>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr size_t kFilterIndex = 1;
constexpr size_t kInputIndex = 2;
constexpr size_t kBiasIndex = 3;
constexpr size_t kOutputIndex = 0;
constexpr size_t kHeightIndex = 1;
constexpr size_t kWidthIndex = 2;
constexpr size_t kFilterHeightIndex = 1;
constexpr size_t kFilterWidthIndex = 2;
constexpr size_t kFilterChannelOutIndex = 0;
constexpr size_t kFilterChannelInIndex = 3;
}  // namespace

std::vector<OpWrapper> BuildTransposeConvOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const std::uint32_t stride_h,
    const std::uint32_t stride_w, const PaddingType padding_type) {
  std::vector<OpWrapper> res;

  // reshape filter
  TensorWrapper& filter_tensor = inputs[kFilterIndex];
  const std::vector<uint32_t>& filters_dims = filter_tensor.GetDims();
  auto& filter_quant_params = filter_tensor.GetQuantParams();
  std::vector<std::uint32_t> permute_dims{filters_dims[1], filters_dims[2],
                                          filters_dims[3], filters_dims[0]};
  if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
          filter_quant_params)) {
    auto& axis_quant_params =
        std::get<AxisScaleOffsetQuantizeParamsWrapper>(filter_quant_params);
    const std::array<std::int32_t, 4> new_axis{3, 0, 1, 2};
    axis_quant_params.SetAxis(new_axis[axis_quant_params.GetAxis()]);
  }

  size_t filter_bytes = filter_tensor.GetTensorBytes();
  TensorWrapper* transposed_filter_tensor = nullptr;
  if (filter_tensor.IsTensorStatic() &&
      filter_tensor.GetDataType() ==
          Qnn_DataType_t::QNN_DATATYPE_SFIXED_POINT_8) {
    auto filter_data = filter_tensor.GetStaticTensorData<std::int8_t>();
    std::vector<int8_t> transpose_weight_int8;
    TransposeFromOHWIToHWIO(filter_data.value(), filters_dims,
                            transpose_weight_int8);
    transposed_filter_tensor = &(tensor_pool.CreateStaticTensor(
        filter_tensor.GetDataType(), filter_quant_params, permute_dims,
        filter_bytes, transpose_weight_int8.data()));
  } else if (filter_tensor.IsTensorStatic() &&
             filter_tensor.GetDataType() ==
                 Qnn_DataType_t::QNN_DATATYPE_UFIXED_POINT_8) {
    auto filter_data = filter_tensor.GetStaticTensorData<std::uint8_t>();
    std::vector<uint8_t> transpose_weight_uint8;
    TransposeFromOHWIToHWIO(filter_data.value(), filters_dims,
                            transpose_weight_uint8);
    transposed_filter_tensor = &(tensor_pool.CreateStaticTensor(
        filter_tensor.GetDataType(), filter_quant_params, permute_dims,
        filter_bytes, transpose_weight_uint8.data()));
  } else {
    transposed_filter_tensor =
        &(tensor_pool.CloneNativeTensorFrom(filter_tensor, permute_dims));

    const std::vector<std::uint32_t> permute_shape{4};
    const std::array<std::uint32_t, 4> permute_data{kHeightIndex, kWidthIndex,
                                                    kFilterChannelInIndex,
                                                    kFilterChannelOutIndex};
    auto& permute_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, permute_shape,
        sizeof(decltype(permute_data)::value_type) * permute_data.size(),
        permute_data.data());

    OpWrapper& transpose_op = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
    transpose_op.AddInputTensor(filter_tensor);
    transpose_op.AddOutputTensor(*transposed_filter_tensor);
    transpose_op.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM, permute_tensor);
  }

  // conv
  OpWrapper& conv_op = CreateOpWrapper(res, QNN_OP_TRANSPOSE_CONV_2D);
  TensorWrapper& input_tensor = inputs[kInputIndex];
  conv_op.AddInputTensor(input_tensor);
  conv_op.AddInputTensor(*transposed_filter_tensor);
  if (inputs.size() - 1 >= kBiasIndex) {
    TensorWrapper& bias_tensor = inputs[kBiasIndex];
    // QNN only support per-tensor quant for bias,
    // and the scale and offset are both zero.
    bias_tensor.ConvertAxisScaleOffsetToScaleOffset();
    conv_op.AddInputTensor(bias_tensor);
  }

  TensorWrapper& output_tensor = outputs[kOutputIndex];
  conv_op.AddOutputTensor(output_tensor);
  // TODO: fused activation

  // stride param
  const std::array<std::uint32_t, 2> stride_data{stride_h, stride_w};
  const std::vector<std::uint32_t> stride_shape{2};
  auto& stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(decltype(stride_data)::value_type) * stride_data.size(),
      stride_data.data());
  conv_op.AddTensorParam(QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE, stride_tensor);

  // padding param
  const auto [padding_before_height, padding_after_height] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kHeightIndex),
                                filter_tensor.GetDim(kFilterHeightIndex),
                                stride_h, 1, padding_type);
  const auto [padding_before_width, padding_after_width] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kWidthIndex),
                                filter_tensor.GetDim(kFilterWidthIndex),
                                stride_w, 1, padding_type);
  const std::array<std::uint32_t, 4> padding_data = {
      padding_before_height, padding_after_height, padding_before_width,
      padding_after_width};
  const std::vector<std::uint32_t> padding_shape{2, 2};
  auto& padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(decltype(padding_data)::value_type) * padding_data.size(),
      padding_data.data());
  conv_op.AddTensorParam(QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT,
                         padding_tensor);

  return res;
}

}  // namespace qnn
