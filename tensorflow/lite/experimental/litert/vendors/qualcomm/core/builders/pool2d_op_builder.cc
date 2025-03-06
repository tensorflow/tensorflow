// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/pool2d_op_builder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {

constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;
constexpr size_t kHeightIndex = 1;
constexpr size_t kWidthIndex = 2;

std::vector<OpWrapper> BuildPool2dOp(
    TensorPool& tensor_pool, const char* op_type, const char* filter_param_name,
    const char* stride_param_name, const char* padding_param_name,
    const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t stride_height, const std::uint32_t stride_width,
    const std::uint32_t filter_height, const std::uint32_t filter_width,
    const PaddingType padding_type) {
  std::vector<OpWrapper> res;

  OpWrapper& pool_op = CreateOpWrapper(res, op_type);

  TensorWrapper& input_tensor = inputs[kInputIndex];
  pool_op.AddInputTensor(input_tensor);

  // filter param
  const std::vector<std::uint32_t> filter_shape{2};
  const std::array<std::uint32_t, 2> filter_data{filter_height, filter_width};
  TensorWrapper& filter_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, filter_shape,
      sizeof(decltype(filter_data)::value_type) * filter_data.size(),
      filter_data.data());
  pool_op.AddTensorParam(filter_param_name, filter_tensor);

  // stride param
  const std::vector<std::uint32_t> stride_shape{2};
  const std::array<std::uint32_t, 2> stride_data{stride_height, stride_width};
  TensorWrapper& stride_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
      sizeof(decltype(stride_data)::value_type) * stride_data.size(),
      stride_data.data());
  pool_op.AddTensorParam(stride_param_name, stride_tensor);

  // padding
  const auto [padding_before_height, padding_after_height] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kHeightIndex),
                                filter_height, stride_height, 1, padding_type);
  const auto [padding_before_width, padding_after_width] =
      ComputePaddingBeforeAfter(input_tensor.GetDim(kWidthIndex), filter_width,
                                stride_width, 1, padding_type);
  const std::vector<std::uint32_t> padding_shape{2, 2};
  const std::array<std::uint32_t, 4> padding_data{
      padding_before_height, padding_after_height, padding_before_width,
      padding_after_width};
  TensorWrapper& padding_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
      sizeof(decltype(padding_data)::value_type) * padding_data.size(),
      padding_data.data());
  pool_op.AddTensorParam(padding_param_name, padding_tensor);

  TensorWrapper& output_tensor = outputs[kOutputIndex];
  pool_op.AddOutputTensor(output_tensor);
  // TODO: fused activation

  return res;
}

}  // namespace

std::vector<OpWrapper> BuildMaxPoolOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t stride_height, const std::uint32_t stride_width,
    const std::uint32_t filter_height, const std::uint32_t filter_width,
    const PaddingType padding_type) {
  return BuildPool2dOp(
      tensor_pool, QNN_OP_POOL_MAX_2D, QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE,
      QNN_OP_POOL_MAX_2D_PARAM_STRIDE, QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT,
      inputs, outputs, stride_height, stride_width, filter_height, filter_width,
      padding_type);
}

std::vector<OpWrapper> BuildAveragePoolOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t stride_height, const std::uint32_t stride_width,
    const std::uint32_t filter_height, const std::uint32_t filter_width,
    const PaddingType padding_type) {
  return BuildPool2dOp(
      tensor_pool, QNN_OP_POOL_AVG_2D, QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE,
      QNN_OP_POOL_AVG_2D_PARAM_STRIDE, QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT,
      inputs, outputs, stride_height, stride_width, filter_height, filter_width,
      padding_type);
}

}  // namespace qnn
