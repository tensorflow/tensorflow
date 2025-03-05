// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/spatial_transform_op_builder.h"

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

std::vector<OpWrapper> BuildSpatialTransformOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const char* op_type,
    const char* block_param, const std::uint32_t block_size) {
  std::vector<OpWrapper> res;

  auto& spatial_transform_op = CreateOpWrapper(res, op_type);
  spatial_transform_op.AddInputTensor(inputs[kInputIndex]);
  spatial_transform_op.AddOutputTensor(outputs[kOutputIndex]);
  const std::array<std::uint32_t, 2> block_data = {block_size, block_size};
  const std::vector<std::uint32_t> block_dims{2};
  auto& block_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, block_dims,
      sizeof(decltype(block_dims)::value_type) * block_dims.size(),
      block_data.data());
  spatial_transform_op.AddTensorParam(block_param, block_tensor);

  return res;
}
}  // namespace

std::vector<OpWrapper> BuildDepthToSpaceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t block_size) {
  return BuildSpatialTransformOp(
      tensor_pool, inputs, outputs, QNN_OP_DEPTH_TO_SPACE,
      QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE, block_size);
}

std::vector<OpWrapper> BuildSpaceToDepthOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs,
    const std::uint32_t block_size) {
  return BuildSpatialTransformOp(
      tensor_pool, inputs, outputs, QNN_OP_SPACE_TO_DEPTH,
      QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE, block_size);
}
}  // namespace qnn
