// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {

constexpr size_t kInputIndex = 0;
constexpr size_t kAxisIndex = 1;
constexpr size_t kOutputIndex = 0;

}  // namespace

std::vector<OpWrapper> BuildCumsumOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool exclusive,
    const bool reverse) {
  std::vector<OpWrapper> res;

  const TensorWrapper& axis_tensor = inputs[kAxisIndex];
  if (!axis_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("Axis tensor must be static in Cumsum op.");
    return res;
  }

  const auto axis_data = axis_tensor.GetStaticTensorData<std::int32_t>();
  if (!axis_data.has_value()) {
    QNN_LOG_ERROR("Failed to get static axis tensor data.");
    return res;
  }

  std::uint32_t axis_value =
      (*axis_data)[0] >= 0
          ? (*axis_data)[0]
          : (*axis_data)[0] + inputs[kInputIndex].get().GetRank();

  OpWrapper& cumsum_op = CreateOpWrapper(res, QNN_OP_CUMULATIVE_SUM);
  cumsum_op.AddInputTensor(inputs[kInputIndex]);
  cumsum_op.AddOutputTensor(outputs[kOutputIndex]);
  cumsum_op.AddScalarParam<std::uint32_t>(QNN_OP_CUMULATIVE_SUM_PARAM_AXIS,
                                          axis_value);
  cumsum_op.AddScalarParam<bool>(QNN_OP_CUMULATIVE_SUM_PARAM_EXCLUSIVE,
                                 exclusive);
  cumsum_op.AddScalarParam<bool>(QNN_OP_CUMULATIVE_SUM_PARAM_REVERSE, reverse);

  return res;
}

}  // namespace qnn
