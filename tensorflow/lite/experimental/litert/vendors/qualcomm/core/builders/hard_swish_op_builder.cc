// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/hard_swish_op_builder.h"

namespace qnn {

namespace {
constexpr size_t kInputIndex = 0;
constexpr size_t kOutputIndex = 0;
}  // namespace

std::vector<OpWrapper> BuildHardSwishOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  OpWrapper& hard_swish_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NEURON);
  hard_swish_op.AddInputTensor(inputs[kInputIndex]);
  hard_swish_op.AddOutputTensor(outputs[kOutputIndex]);
  hard_swish_op.AddScalarParam<std::uint32_t>(
      QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION,
      QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH);

  return res;
}

}  // namespace qnn
