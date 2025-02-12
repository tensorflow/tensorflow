//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/gather_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildGatherOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const std::int32_t axis,
    const std::int32_t batch_dims) {
  std::vector<OpWrapper> res;

  if (batch_dims != 0) {
    // TODO: error log
    return res;
  }

  auto& gather_op = CreateOpWrapper(res, QNN_OP_GATHER);
  for (const auto& input : inputs) {
    gather_op.AddInputTensor(input);
  }
  for (const auto& output : outputs) {
    gather_op.AddOutputTensor(output);
  }
  const std::int32_t adjusted_axis =
      axis >= 0 ? axis : axis + inputs[0].get().GetRank();
  gather_op.AddScalarParam<std::int32_t>(QNN_OP_GATHER_PARAM_AXIS,
                                         adjusted_axis);

  return res;
}

}  // namespace qnn
