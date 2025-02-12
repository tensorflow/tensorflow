//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/softmax_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildSoftmaxOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const float beta) {
  std::vector<OpWrapper> res;

  auto& softmax_op = CreateOpWrapper(res, QNN_OP_SOFTMAX);
  softmax_op.AddInputTensor(inputs[0]);
  softmax_op.AddOutputTensor(outputs[0]);
  softmax_op.AddScalarParam<float>(QNN_OP_SOFTMAX_PARAM_BETA, beta);

  return res;
}

}  // namespace qnn
