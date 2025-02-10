//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/cast_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildCastOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& op = CreateOpWrapper(res, QNN_OP_CAST);
  op.AddInputTensor(inputs[0]);
  op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
