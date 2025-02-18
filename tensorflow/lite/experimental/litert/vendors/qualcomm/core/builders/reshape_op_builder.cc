//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/reshape_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildReshapeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);
  reshape_op.AddInputTensor(inputs[0]);
  reshape_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
