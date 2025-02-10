//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/tanh_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildTanhOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  CreateSimpleActivationOp(res, QNN_OP_TANH, inputs[0], outputs[0]);

  return res;
}

}  // namespace qnn
