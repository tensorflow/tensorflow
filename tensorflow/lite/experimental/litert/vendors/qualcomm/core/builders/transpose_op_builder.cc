//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/transpose_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildTransposeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& perm_tensor = inputs[1];
  if (!perm_tensor.IsTensorStatic()) {
    // TODO: error log
    return res;
  }

  auto& transpose_op = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  transpose_op.AddInputTensor(inputs[0]);
  transpose_op.AddOutputTensor(outputs[0]);
  transpose_op.AddTensorParam(
      QNN_OP_TRANSPOSE_PARAM_PERM,
      tensor_pool.CloneStaticTensorFrom(perm_tensor, QNN_DATATYPE_UINT_32));

  return res;
}

}  // namespace qnn
