//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/embedding_lookup_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildEmbeddingLookupOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& gather_op = CreateOpWrapper(res, QNN_OP_GATHER);
  gather_op.AddInputTensor(inputs[1]);
  gather_op.AddInputTensor(inputs[0]);
  for (const auto& output : outputs) {
    gather_op.AddOutputTensor(output);
  }
  gather_op.AddScalarParam<std::int32_t>(QNN_OP_GATHER_PARAM_AXIS, 0);
  return res;
}

}  // namespace qnn
