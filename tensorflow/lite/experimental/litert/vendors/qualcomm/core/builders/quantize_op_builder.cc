//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/quantize_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildQuantizeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  const char* qnn_op = nullptr;
  if (inputs[0].get().IsPerTensorQuantWithOffsetDiff(outputs[0].get())) {
    qnn_op = QNN_OP_CAST;
  } else if ((inputs[0].get().IsQuant8() || inputs[0].get().IsQuant16()) &&
             (outputs[0].get().IsQuant8() || outputs[0].get().IsQuant16())) {
    qnn_op = QNN_OP_CONVERT;
  } else {
    qnn_op = QNN_OP_QUANTIZE;
  }

  auto& quantize_op = CreateOpWrapper(res, qnn_op);
  quantize_op.AddInputTensor(inputs[0]);
  quantize_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
