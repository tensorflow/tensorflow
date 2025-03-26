// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/quantize_op_builder.h"

#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

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

std::vector<OpWrapper> BuildDequantizeOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;
  const char* qnn_op = nullptr;
  if (inputs[0].get().IsF16() && outputs[0].get().IsF32()) {
    qnn_op = QNN_OP_CAST;
  } else {
    qnn_op = QNN_OP_DEQUANTIZE;
  }

  auto& quantize_op = CreateOpWrapper(res, qnn_op);
  quantize_op.AddInputTensor(inputs[0]);
  quantize_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
