// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/softmax_op_builder.h"

#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

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
