// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/select_op_builder.h"

#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildSelectOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  auto& select_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);
  for (const auto& input : inputs) {
    select_op.AddInputTensor(input);
  }
  select_op.AddOutputTensor(outputs[0]);

  return res;
}

}  // namespace qnn
