//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/pack_op_builder.h"

#include <cstdint>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildPackOp(TensorPool& tensor_pool,
                                   const std::vector<TensorWrapperRef>& inputs,
                                   const std::vector<TensorWrapperRef>& outputs,
                                   const int32_t axis) {
  std::vector<OpWrapper> res;

  // pack op with only one input would violate op definition of qnn
  // we'll replace it with reshape op
  if (inputs.size() == 1) {
    auto& op = CreateOpWrapper(res, QNN_OP_RESHAPE);
    op.AddInputTensor(inputs[0]);
    op.AddOutputTensor(outputs[0]);
    return res;
  }

  if (outputs[0].get().GetRank() != inputs[0].get().GetRank() + 1) {
    auto& concat_op = CreateOpWrapper(res, QNN_OP_CONCAT);
    for (const auto& input : inputs) {
      concat_op.AddInputTensor(input);
    }
    concat_op.AddOutputTensor(outputs[0]);
  } else {
    auto& pack_op = CreateOpWrapper(res, QNN_OP_PACK);
    for (const auto& input : inputs) {
      pack_op.AddInputTensor(input);
    }
    std::uint32_t adjusted_axis =
        axis < 0 ? axis + inputs[0].get().GetRank() : axis;
    pack_op.AddScalarParam<std::uint32_t>(QNN_OP_PACK_PARAM_AXIS,
                                          adjusted_axis);
    pack_op.AddOutputTensor(outputs[0]);
  }

  return res;
}

}  // namespace qnn
