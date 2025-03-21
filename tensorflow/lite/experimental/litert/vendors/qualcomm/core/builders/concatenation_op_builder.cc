// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/concatenation_op_builder.h"

#include <cstdint>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

std::vector<OpWrapper> BuildConcatenationOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const std::int32_t axis) {
  std::vector<OpWrapper> res;

  auto& concat_op = CreateOpWrapper(res, QNN_OP_CONCAT);
  for (const auto& input : inputs) {
    concat_op.AddInputTensor(input);
  }
  concat_op.AddOutputTensor(outputs[0]);

  std::uint32_t adjusted_axis =
      (axis >= 0) ? axis : axis + inputs[0].get().GetRank();
  concat_op.AddScalarParam<std::uint32_t>(QNN_OP_CONCAT_PARAM_AXIS,
                                          adjusted_axis);

  return res;
}

}  // namespace qnn
