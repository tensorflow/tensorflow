//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/mean_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildMeanOp(TensorPool& tensor_pool,
                                   const std::vector<TensorWrapperRef>& inputs,
                                   const std::vector<TensorWrapperRef>& outputs,
                                   const bool keep_dim) {
  std::vector<OpWrapper> res;

  TensorWrapper& axis_tensor = inputs[1];
  if (!axis_tensor.IsTensorStatic() || axis_tensor.GetRank() != 1) {
    // TODO: error log
    return res;
  }

  TensorWrapper& input_tensor = inputs[0];

  // TODO: cannot direcly cast
  auto* axis_data =
      reinterpret_cast<const std::int32_t*>(axis_tensor.GetStaticTensorData());
  std::vector<std::uint32_t> adjusted_axis_data;
  for (size_t i = 0; i < axis_tensor.GetDim(0); ++i) {
    std::uint32_t adjusted_axis = axis_data[i] >= 0
                                      ? axis_data[i]
                                      : axis_data[i] + input_tensor.GetRank();
    if (std::find(adjusted_axis_data.begin(), adjusted_axis_data.end(),
                  adjusted_axis) == adjusted_axis_data.end()) {
      adjusted_axis_data.emplace_back(adjusted_axis);
    }
  }
  TensorWrapper& adjusted_axis_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, axis_tensor.GetQuantParams(),
      {static_cast<const std::uint32_t>(adjusted_axis_data.size())},
      sizeof(std::uint32_t) * adjusted_axis_data.size(),
      adjusted_axis_data.data());

  auto& reduce_op = CreateOpWrapper(res, QNN_OP_REDUCE_MEAN);
  reduce_op.AddInputTensor(input_tensor);
  reduce_op.AddOutputTensor(outputs[0]);
  reduce_op.AddTensorParam(QNN_OP_REDUCE_MEAN_PARAM_AXES, adjusted_axis_tensor);
  reduce_op.AddScalarParam<bool>(QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS, keep_dim);

  return res;
}

}  // namespace qnn
