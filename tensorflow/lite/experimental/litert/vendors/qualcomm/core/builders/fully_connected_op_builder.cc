// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"

#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

namespace {
constexpr int kBiasIdx = 2;
}

std::vector<OpWrapper> BuildFullyConnectedOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_num_dims) {
  std::vector<OpWrapper> res;

  OpWrapper& fully_connected_op = CreateOpWrapper(res, QNN_OP_FULLY_CONNECTED);

  TensorWrapper& input_tensor = inputs[0];
  fully_connected_op.AddInputTensor(input_tensor);
  TensorWrapper& weight_tensor = inputs[1];
  fully_connected_op.AddInputTensor(weight_tensor);
  if (inputs.size() - 1 >= kBiasIdx) {
    TensorWrapper& bias_tensor = inputs[kBiasIdx];
    fully_connected_op.AddInputTensor(bias_tensor);
  }

  TensorWrapper& output_tensor = outputs[0];
  if (keep_num_dims) {
    auto& input_dims = input_tensor.GetDims();
    std::uint32_t input_size = std::accumulate(
        input_dims.begin(), input_dims.end(), 1, std::multiplies<>());
    const std::uint32_t num_units = weight_tensor.GetDim(0);
    const std::uint32_t num_input_elem = weight_tensor.GetDim(1);

    // input_size must be divisible by num_input_elem. This should be validated
    // by QNN.
    const std::uint32_t batch_size = input_size / num_input_elem;
    // QNN output should always be rank 2
    qnn::TensorWrapper& fully_connected_out = tensor_pool.CloneNativeTensorFrom(
        output_tensor, {batch_size, num_units});

    fully_connected_op.AddOutputTensor(fully_connected_out);
    // TODO: fused activation

    qnn::OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_op.AddInputTensor(fully_connected_out);
    reshape_op.AddOutputTensor(output_tensor);
  } else {
    fully_connected_op.AddOutputTensor(outputs[0]);
    // TODO: fused activation
  }

  return res;
}

}  // namespace qnn
