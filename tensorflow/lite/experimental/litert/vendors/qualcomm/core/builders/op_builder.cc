//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"

namespace qnn {

OpWrapper& CreateOpWrapper(std::vector<OpWrapper>& ops, const char* op_type) {
  const auto op_count = ops.size();
  const auto name = "op_type_" + std::string(op_type) + "_op_count_" +
                    std::to_string(op_count);
  return ops.emplace_back(std::move(name), op_type);
}

OpWrapper& CreateSimpleActivationOp(std::vector<OpWrapper>& ops,
                                    const char* op_type,
                                    const TensorWrapper& input_tensor,
                                    const TensorWrapper& output_tensor) {
  auto& ret = CreateOpWrapper(ops, op_type);
  ret.AddInputTensor(input_tensor);
  ret.AddOutputTensor(output_tensor);
  return ret;
}

/*
LiteRtStatus OpMapper::AddFusedActivationNode(
    const tflite::ActivationFunctionType activation,
    const TensorWrapper& input_tensor, const TensorWrapper& output_tensor) {
  switch (activation) {
    case tflite::ActivationFunctionType_RELU: {
      OpWrapper& activation_op =
          CreateSimpleActivationOp(QNN_OP_RELU, input_tensor, output_tensor);
      break;
    }
    case tflite::ActivationFunctionType_RELU_N1_TO_1: {
      OpWrapper& activation_op = CreateSimpleActivationOp(
          QNN_OP_RELU_MIN_MAX, input_tensor, output_tensor);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                          -1.f);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                          1.f);
      break;
    }
    case tflite::ActivationFunctionType_RELU6: {
      OpWrapper& activation_op = CreateSimpleActivationOp(
          QNN_OP_RELU_MIN_MAX, input_tensor, output_tensor);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE,
                                          0.f);
      activation_op.AddScalarParam<float>(QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE,
                                          6.f);
      break;
    }
    case tflite::ActivationFunctionType_TANH: {
      OpWrapper& activation_op =
          CreateSimpleActivationOp(QNN_OP_TANH, input_tensor, output_tensor);
      break;
    }
    default:
      return kLiteRtStatusErrorUnsupported;
  }

  return kLiteRtStatusOk;
}
*/

}  // namespace qnn
