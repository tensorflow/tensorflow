// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/pad_op_builder.h"

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"

namespace qnn {

namespace {

constexpr size_t kInputIndex = 0;
constexpr size_t kPadAmountIndex = 1;
constexpr size_t kPadConstValueIndex = 2;
constexpr size_t kOutputIndex = 0;

}  // namespace

std::vector<OpWrapper> BuildPadOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;

  TensorWrapper& pad_tensor = inputs[kPadAmountIndex];
  if (!pad_tensor.IsTensorStatic()) {
    QNN_LOG_ERROR("QNN only support static pad amount tensor.")
    return res;
  }

  auto* converted_pad_tensor =
      tensor_pool.ConvertStaticTensorFrom<std::uint32_t>(pad_tensor);
  if (converted_pad_tensor == nullptr) {
    QNN_LOG_ERROR("Failed to convert uint32 pad amount tensor.")
    return res;
  }

  OpWrapper& pad_op = CreateOpWrapper(res, QNN_OP_PAD);
  TensorWrapper& input_tensor = inputs[kInputIndex];
  pad_op.AddInputTensor(input_tensor);
  pad_op.AddOutputTensor(outputs[kOutputIndex]);
  pad_op.AddScalarParam<std::uint32_t>(QNN_OP_PAD_PARAM_SCHEME,
                                       QNN_OP_PAD_SCHEME_CONSTANT);
  pad_op.AddTensorParam(QNN_OP_PAD_PARAM_PAD_AMOUNT, *converted_pad_tensor);

  if (input_tensor.IsQuant8() || input_tensor.IsQuant16()) {
    std::int32_t pad_const_value = 0;
    if (inputs.size() >= kPadConstValueIndex + 1) {
      const auto pad_const_data =
          inputs[kPadConstValueIndex].get().GetStaticTensorData<std::int32_t>();
      if (!pad_const_data.has_value()) {
        QNN_LOG_ERROR("Failed to get pad const value data.");
        return {};
      }
      pad_const_value = pad_const_data.value()[0];
    } else {
      if (std::holds_alternative<ScaleOffsetQuantizeParamsWrapper>(
              input_tensor.GetQuantParams())) {
        const auto& quant_param = std::get<ScaleOffsetQuantizeParamsWrapper>(
            input_tensor.GetQuantParams());
        pad_const_value = (pad_const_value / quant_param.GetScale()) -
                          quant_param.GetOffset();
      } else {
        QNN_LOG_ERROR(
            "Unsupported quantization type type for pad const value tensor.");
        return {};
      }
    }
    pad_op.AddScalarParam<std::int32_t>(QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE,
                                        pad_const_value);

  } else if (input_tensor.IsF16() || input_tensor.IsF32()) {
    float pad_const_value = 0;
    if (inputs.size() >= kPadConstValueIndex + 1) {
      const auto pad_const_data =
          inputs[kPadConstValueIndex].get().GetStaticTensorData<float>();
      if (!pad_const_data.has_value()) {
        QNN_LOG_ERROR("Failed to get pad const value data.");
        return {};
      }
      pad_const_value = pad_const_data.value()[0];
    }
    pad_op.AddScalarParam<float>(QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE,
                                 pad_const_value);
  } else {
    QNN_LOG_ERROR("Unsupported input tensor type.")
    return {};
  }

  return res;
}

}  // namespace qnn
