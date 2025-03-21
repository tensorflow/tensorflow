// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

#include <string>
#include <utility>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

OpWrapper::OpWrapper(std::string name, const char* op_type)
    : type_name_{op_type}, name_{std::move(name)} {}

OpWrapper::OpWrapper(OpWrapper&& other)
    : type_name_{other.type_name_},
      name_{std::move(other.name_)},
      input_tensors_{std::move(other.input_tensors_)},
      output_tensors_{std::move(other.output_tensors_)},
      scalar_params_{std::move(other.scalar_params_)},
      tensor_params_{std::move(other.tensor_params_)},
      qnn_input_tensors_{std::move(other.qnn_input_tensors_)},
      qnn_output_tensors_{std::move(other.qnn_output_tensors_)},
      qnn_params_{std::move(other.qnn_params_)} {}

OpWrapper::~OpWrapper() = default;

void OpWrapper::AddInputTensor(const TensorWrapper& tensor) {
  input_tensors_.emplace_back(tensor);
}

void OpWrapper::AddOutputTensor(const TensorWrapper& tensor) {
  output_tensors_.emplace_back(tensor);
}

void OpWrapper::AddTensorParam(const char* name, const TensorWrapper& tensor) {
  tensor_params_.emplace_back(name, tensor);
}

Qnn_OpConfig_t OpWrapper::GetOpConfig() {
  Qnn_OpConfig_t qnn_op = QNN_OPCONFIG_INIT;
  qnn_op.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  qnn_op.v1.typeName = type_name_;
  qnn_op.v1.name = name_.data();
  // input tensors
  qnn_input_tensors_.reserve(input_tensors_.size());
  qnn_input_tensors_.clear();
  for (const auto& input_tensor : input_tensors_) {
    auto& back = qnn_input_tensors_.emplace_back();
    input_tensor.get().CloneTo(back);
  }
  qnn_op.v1.numOfInputs = qnn_input_tensors_.size();
  qnn_op.v1.inputTensors = qnn_input_tensors_.data();
  // output tensors
  qnn_output_tensors_.reserve(output_tensors_.size());
  qnn_output_tensors_.clear();
  for (const auto& output_tensor : output_tensors_) {
    auto& back = qnn_output_tensors_.emplace_back();
    output_tensor.get().CloneTo(back);
  }
  qnn_op.v1.numOfOutputs = qnn_output_tensors_.size();
  qnn_op.v1.outputTensors = qnn_output_tensors_.data();
  // params
  qnn_params_.reserve(scalar_params_.size() + tensor_params_.size());
  qnn_params_.clear();
  for (const auto& scalar_param : scalar_params_) {
    auto& back = qnn_params_.emplace_back();
    scalar_param.CloneTo(back);
  }
  for (const auto& tensor_param : tensor_params_) {
    auto& back = qnn_params_.emplace_back();
    tensor_param.CloneTo(back);
  }
  qnn_op.v1.numOfParams = qnn_params_.size();
  qnn_op.v1.params = qnn_params_.data();
  return qnn_op;
}

}  // namespace qnn
