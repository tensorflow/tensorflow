//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"

namespace qnn {

OpWrapper::OpWrapper(std::string name, const char* op_type)
    : name_{std::move(name)} {
  qnn_op_.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  qnn_op_.v1.typeName = op_type;
  qnn_op_.v1.name = name_.c_str();
}

OpWrapper::OpWrapper(const OpWrapper& other)
    : qnn_op_{other.qnn_op_},
      name_{other.name_},
      params_{other.params_},
      input_tensors_{other.input_tensors_},
      output_tensors_{other.output_tensors_} {
  qnn_op_.v1.name = name_.c_str();
  qnn_op_.v1.params = params_.data();
  qnn_op_.v1.inputTensors = input_tensors_.data();
  qnn_op_.v1.outputTensors = output_tensors_.data();
}

OpWrapper::OpWrapper(OpWrapper&& other)
    : qnn_op_{other.qnn_op_},
      name_{std::move(other.name_)},
      params_{std::move(other.params_)},
      input_tensors_{std::move(other.input_tensors_)},
      output_tensors_{std::move(other.output_tensors_)} {
  qnn_op_.v1.name = name_.c_str();
  qnn_op_.v1.params = params_.data();
  qnn_op_.v1.inputTensors = input_tensors_.data();
  qnn_op_.v1.outputTensors = output_tensors_.data();
}

OpWrapper::~OpWrapper() = default;

void OpWrapper::AddInputTensor(const TensorWrapper& tensor) {
  auto& back = input_tensors_.emplace_back();
  tensor.CloneTo(back);

  qnn_op_.v1.numOfInputs = input_tensors_.size();
  qnn_op_.v1.inputTensors = input_tensors_.data();
}

void OpWrapper::AddOutputTensor(const TensorWrapper& tensor) {
  auto& back = output_tensors_.emplace_back();
  tensor.CloneTo(back);

  qnn_op_.v1.numOfOutputs = output_tensors_.size();
  qnn_op_.v1.outputTensors = output_tensors_.data();
}

void OpWrapper::AddTensorParam(const char* name, const TensorWrapper& tensor) {
  TensorParamWrapper param_wrapper(name, tensor);

  auto& back = params_.emplace_back();
  param_wrapper.CloneTo(back);

  qnn_op_.v1.numOfParams = params_.size();
  qnn_op_.v1.params = params_.data();
}

const Qnn_OpConfig_t& OpWrapper::GetOpConfig() const { return qnn_op_; }

}  // namespace qnn
