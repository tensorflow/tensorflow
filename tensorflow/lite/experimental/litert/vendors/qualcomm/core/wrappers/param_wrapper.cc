// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/param_wrapper.h"

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {

void ScalarParamWrapper::CloneTo(Qnn_Param_t& dst) const {
  dst.name = name_;
  dst.paramType = QNN_PARAMTYPE_SCALAR;
  dst.scalarParam = qnn_scalar_;
}

TensorParamWrapper::TensorParamWrapper(const char* name,
                                       const TensorWrapper& tensor)
    : name_{name}, tensor_{tensor} {}

void TensorParamWrapper::CloneTo(Qnn_Param_t& dst) const {
  dst.name = name_;
  dst.paramType = QNN_PARAMTYPE_TENSOR;
  tensor_.CloneTo(dst.tensorParam);
}

}  // namespace qnn
