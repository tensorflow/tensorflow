//  Copyright (c) Qualcomm Innovation Center, Inc.
//  All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/param_wrapper.h"

namespace qnn {

void ScalarParamWrapper::CloneTo(Qnn_Param_t& dst) const { dst = qnn_param_; }

TensorParamWrapper::TensorParamWrapper(const char* name,
                                       const TensorWrapper& tensor) {
  qnn_param_.name = name;
  qnn_param_.paramType = QNN_PARAMTYPE_TENSOR;
  tensor.CloneTo(qnn_param_.tensorParam);
}

void TensorParamWrapper::CloneTo(Qnn_Param_t& dst) const { dst = qnn_param_; }

}  // namespace qnn
