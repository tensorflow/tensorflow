/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/next_pluggable_device/c_plugin_variable.h"

#include "absl/status/status.h"
#include "tensorflow/c/experimental/next_pluggable_device/c_api.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_helper.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"

namespace tensorflow {

CPluginVariable::~CPluginVariable() { TF_DeleteVariableInfo(var_info_); }

absl::Status CPluginVariable::GetTensorInternal() {
  // Note: we assume once a variable is initialized, it's underlying tensor
  // won't change during it's lifecycle.
  if (tensor_obtained_) {
    return absl::OkStatus();
  }
  TF_StatusPtr c_status_ptr(TF_NewStatus());
  TF_Tensor* c_tensor =
      TF_GetTensorFromVariableInfo(var_info_, c_status_ptr.get());
  TF_TensorPtr c_tensor_ptr(c_tensor);
  if (TF_GetCode(c_status_ptr.get()) != TF_OK) {
    return StatusFromTF_Status(c_status_ptr.get());
  }
  TF_RETURN_IF_ERROR(TF_TensorToTensor(c_tensor_ptr.get(), &tensor_));
  tensor_obtained_ = true;
  return absl::OkStatus();
}

absl::Status CPluginVariable::GetTensor(const Tensor** result_tensor) {
  TF_RETURN_IF_ERROR(GetTensorInternal());
  *result_tensor = &tensor_;
  return absl::OkStatus();
}

absl::Status CPluginVariable::GetMutableTensor(Tensor** result_tensor) {
  // Note: we assume once a variable is initialized, it's underlying tensor
  // won't change during it's lifecycle.
  TF_RETURN_IF_ERROR(GetTensorInternal());
  *result_tensor = &tensor_;
  return absl::OkStatus();
}

}  // namespace tensorflow
