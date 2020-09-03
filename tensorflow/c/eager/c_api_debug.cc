/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_tensor_debug_info_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status_internal.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/platform/status.h"

using tensorflow::string;

namespace {

std::vector<tensorflow::int64> TensorShapeAsVector(
    const tensorflow::TensorHandle& handle, tensorflow::Status* status) {
  std::vector<tensorflow::int64> shape;
  int rank = -1;
  *status = handle.NumDims(&rank);
  if (!status->ok()) {
    return shape;
  }
  shape.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    tensorflow::int64 dim;
    *status = handle.Dim(i, &dim);
    if (!status->ok()) {
      return shape;
    }
    shape.push_back(dim);
  }
  return shape;
}

}  // namespace

extern "C" {

TF_CAPI_EXPORT extern TFE_TensorDebugInfo* TFE_TensorHandleTensorDebugInfo(
    TFE_TensorHandle* h, TF_Status* status) {
  tensorflow::TensorHandle* handle =
      TensorHandleFromInterface(tensorflow::unwrap(h));
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }

  std::vector<tensorflow::int64> dev_dims =
      TensorShapeAsVector(*handle, &status->status);
  if (!status->status.ok()) {
    return nullptr;
  }
  return new TFE_TensorDebugInfo(dev_dims);
}

TF_CAPI_EXPORT extern void TFE_DeleteTensorDebugInfo(
    TFE_TensorDebugInfo* debug_info) {
  delete debug_info;
}

TF_CAPI_EXPORT extern int TFE_TensorDebugInfoOnDeviceNumDims(
    TFE_TensorDebugInfo* debug_info) {
  return debug_info->dev_dims.size();
}

TF_CAPI_EXPORT extern int64_t TFE_TensorDebugInfoOnDeviceDim(
    TFE_TensorDebugInfo* debug_info, int dim_index) {
  return debug_info->dev_dims[dim_index];
}

}  // extern "C"
