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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api_helper.h"

#include <functional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_tensor_utils.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/status.h"

namespace tensorflow {

namespace {

struct TF_DeviceContext_Deleter {
  void operator()(TF_DeviceContext* c_device_context);
};

tsl::StatusCallback FromC(TF_StatusCallback* callback) {
  return [callback](absl::Status status) {
    TF_Status* c_status = TF_NewStatus();
    Set_TF_Status_from_Status(c_status, status);
    callback->callback(callback->context, c_status);
    TF_DeleteStatus(c_status);
  };
}

void CpuToDeviceThunk(void* context,
                      TF_DeviceContext_CopyCPUTensorToDevice_Params* params) {
  DeviceContext* device_context = static_cast<DeviceContext*>(context);
  Tensor* cpu_tensor = new Tensor();
  Tensor* device_tensor = new Tensor();
  tsl::StatusCallback done = [params, device_tensor,
                              cpu_tensor](absl::Status status) {
    delete cpu_tensor;
    // TODO: find a way to convert device tensor.
    // params->device_tensor = TF_TensorFromTensor(*device_tensor,
    //                                             &tensor_status);
    delete device_tensor;
    FromC(params->done)(status);
  };
  CopyTF_TensorToTensor(params->cpu_tensor, cpu_tensor);
  bool sync_dst_compute = params->sync_dst_compute;
  device_context->CopyCPUTensorToDevice(cpu_tensor, /* device = */ nullptr,
                                        device_tensor, std::move(done),
                                        sync_dst_compute);
}

void DeviceToCpuThunk(void* context,
                      TF_DeviceContext_CopyDeviceTensorToCPU_Params* params) {
  DeviceContext* device_context = static_cast<DeviceContext*>(context);
  Tensor* cpu_tensor = new Tensor();
  Tensor* device_tensor = new Tensor();
  tsl::StatusCallback done = [params, device_tensor,
                              cpu_tensor](absl::Status status) {
    delete device_tensor;
    params->cpu_tensor = CopyTensorToTF_Tensor(*cpu_tensor);
    delete cpu_tensor;
    FromC(params->done)(status);
  };
  std::string_view tensor_name(params->tensor_name, params->tensor_name_len);
  // TODO: Find a way to convert device tensor.
  device_context->CopyDeviceTensorToCPU(
      /* device_tensor = */ nullptr, tensor_name,
      /* device = */ nullptr, cpu_tensor, std::move(done));
}

void SameDeviceThunk(void* context,
                     TF_DeviceContext_CopyTensorInSameDevice_Params* params) {
  LOG(FATAL) << "Unimplemented";  // Crash OK
}

}  // namespace

void DeviceContext_Destroy(TF_DeviceContext* c_device_context) {}

void TF_DeviceContext_Deleter::operator()(TF_DeviceContext* c_device_context) {
  DeviceContext_Destroy(c_device_context);
  delete c_device_context;
}

TF_DeviceContext* DeviceContext_ToC(DeviceContext* device_context) {
  TF_DeviceContext* c_device_context = new TF_DeviceContext();
  c_device_context->device_context = static_cast<void*>(device_context);
  c_device_context->cpu_to_device_func = CpuToDeviceThunk;
  c_device_context->device_to_cpu_func = DeviceToCpuThunk;
  c_device_context->same_device_func = SameDeviceThunk;
  return c_device_context;
}

}  // namespace tensorflow
