/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_context.h"

#include <functional>
#include <utility>

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/next_pluggable_device_api.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

namespace {

struct StatusCallbackInvocationParams {
  StatusCallback callback;
  TF_Status* status;
  TF_Tensor* c_cpu_tensor;
  TF_Tensor* c_device_tensor;
  TFNPD_DeviceEvent* event;
  const TFNPD_Api* api;

  ~StatusCallbackInvocationParams() {
    TF_DeleteStatus(status);
    // Release the C Tensors.
    TF_DeleteTensor(c_cpu_tensor);
    TF_DeleteTensor(c_device_tensor);
    api->TFNPD_DeviceEventDelete(event);
  }
};

// This function is passed to the AndThen C API. The AndThen C API waits for
// the event to become ready, and invoke this function.
void InvokeStatusCallbackFn(void* arg) {
  StatusCallbackInvocationParams* params =
      reinterpret_cast<StatusCallbackInvocationParams*>(arg);
  tensorflow::Status cc_status = StatusFromTF_Status(params->status);
  // Invokes the "done" callback here.
  params->callback(cc_status);
  // Explicitly delete the params after callback is done.
  delete params;
}

}  // namespace

NextPluggableDeviceContext::NextPluggableDeviceContext(int device_ordinal) {
  api_ = TfnpdApi();
  context_ = api_->TFNPD_DeviceContextCreate(device_ordinal);
}

NextPluggableDeviceContext::~NextPluggableDeviceContext() {
  api_->TFNPD_DeviceContextDelete(context_);
}

void NextPluggableDeviceContext::CopyDeviceTensorToCPU(
    const Tensor* device_tensor, absl::string_view tensor_name, Device* device,
    Tensor* cpu_tensor, StatusCallback done) {
  tsl::profiler::TraceMeProducer traceme(
      [] { return "NextPluggableDeviceContext::CopyDeviceTensorToCPU"; },
      tsl::profiler::ContextType::kGeneric);
  tensorflow::Status s;
  TF_Tensor* c_cpu_tensor = TF_TensorFromTensor(*cpu_tensor, &s);
  if (!s.ok()) {
    done(s);
  }
  TF_Tensor* c_device_tensor = TF_TensorFromTensor(*device_tensor, &s);
  if (!s.ok()) {
    done(s);
  }

  TF_Status* c_status = TF_NewStatus();
  TFNPD_DeviceEvent* event = api_->TFNPD_DeviceTensorToHostTensor(
      context_, c_device_tensor, c_cpu_tensor, c_status);

  // Store the std::function to the param because the "done" callback may have
  // captures and thus cannot be converted to a function pointer.
  StatusCallbackInvocationParams* params = new StatusCallbackInvocationParams{
      std::move(done), c_status, c_cpu_tensor, c_device_tensor, event, api_};

  api_->TFNPD_DeviceEventAndThen(event, &InvokeStatusCallbackFn,
                                 /*callback_arg=*/params);
}

void NextPluggableDeviceContext::CopyCPUTensorToDevice(
    const Tensor* cpu_tensor, Device* device, Tensor* device_tensor,
    StatusCallback done, bool sync_dst_compute) const {
  tsl::profiler::TraceMeProducer traceme(
      [] { return "NextPluggableDeviceContext::CopyCPUTensorToDevice"; },
      tsl::profiler::ContextType::kGeneric);
  tensorflow::Status s;
  TF_Tensor* c_cpu_tensor = TF_TensorFromTensor(*cpu_tensor, &s);
  if (!s.ok()) {
    done(s);
  }
  TF_Tensor* c_device_tensor = TF_TensorFromTensor(*device_tensor, &s);
  if (!s.ok()) {
    done(s);
  }

  TF_Status* c_status = TF_NewStatus();
  TFNPD_DeviceEvent* event = api_->TFNPD_HostTensorToDeviceTensor(
      context_, c_cpu_tensor, c_device_tensor, c_status);

  // Store the std::function to the param because the "done" callback may have
  // captures and thus cannot be converted to a function pointer.
  StatusCallbackInvocationParams* params = new StatusCallbackInvocationParams{
      std::move(done), c_status, c_cpu_tensor, c_device_tensor, event, api_};

  api_->TFNPD_DeviceEventAndThen(event, &InvokeStatusCallbackFn,
                                 /*callback_arg=*/params);
}

void NextPluggableDeviceContext::CopyTensorInSameDevice(
    const Tensor* input_tensor, Device* device, Tensor* output_tensor,
    StatusCallback done) const {
  done(errors::Unimplemented("Same-device copies not implemented."));
}

}  // namespace tensorflow
