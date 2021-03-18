/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_context.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

void PluggableDeviceContext::CopyCPUTensorToDevice(
    const Tensor* cpu_tensor, Device* device, Tensor* device_tensor,
    StatusCallback done, bool sync_dst_compute) const {
  PluggableDeviceUtil::CopyCPUTensorToPluggableDevice(
      cpu_tensor, this, device, device_tensor, done, sync_dst_compute);
}

void PluggableDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                                   StringPiece tensor_name,
                                                   Device* device,
                                                   Tensor* cpu_tensor,
                                                   StatusCallback done) {
  PluggableDeviceUtil::CopyPluggableDeviceTensorToCPU(
      device, this, device_tensor, cpu_tensor, done);
}

void PluggableDeviceContext::CopyTensorInSameDevice(const Tensor* input_tensor,
                                                    Device* device,
                                                    Tensor* output_tensor,
                                                    StatusCallback done) const {
  PluggableDeviceUtil::CopyPluggableDeviceTensorToSameDevice(
      device, this, input_tensor, output_tensor, done);
}

Status PluggableDeviceContext::ThenExecute(Device* device, se::Stream* stream,
                                           std::function<void()> func) {
  const DeviceBase::GpuDeviceInfo* device_info =
      device->tensorflow_gpu_device_info();
  device_info->event_mgr->ThenExecute(stream, func);
  return Status::OK();
}

bool PluggableDeviceContext::IsPluggableDevice() { return true; }

}  // namespace tensorflow
