/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

void GPUDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  GPUUtil::CopyCPUTensorToGPU(cpu_tensor, this, device, device_tensor, done);
}

void GPUDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             const string& tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  GPUUtil::CopyGPUTensorToCPU(device, this, device_tensor, cpu_tensor, done);
}

}  // namespace tensorflow
