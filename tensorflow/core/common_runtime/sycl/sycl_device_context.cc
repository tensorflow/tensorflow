/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/sycl/sycl_device_context.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

void SYCLDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                              Device* device,
                                              Tensor* device_tensor,
                                              StatusCallback done) const {
  const int64 total_bytes = cpu_tensor->TotalBytes();
  if (total_bytes > 0) {
    const void* src_ptr = DMAHelper::base(cpu_tensor);
    void* dst_ptr = DMAHelper::base(device_tensor);
    ::memcpy(dst_ptr, src_ptr, total_bytes);
  }
  done(Status::OK());
}

void SYCLDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                              StringPiece edge_name,
                                              Device* device,
                                              Tensor* cpu_tensor,
                                              StatusCallback done) {
  const int64 total_bytes = device_tensor->TotalBytes();
  if (total_bytes > 0) {
    const void* src_ptr = DMAHelper::base(device_tensor);
    void* dst_ptr = DMAHelper::base(cpu_tensor);
    ::memcpy(dst_ptr, src_ptr, total_bytes);
  }
  done(Status::OK());
}

}  // namespace tensorflow
