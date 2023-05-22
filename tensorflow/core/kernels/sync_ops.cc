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

#include "tensorflow/core/framework/op_kernel.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace {

class SyncDeviceOp : public OpKernel {
 public:
  explicit SyncDeviceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SyncDeviceOp);
};

REGISTER_KERNEL_BUILDER(Name("SyncDevice").Device(DEVICE_DEFAULT),
                        SyncDeviceOp);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
class SyncDeviceGpuOp : public OpKernel {
 public:
  explicit SyncDeviceGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const DeviceBase::AcceleratorDeviceInfo* info =
        context->device()->tensorflow_accelerator_device_info();
    if (info && info->stream) {
      OP_REQUIRES_OK(context, info->stream->BlockHostUntilDone());
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SyncDeviceGpuOp);
};

REGISTER_KERNEL_BUILDER(Name("SyncDevice").Device(DEVICE_GPU), SyncDeviceGpuOp);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace
}  // namespace tensorflow
