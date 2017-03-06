/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {

// Op that measures the peak memory in bytes.
class MaxBytesInUseOp : public OpKernel {
 public:
  explicit MaxBytesInUseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Allocator* allocator =
        context->device()->GetAllocator(AllocatorAttributes());
    AllocatorStats allocator_stats;
    allocator->GetStats(&allocator_stats);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<int64>()() = allocator_stats.max_bytes_in_use;
  }
};

// MallocExtension_GetAllocatedSize doesn't return the allocated size reliably
// for CPU allocators, so we register this op on GPU only.
REGISTER_KERNEL_BUILDER(
    Name("MaxBytesInUse").Device(DEVICE_GPU).HostMemory("out"),
    MaxBytesInUseOp);

}  // namespace tensorflow
