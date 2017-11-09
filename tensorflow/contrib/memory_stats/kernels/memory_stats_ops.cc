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

// Base class of ops that collects statistics of the allocator of a device.
class MemoryStatsOp : public OpKernel {
 public:
  explicit MemoryStatsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    Allocator* allocator =
        context->device()->GetAllocator(AllocatorAttributes());
    AllocatorStats allocator_stats;
    allocator->GetStats(&allocator_stats);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({}), &output_tensor));
    output_tensor->scalar<int64>()() = ExtractAllocatorStats(allocator_stats);
  }

 protected:
  // Extracts a certain field (determined by subclasses) from an allocator
  // stats.
  virtual int64 ExtractAllocatorStats(
      const AllocatorStats& allocator_stats) const = 0;
};

// Op that measures current memory in bytes.
class BytesInUseOp : public MemoryStatsOp {
 public:
  explicit BytesInUseOp(OpKernelConstruction* context)
      : MemoryStatsOp(context) {}

 private:
  int64 ExtractAllocatorStats(
      const AllocatorStats& allocator_stats) const override {
    return allocator_stats.bytes_in_use;
  }
};

// Register this op on GPU only, see comment for MaxBytesInUse for reason
REGISTER_KERNEL_BUILDER(Name("BytesInUse").Device(DEVICE_GPU).HostMemory("out"),
                        BytesInUseOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("BytesInUse").Device(DEVICE_SYCL).HostMemory("out"), MaxBytesInUseOp);
#endif  // TENSORFLOW_USE_SYCL

// Op that measures the total memory (in bytes) of a device.
class BytesLimitOp : public MemoryStatsOp {
 public:
  explicit BytesLimitOp(OpKernelConstruction* context)
      : MemoryStatsOp(context) {}

 private:
  int64 ExtractAllocatorStats(
      const AllocatorStats& allocator_stats) const override {
    return allocator_stats.bytes_limit;
  }
};

REGISTER_KERNEL_BUILDER(Name("BytesLimit").Device(DEVICE_CPU), BytesLimitOp);
REGISTER_KERNEL_BUILDER(Name("BytesLimit").Device(DEVICE_GPU).HostMemory("out"),
                        BytesLimitOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("BytesLimit").Device(DEVICE_SYCL).HostMemory("out"),
                        BytesLimitOp);
#endif // TENSORFLOW_USE_SYCL

// Op that measures the peak memory in bytes.
class MaxBytesInUseOp : public MemoryStatsOp {
 public:
  explicit MaxBytesInUseOp(OpKernelConstruction* context)
      : MemoryStatsOp(context) {}

 private:
  int64 ExtractAllocatorStats(
      const AllocatorStats& allocator_stats) const override {
    return allocator_stats.max_bytes_in_use;
  }
};

// MallocExtension_GetAllocatedSize doesn't return the allocated size reliably
// for CPU allocators, so we register this op on GPU only.
REGISTER_KERNEL_BUILDER(
    Name("MaxBytesInUse").Device(DEVICE_GPU).HostMemory("out"),
    MaxBytesInUseOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("MaxBytesInUse").Device(DEVICE_SYCL).HostMemory("out"),
    MaxBytesInUseOp);
#endif // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
