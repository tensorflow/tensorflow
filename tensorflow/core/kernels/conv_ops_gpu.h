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

#ifndef TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
#define TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_

#if GOOGLE_CUDA

#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// TODO(zhengxq): move this to gpu_util.h. The use of such wrappers is wide
// spread.
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory,
                                                    uint64 size) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory),
                                                size * sizeof(T));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}

// Get the Cudnn workspace limit from the environment variable, which is in MB.
// Return the workspace memory limit in bytes. If no value is set, return the
// default value.
int64 GetCudnnWorkspaceLimit(const string& envvar_in_mb,
                             int64 default_value_in_bytes);

// A class to provide scratch-space allocator for Stream-Executor Cudnn
// callback. TensorFlow is responsible for releasing the temporary buffers after
// the kernel finishes.
class CudnnScratchAllocator : public perftools::gputools::ScratchAllocator {
 public:
  virtual ~CudnnScratchAllocator() {}
  CudnnScratchAllocator(int64 memory_limit, OpKernelContext* context)
      : memory_limit_(memory_limit), context_(context) {}
  virtual int64 GetMemoryLimitInBytes(
      perftools::gputools::Stream* stream) override {
    return memory_limit_;
  }
  virtual perftools::gputools::port::StatusOr<
      perftools::gputools::DeviceMemory<uint8>>
  AllocateBytes(perftools::gputools::Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;

    AllocationAttributes allocation_attr;
    allocation_attr.no_retry_on_failure = true;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory,
        AllocatorAttributes(), allocation_attr));
    if (!allocation_status.ok()) {
      return perftools::gputools::port::StatusOr<
          perftools::gputools::DeviceMemory<uint8>>(
          AsDeviceMemory<uint8>(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return perftools::gputools::port::StatusOr<
        perftools::gputools::DeviceMemory<uint8>>(
        AsDeviceMemory(temporary_memory.flat<uint8>().data(),
                       temporary_memory.flat<uint8>().size()));
  }

 private:
  int64 memory_limit_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_CONV_OPS_GPU_H_
