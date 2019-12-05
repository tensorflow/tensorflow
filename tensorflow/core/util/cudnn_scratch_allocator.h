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

#ifndef TENSORFLOW_CORE_UTIL_CUDNN_SCRATCH_ALLOCATOR_H_
#define TENSORFLOW_CORE_UTIL_CUDNN_SCRATCH_ALLOCATOR_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

namespace tensorflow {

using stream_executor::ScratchAllocator;
using stream_executor::port::StatusOr;
using stream_executor::DeviceMemory;

// A helper to allocate temporary scratch memory for CUDNN ops. It
// takes the ownership of the underlying memory. The expectation is that the
// memory should be alive for the span of the cudnnXXX itself.
class CudnnAllocatorInTemp : public ScratchAllocator {
 public:
  explicit CudnnAllocatorInTemp(OpKernelContext* context);
  ~CudnnAllocatorInTemp() override;
  int64 GetMemoryLimitInBytes() override;
  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64 byte_size) override;
  int64 TotalByteSize() const;
  Tensor get_allocated_tensor(int index) const;

 private:
  int64 total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  std::vector<Tensor> allocated_tensors_;

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnAllocatorInTemp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_CUDNN_STREAM_ALLOCATOR_H_
