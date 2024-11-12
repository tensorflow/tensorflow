/* Copyright 2019 The OpenXLA Authors.

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

// The CUDA implementation of the StreamExecutor functionality.
// CUDA inclusions are ideally confined to this implementation file.
//
// The notions from the StreamExecutor basically correspond to the CUDA streams
// programming model provided by the libcuda.so driver APIs, so we don't have
// to do much more than wrap the calls to the libraries appropriately.
#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_KERNEL_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/logging.h"

namespace stream_executor::gpu {

class RocmKernel : public Kernel {
 public:
  explicit RocmKernel(StreamExecutor* executor) : executor_(executor) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the StreamExecutor.
  ~RocmKernel() override { executor_->UnloadKernel(this); }

  // As arity cannot be reflected upon using the HIP API, the arity is
  // explicitly set during the RocmExecutor::GetKernel initialization process.
  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  // Simple accessor methods.
  hipFunction_t gpu_function() const { return rocm_function_; }
  void set_gpu_function(hipFunction_t rocm_function) {
    rocm_function_ = rocm_function;
  }

  // Collects metadata for the specified kernel.
  absl::StatusOr<KernelMetadata> GetKernelMetadata();

 private:
  StreamExecutor* executor_ = nullptr;

  hipFunction_t rocm_function_ = nullptr;  // wrapped HIP kernel handle
  unsigned arity_ = 0;  // number of formal parameters the kernel takes
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_KERNEL_H_
