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
#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/logging.h"

namespace stream_executor::gpu {

class GpuKernel : public Kernel {
 public:
  explicit GpuKernel(GpuExecutor* gpu_executor)
      : gpu_executor_(gpu_executor),
        gpu_context_(gpu_executor->gpu_context()) {}

  // Note that the function is unloaded when the module is unloaded, and the
  // module that the function is contained in is owned by the GpuExecutor.
  ~GpuKernel() override { gpu_executor_->UnloadKernel(this); }

  // As arity cannot be reflected upon using the CUDA API, the arity is
  // explicitly set during the GpuExecutor::GetKernel initialization process.
  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

  // Simple accessor methods.
  GpuFunctionHandle gpu_function() const { return gpu_function_; }
  void set_gpu_function(GpuFunctionHandle gpu_function) {
    gpu_function_ = gpu_function;
  }

 private:
  GpuExecutor* gpu_executor_ = nullptr;
  Context* gpu_context_ = nullptr;  // context where kernel is loaded

  GpuFunctionHandle gpu_function_ = nullptr;  // wrapped CUDA kernel handle
  unsigned arity_ = 0;  // number of formal parameters the kernel takes
};

inline const GpuKernel* AsGpuKernel(const Kernel* kernel) {
  return static_cast<const GpuKernel*>(kernel);
}

inline GpuKernel* AsGpuKernel(Kernel* kernel) {
  return static_cast<GpuKernel*>(kernel);
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
