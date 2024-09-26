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

#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"

namespace stream_executor::gpu {

// A GpuKernel is a `Kernel` that can be launched on a GPU. It allows
// access to the underlying GPU function through `gpu_function()`.
class GpuKernel : public Kernel {
 public:
  virtual GpuFunctionHandle gpu_function() const = 0;
};

inline const GpuKernel* AsGpuKernel(const Kernel* kernel) {
  return static_cast<const GpuKernel*>(kernel);
}

inline GpuKernel* AsGpuKernel(Kernel* kernel) {
  return static_cast<GpuKernel*>(kernel);
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_H_
