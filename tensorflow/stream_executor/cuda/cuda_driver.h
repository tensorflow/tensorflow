/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// CUDA userspace driver library wrapper functionality.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_

#include "tensorflow/stream_executor/gpu/gpu_driver.h"

namespace stream_executor {
namespace gpu {
// CUDAContext wraps a cuda CUcontext handle, and includes a unique id. The
// unique id is positive, and ids are not repeated within the process.
class GpuContext {
 public:
  GpuContext(CUcontext context, int64 id) : context_(context), id_(id) {}

  CUcontext context() const { return context_; }
  int64 id() const { return id_; }

  // Disallow copying and moving.
  GpuContext(GpuContext&&) = delete;
  GpuContext(const GpuContext&) = delete;
  GpuContext& operator=(GpuContext&&) = delete;
  GpuContext& operator=(const GpuContext&) = delete;

 private:
  CUcontext const context_;
  const int64 id_;
};

}  // namespace gpu

namespace cuda {

using MemorySpace = gpu::MemorySpace;

using CUDADriver = gpu::GpuDriver;

using ScopedActivateContext = gpu::ScopedActivateContext;

using CudaContext = gpu::GpuContext;

// Returns the current context set in CUDA. This is done by calling the cuda
// driver (e.g., this value is not our cached view of the current context).
CUcontext CurrentContextOrDie();

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
