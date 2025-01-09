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

// GPU (SYCL / ROCm / CUDA) specific type handle resolution

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_

#if TENSORFLOW_USE_SYCL

#include "sycl/sycl.hpp"

#elif TENSORFLOW_USE_ROCM

#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hiprand/hiprand.h"

#else  // CUDA

#include "third_party/gpus/cuda/include/cuda.h"

#endif

namespace stream_executor {
namespace gpu {

#if TENSORFLOW_USE_SYCL

using GpuStreamHandle = ::sycl::queue*;

#elif TENSORFLOW_USE_ROCM

using GpuStreamHandle = hipStream_t;
#else  // CUDA

using GpuStreamHandle = CUstream;

#endif

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_TYPES_H_
