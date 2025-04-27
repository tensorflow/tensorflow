/* Copyright 2025 The OpenXLA Authors.

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

#include "absl/base/casts.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/all_reduce_kernel_lib.cu.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"

#define REGISTER_ALL_REDUCE_KERNEL(TYPE)                                      \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                             \
      AllReduceKernelCuda##TYPE, stream_executor::gpu::AllReduceKernel<TYPE>, \
      stream_executor::cuda::kCudaPlatformId, ([] {                           \
        stream_executor::MultiKernelLoaderSpec spec(4);                       \
        spec.AddInProcessSymbol(                                              \
            absl::bit_cast<void*>(                                            \
                &stream_executor::gpu::AllReduceKernelImpl<TYPE>),            \
            "one_shot_all_reduce_" #TYPE);                                    \
        return spec;                                                          \
      }));

// Register the kernel for different types using the macro
REGISTER_ALL_REDUCE_KERNEL(float);
