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
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/all_reduce_kernel_lib.cu.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/types.h"

#define REGISTER_ALL_REDUCE_KERNEL(SUFFIX, XLA_TYPE, NV_TYPE)         \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                     \
      AllReduceKernelCuda##SUFFIX,                                    \
      stream_executor::gpu::AllReduceKernel<XLA_TYPE>,                \
      stream_executor::cuda::kCudaPlatformId, ([] {                   \
        stream_executor::MultiKernelLoaderSpec spec(6);               \
        spec.AddInProcessSymbol(                                      \
            absl::bit_cast<void*>(                                    \
                &stream_executor::gpu::AllReduceKernelImpl<NV_TYPE>), \
            "one_shot_all_reduce_" #SUFFIX);                          \
        return spec;                                                  \
      }));

// Register the kernel for different types using the macro
REGISTER_ALL_REDUCE_KERNEL(bf16, xla::bfloat16, __nv_bfloat16);
REGISTER_ALL_REDUCE_KERNEL(f32, float, float);
