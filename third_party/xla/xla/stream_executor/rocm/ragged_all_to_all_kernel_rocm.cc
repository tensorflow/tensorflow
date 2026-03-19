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

#include <cstddef>

#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel_lib.cu.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"

#define REGISTER_RAGGED_ALL_TO_ALL_KERNEL(VECTOR_SIZE)                         \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                              \
      RaggedAllToAllKernelRocm##VECTOR_SIZE##Bytes,                            \
      stream_executor::gpu::RaggedAllToAllKernel<VECTOR_SIZE>,                 \
      stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {              \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(   \
            absl::bit_cast<void*>(                                             \
                &stream_executor::gpu::RaggedAllToAllKernelImpl<VECTOR_SIZE>), \
            "ragged_all_to_all_kernel_" #VECTOR_SIZE "_bytes", arity);         \
      }));

// Register the kernel for different integer types using the macro
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(1);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(2);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(4);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(8);
