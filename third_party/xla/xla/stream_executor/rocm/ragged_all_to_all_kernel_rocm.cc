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

#include <array>
#include <cstddef>
#include <cstdint>

#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel_lib.cu.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"

#define REGISTER_RAGGED_ALL_TO_ALL_KERNEL(TYPE, BITS)                        \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                            \
      RaggedAllToAllKernelRocmUInt##BITS,                                    \
      stream_executor::gpu::RaggedAllToAllKernel<TYPE>,                      \
      stream_executor::rocm::kROCmPlatformId, ([](size_t arity) {            \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec( \
            absl::bit_cast<void*>(                                           \
                &stream_executor::gpu::RaggedAllToAllKernelImpl<TYPE>),      \
            "ragged_all_to_all_kernel_uint" #BITS, arity);                   \
      }));

// Register the kernel for different integer types using the macro
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(uint8_t, 8);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(uint16_t, 16);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(uint32_t, 32);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(uint64_t, 64);
