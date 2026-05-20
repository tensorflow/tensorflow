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

#include "absl/base/casts.h"
#include "third_party/nccl/nccl.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel_lib.cu.h"

using PtrArray = stream_executor::gpu::RaggedAllToAllOutputPtrs;

#define SINGLE_ARG(...) __VA_ARGS__

#define REGISTER_RAGGED_ALL_TO_ALL_KERNEL(VECTOR_SIZE)                         \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                              \
      RaggedAllToAllHostArrayPtrsKernelCuda##VECTOR_SIZE##Bytes,               \
      SINGLE_ARG(                                                              \
          stream_executor::gpu::RaggedAllToAllKernel<PtrArray, VECTOR_SIZE>),  \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {              \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(   \
            absl::bit_cast<void*>(&SINGLE_ARG(                                 \
                stream_executor::gpu::RaggedAllToAllKernelImpl<PtrArray,       \
                                                               VECTOR_SIZE>)), \
            "ragged_all_to_all_kernel_host_array_ptrs_" #VECTOR_SIZE "_bytes", \
            arity);                                                            \
      }));                                                                     \
                                                                               \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                              \
      RaggedAllToAllSymmetricMemoryKernelCuda##VECTOR_SIZE##Bytes,             \
      SINGLE_ARG(                                                              \
          stream_executor::gpu::RaggedAllToAllWithSymmetricMemoryKernel<       \
              VECTOR_SIZE>),                                                   \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {              \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(   \
            absl::bit_cast<void*>(                                             \
                &SINGLE_ARG(stream_executor::gpu::                             \
                                RaggedAllToAllWithSymmetricMemoryKernelImpl<   \
                                    VECTOR_SIZE>)),                            \
            "ragged_all_to_all_kernel_symmetric_memory_" #VECTOR_SIZE          \
            "_bytes",                                                          \
            arity);                                                            \
      }));                                                                     \
                                                                               \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                              \
      RaggedAllToAllDeviceAddressPtrsKernelCuda##VECTOR_SIZE##Bytes,           \
      SINGLE_ARG(stream_executor::gpu::RaggedAllToAllKernel<                   \
                 stream_executor::DeviceAddressBase, VECTOR_SIZE>),            \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {              \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(   \
            absl::bit_cast<void*>(&SINGLE_ARG(                                 \
                stream_executor::gpu::RaggedAllToAllKernelImpl<void**,         \
                                                               VECTOR_SIZE>)), \
            "ragged_all_to_all_kernel_device_address_ptrs_" #VECTOR_SIZE       \
            "_bytes",                                                          \
            arity);                                                            \
      }));

// Register the kernel for different integer types using the macro
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(1);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(2);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(4);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(8);
