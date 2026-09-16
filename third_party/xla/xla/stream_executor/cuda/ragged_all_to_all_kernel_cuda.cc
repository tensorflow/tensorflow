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
#include <cstdint>

#include "absl/base/casts.h"
#include "third_party/nccl/nccl.h"
#include "third_party/nccl/nccl_device.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel_lib.cu.h"

namespace stream_executor::gpu {

template <int64_t kVectorSize>
__global__ void __launch_bounds__(128)
    RaggedAllToAllWithSymmetricMemoryKernelImpl(
        const void* __restrict__ input_ptr,
        ncclWindow_t output_ptrs_symmetric_memory, size_t output_sym_offset,
        const int64_t* __restrict__ input_offsets_ptr,
        const int64_t* __restrict__ send_sizes_ptr,
        const int64_t* __restrict__ output_offsets_ptr,
        int64_t num_updates_per_replica, int64_t num_row_elements) {
  using T = Vec<kVectorSize>;
  const T* typed_input_ptr = static_cast<const T* __restrict__>(input_ptr);

  T* output_ptr = static_cast<T* __restrict__>(ncclGetLsaPointer(
      output_ptrs_symmetric_memory, output_sym_offset, blockIdx.x));

  TransferDataToLsaPeer(typed_input_ptr, output_ptr, input_offsets_ptr,
                        send_sizes_ptr, output_offsets_ptr,
                        num_updates_per_replica, num_row_elements);
}

}  // namespace stream_executor::gpu

#define REGISTER_RAGGED_ALL_TO_ALL_KERNEL(VECTOR_SIZE)                         \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                              \
      RaggedAllToAllHostArrayPtrsKernelCuda##VECTOR_SIZE##Bytes,               \
      stream_executor::gpu::RaggedAllToAllKernel<VECTOR_SIZE>,                 \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {              \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(   \
            absl::bit_cast<void*>(                                             \
                &stream_executor::gpu::RaggedAllToAllKernelImpl<VECTOR_SIZE>), \
            "ragged_all_to_all_kernel_host_array_ptrs_" #VECTOR_SIZE "_bytes", \
            arity);                                                            \
      }));                                                                     \
                                                                               \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                              \
      RaggedAllToAllSymmetricMemoryKernelCuda##VECTOR_SIZE##Bytes,             \
      stream_executor::gpu::RaggedAllToAllWithSymmetricMemoryKernel<           \
          VECTOR_SIZE>,                                                        \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {              \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(   \
            absl::bit_cast<void*>(                                             \
                &stream_executor::gpu::                                        \
                    RaggedAllToAllWithSymmetricMemoryKernelImpl<VECTOR_SIZE>), \
            "ragged_all_to_all_kernel_symmetric_memory_" #VECTOR_SIZE          \
            "_bytes",                                                          \
            arity);                                                            \
      }));

// Register the kernel for different integer types using the macro
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(1);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(2);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(4);
REGISTER_RAGGED_ALL_TO_ALL_KERNEL(8);
