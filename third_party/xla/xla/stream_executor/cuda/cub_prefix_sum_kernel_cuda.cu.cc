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

#include "cub/block/block_scan.cuh"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/prefix_sum_kernel.h"
#include "xla/stream_executor/kernel_spec.h"

namespace se = stream_executor;

namespace stream_executor::cuda {
namespace {

template <unsigned int BLOCK_SIZE, typename ElementT>
__device__ void RowPrefixSum(const ElementT* data_in, ElementT* data_out,
                             size_t num_items) {
  // `BLOCK_SIZE` must be a power of 2 no larger than 512.
  static_assert(BLOCK_SIZE <= 512 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0);
  using BlockScan = cub::BlockScan<ElementT, BLOCK_SIZE>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  ElementT total = 0;
  size_t thread_idx =
      ((threadIdx.z * blockDim.y) + threadIdx.y) * blockDim.x + threadIdx.x;
  for (size_t offset = thread_idx; offset < num_items; offset += BLOCK_SIZE) {
    if (offset < num_items) {
      ElementT thread_data = data_in[offset];
      ElementT block_aggregate;
      BlockScan(temp_storage)
          .InclusiveSum(thread_data, thread_data, block_aggregate);
      data_out[offset] = thread_data + total;
      total += block_aggregate;
      __syncthreads();
    }
  }
}

template <typename ElementT>
__global__ void PrefixSum(const void* data_in, void* data_out,
                          size_t num_items) {
  const ElementT* data_in_typed = static_cast<const ElementT*>(data_in);
  ElementT* data_out_typed = static_cast<ElementT*>(data_out);
  int64_t block_idx =
      ((static_cast<int64_t>(blockIdx.z) * gridDim.y) + blockIdx.y) *
          gridDim.x +
      blockIdx.x;
  int64_t row_offset = block_idx * num_items;
  // https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/:
  // CUDA architecture limits the numbers of threads per block (1024 threads
  // per block limit). We need to limit it to 512 to avoid running out of shared
  // memory with 8 byte data types.
  switch (blockDim.x * blockDim.y * blockDim.z) {
    case 512:
      RowPrefixSum<512>(data_in_typed + row_offset, data_out_typed + row_offset,
                        num_items);
      break;
    case 256:
      RowPrefixSum<256>(data_in_typed + row_offset, data_out_typed + row_offset,
                        num_items);
      break;
    case 128:
      RowPrefixSum<128>(data_in_typed + row_offset, data_out_typed + row_offset,
                        num_items);
      break;
    case 64:
      RowPrefixSum<64>(data_in_typed + row_offset, data_out_typed + row_offset,
                       num_items);
      break;
    case 32:
      RowPrefixSum<32>(data_in_typed + row_offset, data_out_typed + row_offset,
                       num_items);
      break;
    case 16:
      RowPrefixSum<16>(data_in_typed + row_offset, data_out_typed + row_offset,
                       num_items);
      break;
    case 8:
      RowPrefixSum<8>(data_in_typed + row_offset, data_out_typed + row_offset,
                      num_items);
      break;
    case 4:
      RowPrefixSum<4>(data_in_typed + row_offset, data_out_typed + row_offset,
                      num_items);
      break;
    case 2:
      RowPrefixSum<2>(data_in_typed + row_offset, data_out_typed + row_offset,
                      num_items);
      break;
    case 1:
      RowPrefixSum<1>(data_in_typed + row_offset, data_out_typed + row_offset,
                      num_items);
      break;
    default:
      // Unsupported block size.
      assert(false);
      return;
  }
}

#define XLA_CUB_PREFIX_SUM_KERNEL_SPEC(primitive_type, native_type)          \
  se::KernelLoaderSpec GetPrefixSum##primitive_type##KernelSpec(int arity) { \
    return se::KernelLoaderSpec::CreateInProcessSymbolSpec(                  \
        absl::bit_cast<void*>(&PrefixSum<native_type>),                      \
        "PrefixSum##primitive_type##Kernel", arity);                         \
  }

// Floating point types.
#ifdef CUB_TYPE_BF16
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(BF16, __nv_bfloat16)
#endif
#ifdef CUB_TYPE_F16
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(F16, __half)
#endif
#ifdef CUB_TYPE_F32
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(F32, float)
#endif
#ifdef CUB_TYPE_F64
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(F64, double)
#endif

// Signed integer types.
#ifdef CUB_TYPE_S8
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(S8, int8_t)
#endif
#ifdef CUB_TYPE_S16
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(S16, int16_t)
#endif
#ifdef CUB_TYPE_S32
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(S32, int32_t)
#endif
#ifdef CUB_TYPE_S64
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(S64, int64_t)
#endif

// Unsigned integer types.
#ifdef CUB_TYPE_U8
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(U8, uint8_t)
#endif
#ifdef CUB_TYPE_U16
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(U16, uint16_t)
#endif
#ifdef CUB_TYPE_U32
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(U32, uint32_t)
#endif
#ifdef CUB_TYPE_U64
XLA_CUB_PREFIX_SUM_KERNEL_SPEC(U64, uint64_t)
#endif

}  // namespace

#define REGISTER_PREFIX_SUM_KERNEL(primitive_type)                 \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                  \
      PrefixSum##primitive_type##Kernel,                           \
      se::gpu::PrefixSum##primitive_type##Kernel, kCudaPlatformId, \
      GetPrefixSum##primitive_type##KernelSpec)

#ifdef CUB_TYPE_BF16
REGISTER_PREFIX_SUM_KERNEL(BF16)
#endif
#ifdef CUB_TYPE_F16
REGISTER_PREFIX_SUM_KERNEL(F16)
#endif
#ifdef CUB_TYPE_F32
REGISTER_PREFIX_SUM_KERNEL(F32)
#endif
#ifdef CUB_TYPE_F64
REGISTER_PREFIX_SUM_KERNEL(F64)
#endif
#ifdef CUB_TYPE_S8
REGISTER_PREFIX_SUM_KERNEL(S8)
#endif
#ifdef CUB_TYPE_S16
REGISTER_PREFIX_SUM_KERNEL(S16)
#endif
#ifdef CUB_TYPE_S32
REGISTER_PREFIX_SUM_KERNEL(S32)
#endif
#ifdef CUB_TYPE_S64
REGISTER_PREFIX_SUM_KERNEL(S64)
#endif
#ifdef CUB_TYPE_U8
REGISTER_PREFIX_SUM_KERNEL(U8)
#endif
#ifdef CUB_TYPE_U16
REGISTER_PREFIX_SUM_KERNEL(U16)
#endif
#ifdef CUB_TYPE_U32
REGISTER_PREFIX_SUM_KERNEL(U32)
#endif
#ifdef CUB_TYPE_U64
REGISTER_PREFIX_SUM_KERNEL(U64)
#endif

}  // namespace stream_executor::cuda
