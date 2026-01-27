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

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "absl/base/casts.h"
#include "third_party/gpus/cuda/include/cuda/atomic"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/buffer_debug_xor_checksum_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/tsl/platform/logging.h"

namespace se = stream_executor;

namespace {

__device__ unsigned int ThreadIdx() {
  return threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
         threadIdx.x;
}

__device__ unsigned int BlockIdx() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
         blockIdx.x;
}

// Based on
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int BLOCK_SIZE>
__device__ void WarpReduceXor(unsigned int tid, volatile uint32_t* data) {
  if (BLOCK_SIZE >= 64) data[tid] ^= data[tid + 32];
  if (BLOCK_SIZE >= 32) data[tid] ^= data[tid + 16];
  if (BLOCK_SIZE >= 16) data[tid] ^= data[tid + 8];
  if (BLOCK_SIZE >= 8) data[tid] ^= data[tid + 4];
  if (BLOCK_SIZE >= 4) data[tid] ^= data[tid + 2];
  if (BLOCK_SIZE >= 2) data[tid] ^= data[tid + 1];
}

// Calculates a XOR of all elements of `input` and puts the result in `output`.
//
// Optimized implementation based on
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// that takes advantage of `BLOCK_SIZE` threads.
//
// `BLOCK_SIZE` must be a power of 2 no larger than 1024.
template <unsigned int BLOCK_SIZE>
__device__ void ReduceXor(const uint32_t* input, uint64_t input_size,
                          uint32_t* output) {
  __shared__ uint32_t scratch[BLOCK_SIZE];

  assert(BlockIdx() == 0);
  const unsigned int tid = ThreadIdx();

  scratch[tid] = 0;
  for (unsigned int i = tid; i < input_size; i += BLOCK_SIZE) {
    scratch[tid] ^= input[i];
  }

  __syncthreads();

  if (BLOCK_SIZE >= 1024) {
    if (tid < 512) {
      scratch[tid] ^= scratch[tid + 512];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 512) {
    if (tid < 256) {
      scratch[tid] ^= scratch[tid + 256];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 256) {
    if (tid < 128) {
      scratch[tid] ^= scratch[tid + 128];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 128) {
    if (tid < 64) {
      scratch[tid] ^= scratch[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32) WarpReduceXor<BLOCK_SIZE>(tid, scratch);
  if (tid == 0) *output = scratch[0];
}

// Attempts to append the checksum of the `input` buffer to the `log_entries`,
// using `log_header` to track available capacity and used space.
//
// The log entry is tagged with `entry_id`. The checksum is parallelized as
// much as block dimensions allow it.
//
// If the log does not have enough space for the new entry, the entry is
// discarded.
//
// `input_size` is the size of the input buffer in bytes.
//
// LIMITATIONS:
// - Only a single thread block is supported.
// - Block dimensions must be a power of 2.
__global__ void AppendChecksum(xla::gpu::BufferDebugLogEntryId entry_id,
                               const uint8_t* input, uint64_t input_size,
                               xla::gpu::BufferDebugLogHeader* log_header,
                               xla::gpu::BufferDebugLogEntry* log_entries) {
  const uint32_t block_size = blockDim.x * blockDim.y * blockDim.z;
  const uint32_t* input_u32 = reinterpret_cast<const uint32_t*>(input);
  const uint64_t input_u32_size = input_size / sizeof(uint32_t);
  uint32_t checksum = 0;

  assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
  if (BlockIdx() != 0) {
    return;
  }

  // https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/:
  // > CUDA architecture limits the numbers of threads per block (1024 threads
  // > per block limit).
  switch (block_size) {
    case 1024:
      ReduceXor<1024>(input_u32, input_u32_size, &checksum);
      break;
    case 512:
      ReduceXor<512>(input_u32, input_u32_size, &checksum);
      break;
    case 256:
      ReduceXor<256>(input_u32, input_u32_size, &checksum);
      break;
    case 128:
      ReduceXor<128>(input_u32, input_u32_size, &checksum);
      break;
    case 64:
      ReduceXor<64>(input_u32, input_u32_size, &checksum);
      break;
    case 32:
      ReduceXor<32>(input_u32, input_u32_size, &checksum);
      break;
    case 16:
      ReduceXor<16>(input_u32, input_u32_size, &checksum);
      break;
    case 8:
      ReduceXor<8>(input_u32, input_u32_size, &checksum);
      break;
    case 4:
      ReduceXor<4>(input_u32, input_u32_size, &checksum);
      break;
    case 2:
      ReduceXor<2>(input_u32, input_u32_size, &checksum);
      break;
    case 1:
      ReduceXor<1>(input_u32, input_u32_size, &checksum);
      break;
    default:
      // Unsupported block size.
      assert(false);
      return;
  }

  if (ThreadIdx() == 0) {
    const size_t last_chunk_size = input_size % sizeof(uint32_t);
    uint32_t last_chunk = 0;
    memcpy(&last_chunk, input + input_u32_size * sizeof(uint32_t),
           last_chunk_size);
    checksum ^= last_chunk;

    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
        checksum_log_write_idx(log_header->write_idx);
#if __CUDA_ARCH__ >= 600
    const uint32_t write_idx = checksum_log_write_idx.fetch_add(1);
    if (write_idx < log_header->capacity) {
      log_entries[write_idx] = {entry_id, checksum};
    }
#else
    // Our toolchains generate a fetch_add PTX instructions with system scope,
    // which is not supported on pre-Pascal architectures.
    assert(false);
#endif
  }
}

se::KernelLoaderSpec GetChecksumKernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&AppendChecksum), "BufferDebugXorChecksumKernel",
      arity);
}

}  // namespace

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugXorChecksumKernel, se::gpu::BufferDebugXorChecksumKernel,
    se::cuda::kCudaPlatformId, GetChecksumKernelSpec);
