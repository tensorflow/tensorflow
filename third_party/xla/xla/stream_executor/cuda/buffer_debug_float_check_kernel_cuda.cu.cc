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

#include <cassert>
#include <cmath>
#include <cstdint>

#include "absl/base/casts.h"
#include "third_party/gpus/cuda/include/cuda/atomic"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"

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
__device__ void WarpReduceSum(unsigned int tid, volatile uint32_t* data) {
  if (BLOCK_SIZE >= 64) data[tid] += data[tid + 32];
  if (BLOCK_SIZE >= 32) data[tid] += data[tid + 16];
  if (BLOCK_SIZE >= 16) data[tid] += data[tid + 8];
  if (BLOCK_SIZE >= 8) data[tid] += data[tid + 4];
  if (BLOCK_SIZE >= 4) data[tid] += data[tid + 2];
  if (BLOCK_SIZE >= 2) data[tid] += data[tid + 1];
}

__device__ inline bool IsNan(float v) { return isnan(v); }
__device__ inline bool IsNan(__nv_bfloat16 v) { return __isnan(v); }
__device__ inline bool IsInf(float v) { return isinf(v); }
__device__ inline bool IsInf(__nv_bfloat16 v) { return __isinf(v); }
__device__ inline bool IsZero(float v) { return v == 0.0f; }
__device__ inline bool IsZero(__nv_bfloat16 v) {
  return v == __nv_bfloat16(0.0f);
}

// Calculates count of NaNs of all elements of `input` and puts result in
// `output`.
//
// Optimized implementation based on
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// that takes advantage of `BLOCK_SIZE` threads.
//
// `BLOCK_SIZE` must be a power of 2 no larger than 1024.
template <typename T, unsigned int BLOCK_SIZE>
__device__ void ReduceSum(const T* input, uint64_t input_size,
                          uint32_t* nan_counter, uint32_t* inf_counter,
                          uint32_t* zero_counter) {
  __shared__ uint32_t nan_count[BLOCK_SIZE];
  __shared__ uint32_t inf_count[BLOCK_SIZE];
  __shared__ uint32_t zero_count[BLOCK_SIZE];

  assert(BlockIdx() == 0);
  const unsigned int tid = ThreadIdx();

  nan_count[tid] = 0;
  inf_count[tid] = 0;
  zero_count[tid] = 0;
  for (unsigned int i = tid; i < input_size; i += BLOCK_SIZE) {
    if (IsNan(input[i])) {
      nan_count[tid]++;
    }
    if (IsInf(input[i])) {
      inf_count[tid]++;
    }
    if (IsZero(input[i])) {
      zero_count[tid]++;
    }
  }

  __syncthreads();

  if (BLOCK_SIZE >= 1024) {
    if (tid < 512) {
      nan_count[tid] += nan_count[tid + 512];
      inf_count[tid] += inf_count[tid + 512];
      zero_count[tid] += zero_count[tid + 512];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 512) {
    if (tid < 256) {
      nan_count[tid] += nan_count[tid + 256];
      inf_count[tid] += inf_count[tid + 256];
      zero_count[tid] += zero_count[tid + 256];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 256) {
    if (tid < 128) {
      nan_count[tid] += nan_count[tid + 128];
      inf_count[tid] += inf_count[tid + 128];
      zero_count[tid] += zero_count[tid + 128];
    }
    __syncthreads();
  }
  if (BLOCK_SIZE >= 128) {
    if (tid < 64) {
      nan_count[tid] += nan_count[tid + 64];
      inf_count[tid] += inf_count[tid + 64];
      zero_count[tid] += zero_count[tid + 64];
    }
    __syncthreads();
  }
  if (tid < 32) {
    WarpReduceSum<BLOCK_SIZE>(tid, nan_count);
    WarpReduceSum<BLOCK_SIZE>(tid, inf_count);
    WarpReduceSum<BLOCK_SIZE>(tid, zero_count);
  }
  if (tid == 0) {
    *nan_counter = nan_count[0];
    *inf_counter = inf_count[0];
    *zero_counter = zero_count[0];
  }
}

// Attempts to append the NaN count of the `input` buffer to the
// `float_check_entries`, using `log_header` to track available capacity and
// used space.
//
// The log entry is tagged with `entry_id`. The NaN count is parallelized as
// much as block dimensions allow it.
//
// If the log does not have enough space for the new entry, the entry is
// discarded.
//
// `input_size_in_bytes` is the size of the input buffer in bytes.
//
// LIMITATIONS:
// - Only a single thread block is supported.
// - Block dimensions must be a power of 2.
template <typename T>
__global__ void AppendFloatCheck(
    xla::gpu::BufferDebugLogEntryId entry_id, const T* input,
    uint64_t input_size_in_bytes, xla::gpu::BufferDebugLogHeader* log_header,
    xla::gpu::BufferDebugFloatCheckEntry* float_check_entries) {
  const uint32_t block_size = blockDim.x * blockDim.y * blockDim.z;
  const uint64_t input_size = input_size_in_bytes / sizeof(T);
  uint32_t nan_count = 0;
  uint32_t inf_count = 0;
  uint32_t zero_count = 0;

  assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
  if (BlockIdx() != 0) {
    return;
  }

  // https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/:
  // > CUDA architecture limits the numbers of threads per block (1024 threads
  // > per block limit).
  switch (block_size) {
    case 1024:
      ReduceSum<T, 1024>(input, input_size, &nan_count, &inf_count,
                         &zero_count);
      break;
    case 512:
      ReduceSum<T, 512>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 256:
      ReduceSum<T, 256>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 128:
      ReduceSum<T, 128>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 64:
      ReduceSum<T, 64>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 32:
      ReduceSum<T, 32>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 16:
      ReduceSum<T, 16>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 8:
      ReduceSum<T, 8>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 4:
      ReduceSum<T, 4>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 2:
      ReduceSum<T, 2>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    case 1:
      ReduceSum<T, 1>(input, input_size, &nan_count, &inf_count, &zero_count);
      break;
    default:
      // Unsupported block size.
      assert(false);
      return;
  }

  if (ThreadIdx() == 0) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system>
        nan_count_log_write_idx(log_header->write_idx);
#if __CUDA_ARCH__ >= 600
    const uint32_t write_idx = nan_count_log_write_idx.fetch_add(1);
    if (nan_count_log_write_idx.load() < log_header->capacity) {
      float_check_entries[write_idx] = xla::gpu::BufferDebugFloatCheckEntry{
          entry_id, nan_count, inf_count, zero_count};
    }
#else
    // Our toolchains generate a fetch_add PTX instructions with system scope,
    // which is not supported on pre-Pascal architectures.
    assert(false);
#endif
  }
}

se::KernelLoaderSpec GetFloatCheckF32KernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&AppendFloatCheck<float>),
      "BufferDebugFloatCheckF32Kernel", arity);
}

se::KernelLoaderSpec GetFloatCheckBf16KernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&AppendFloatCheck<__nv_bfloat16>),
      "BufferDebugFloatCheckBf16Kernel", arity);
}

}  // namespace

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF32Kernel, se::gpu::BufferDebugFloatCheckF32Kernel,
    se::cuda::kCudaPlatformId, GetFloatCheckF32KernelSpec);

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckBf16Kernel, se::gpu::BufferDebugFloatCheckBf16Kernel,
    se::cuda::kCudaPlatformId, GetFloatCheckBf16KernelSpec);
