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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

#include "absl/base/casts.h"
#include "third_party/gpus/cuda/include/cuda/atomic"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/util.h"

namespace se = stream_executor;

namespace {

using xla::gpu::FloatCheckResult;

// https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/:
// > CUDA architecture limits the numbers of threads per block (1024 threads
// > per block limit).
static constexpr uint64_t kBlockSize = 1024;
// warpSize is not a compile time constant on all OSS CI builds, but we need it
// to be one for static array initialization. We assert this value matches
// warpSize at runtime.
static constexpr uint64_t kWarpSize = 32;
static constexpr uint64_t kMaxWarpsPerBlock = kBlockSize / kWarpSize;
template <typename T>
static constexpr uint64_t kElementsPerMemoryAccess =
    std::max<uint64_t>(16 / sizeof(T), 1);
template <typename T>
using Chunk = std::array<T, kElementsPerMemoryAccess<T>>;

__device__ unsigned int ThreadIdx() {
  return threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
         threadIdx.x;
}

__device__ unsigned int BlockIdx() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
         blockIdx.x;
}

// Reduce a warp worth of values into a single one and have the 0th thread in
// the warp return it.
__device__ uint32_t WarpReduceSum(uint32_t value) {
  static constexpr uint32_t kFullMask = ~0;
  for (unsigned int offset = 1; offset < kWarpSize; offset <<= 1) {
    value += __shfl_down_sync(kFullMask, value, offset);
  }
  return value;
}

// Sum up a block worth of FloatCheckResults into a single one and have the 0th
// thread in the block return it.
__device__ FloatCheckResult BlockReduceSum(uint32_t tid,
                                           FloatCheckResult value) {
  assert(kWarpSize == warpSize);
  static_assert(kBlockSize == kWarpSize * kMaxWarpsPerBlock);
  // Required to do the second warp reduction.
  static_assert(kMaxWarpsPerBlock == kWarpSize);

  const size_t warp_idx = tid / kWarpSize;
  const size_t lane_idx = tid % kWarpSize;

  value.nan_count = WarpReduceSum(value.nan_count);
  value.inf_count = WarpReduceSum(value.inf_count);
  value.zero_count = WarpReduceSum(value.zero_count);

  __shared__ uint32_t scratch_nan[kMaxWarpsPerBlock];
  __shared__ uint32_t scratch_inf[kMaxWarpsPerBlock];
  __shared__ uint32_t scratch_zero[kMaxWarpsPerBlock];
  if (lane_idx == 0) {
    scratch_nan[warp_idx] = value.nan_count;
    scratch_inf[warp_idx] = value.inf_count;
    scratch_zero[warp_idx] = value.zero_count;
  }

  __syncthreads();
  // The first warp reduces the results from all warps.
  if (warp_idx == 0) {
    value.nan_count = scratch_nan[lane_idx];
    value.inf_count = scratch_inf[lane_idx];
    value.zero_count = scratch_zero[lane_idx];
    value.nan_count = WarpReduceSum(value.nan_count);
    value.inf_count = WarpReduceSum(value.inf_count);
    value.zero_count = WarpReduceSum(value.zero_count);
  } else {
    value.nan_count = 0;
    value.inf_count = 0;
    value.zero_count = 0;
  }

  return value;
}

__device__ inline bool IsNan(float v) { return isnan(v); }
__device__ inline bool IsNan(__nv_bfloat16 v) { return __isnan(v); }
__device__ inline bool IsInf(float v) { return isinf(v); }
__device__ inline bool IsInf(__nv_bfloat16 v) { return __isinf(v); }
__device__ inline bool IsZero(float v) { return v == 0.0f; }
__device__ inline bool IsZero(__nv_bfloat16 v) {
  return v == __nv_bfloat16(0.0f);
}

// Get a part of the input buffer current thread block is responsible for
// processing, assuming the load is spread up to max_blocks across the entire
// grid. If max_blocks is not provided, the entire grid is used.
template <typename T>
__device__ inline std::tuple<const T*, uint64_t> GetBlockInput(
    const T* input, uint64_t input_size,
    std::optional<uint64_t> max_blocks = std::nullopt) {
  size_t grid_size = gridDim.x * gridDim.y * gridDim.z;
  if (max_blocks.has_value()) {
    grid_size = std::min<size_t>(grid_size, *max_blocks);
  }
  const uint64_t max_block_input_size = xla::RoundUpTo(
      xla::CeilOfRatio(input_size, grid_size), kElementsPerMemoryAccess<T>);
  const uint64_t block_input_offset = BlockIdx() * max_block_input_size;
  const uint64_t block_input_size = std::min(
      max_block_input_size,
      input_size >= block_input_offset ? input_size - block_input_offset : 0);
  return {input + block_input_offset, block_input_size};
}

template <typename T>
__device__ FloatCheckResult CheckFloats(const T* input, uint64_t input_size,
                                        uint64_t max_blocks) {
  const unsigned int tid = ThreadIdx();
  const auto [block_input, block_input_size] =
      GetBlockInput(input, input_size, max_blocks);

  const Chunk<T>* chunked_input =
      reinterpret_cast<const Chunk<T>*>(block_input);
  const uint64_t input_chunks =
      xla::FloorOfRatio(block_input_size, kElementsPerMemoryAccess<T>);
  // This may be less than block_input_size only for the last block.
  const uint64_t chunked_input_size =
      xla::RoundDownTo(block_input_size, kElementsPerMemoryAccess<T>);

  FloatCheckResult result{};
  for (uint64_t i = tid; i < input_chunks; i += kBlockSize) {
    Chunk<T> values = chunked_input[i];
    for (const T value : values) {
      result.nan_count += IsNan(value);
      result.inf_count += IsInf(value);
      result.zero_count += IsZero(value);
    }
  }

  if (tid == 0 && chunked_input_size < block_input_size) {
    const size_t rest = block_input_size - chunked_input_size;
    for (uint64_t j = 0; j < rest; ++j) {
      const T value = block_input[input_chunks + j];
      result.nan_count += IsNan(value);
      result.inf_count += IsInf(value);
      result.zero_count += IsZero(value);
    }
  }

  return BlockReduceSum(tid, result);
}

__device__ FloatCheckResult ReduceResults(const FloatCheckResult* input,
                                          uint64_t input_size) {
  const unsigned int tid = ThreadIdx();
  const auto [block_input, block_input_size] = GetBlockInput(input, input_size);

  FloatCheckResult result{};
  for (uint64_t i = tid; i < input_size; i += kBlockSize) {
    const FloatCheckResult value = block_input[i];
    result.nan_count += value.nan_count;
    result.inf_count += value.inf_count;
    result.zero_count += value.zero_count;
  }

  // Now reduce a block worth of values into a single one.
  return BlockReduceSum(tid, result);
}

// Count the number of floats for NaNs, Infs and zeros in input buffer and store
// partially accumulated results in the tmp array.
template <typename T>
__global__ void FloatCheck(const T* input, uint64_t input_size,
                           xla::gpu::FloatCheckResult* tmp, uint64_t tmp_size) {
  assert(blockDim.x * blockDim.y * blockDim.z == kBlockSize);
  assert(BlockIdx() < tmp_size);
  if (BlockIdx() >= tmp_size) {
    return;
  }

  const FloatCheckResult result = CheckFloats(input, input_size, tmp_size);
  if (ThreadIdx() == 0) {
    tmp[BlockIdx()] = result;
  }
}

// Reduce the partially accumulated results from `FloatCheck` invocations and
// append the result to the buffer debug log.
__global__ void ReduceFloatCheckResults(
    xla::gpu::FloatCheckResult* tmp, uint64_t tmp_size,
    xla::gpu::BufferDebugLogEntryId entry_id,
    xla::gpu::BufferDebugLogHeader* log_header,
    xla::gpu::BufferDebugFloatCheckEntry* log_entries) {
  assert(blockDim.x * blockDim.y * blockDim.z == kBlockSize);
  assert(BlockIdx() == 0);
  if (BlockIdx() >= 1) {
    return;
  }

  assert(tmp_size > 0);
  FloatCheckResult total = ReduceResults(tmp, tmp_size);

  if (BlockIdx() == 0 && ThreadIdx() == 0) {
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> log_write_idx(
        log_header->write_idx);
#if __CUDA_ARCH__ >= 600
    const uint32_t write_idx = log_write_idx.fetch_add(1);
    if (write_idx < log_header->capacity) {
      log_entries[write_idx] = xla::gpu::BufferDebugFloatCheckEntry{
          entry_id, total.nan_count, total.inf_count, total.zero_count};
    }
#else
    // Our toolchains generate a fetch_add PTX instructions with system scope,
    // which is not supported on pre-Pascal architectures.
    (void)total;
    assert(false);
#endif
  }
}

se::KernelLoaderSpec GetFloatCheckF32KernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&FloatCheck<float>),
      "BufferDebugFloatCheckF32Kernel", arity);
}

se::KernelLoaderSpec GetFloatCheckBf16KernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&FloatCheck<__nv_bfloat16>),
      "BufferDebugFloatCheckBf16Kernel", arity);
}

se::KernelLoaderSpec GetReduceFloatCheckResultsKernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&ReduceFloatCheckResults),
      "BufferDebugReduceFloatCheckResultsKernel", arity);
}

}  // namespace

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckF32Kernel, se::gpu::BufferDebugFloatCheckF32Kernel,
    se::cuda::kCudaPlatformId, GetFloatCheckF32KernelSpec);

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugFloatCheckBf16Kernel, se::gpu::BufferDebugFloatCheckBf16Kernel,
    se::cuda::kCudaPlatformId, GetFloatCheckBf16KernelSpec);

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugReduceFloatCheckResultsKernel,
    se::gpu::BufferDebugAppendReducedFloatCheckResultsKernel,
    se::cuda::kCudaPlatformId, GetReduceFloatCheckResultsKernelSpec);
