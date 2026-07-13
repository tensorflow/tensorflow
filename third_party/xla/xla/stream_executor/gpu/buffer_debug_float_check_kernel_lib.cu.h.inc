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
#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_FLOAT_CHECK_KERNEL_LIB_CU_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_FLOAT_CHECK_KERNEL_LIB_CU_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/util.h"

namespace stream_executor::gpu {

using xla::gpu::FloatCheckResult;

// https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/:
// > CUDA architecture limits the numbers of threads per block (1024 threads
// > per block limit).
static constexpr uint64_t kBlockSize = 1024;
// `kWarpSize` must be defined by the platform-specific translation unit (in
// namespace stream_executor::gpu) before including this header. warpSize is not
// a compile time constant on all OSS CI builds, but we need it to be one for
// static array initialization, and it differs between CUDA and ROCm.
__device__ static constexpr uint64_t kMaxWarpsPerBlock = kBlockSize / kWarpSize;
template <typename T>
static constexpr uint64_t kElementsPerMemoryAccess =
    std::max<uint64_t>(16 / sizeof(T), 1);
template <typename T>
using Chunk = std::array<T, kElementsPerMemoryAccess<T>>;

template <typename T>
__device__ static constexpr T kInfinity = std::numeric_limits<T>::infinity();

template <typename T>
__device__ inline bool IsNan(T v) {
  return isnan(v);
}
template <typename T>
__device__ inline bool IsInf(T v) {
  return isinf(v);
}
template <typename T>
__device__ inline bool IsZero(T v) {
  return v == T(0);
}

__device__ inline uint32_t AtomicIncSystem(uint32_t* write_idx);

__device__ inline unsigned int ThreadIdx() {
  return threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x +
         threadIdx.x;
}

__device__ inline unsigned int BlockIdx() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
         blockIdx.x;
}

template <typename T>
__device__ T FiniteOr(T val, T fallback) {
  if (IsNan(val) || IsInf(val)) {
    return fallback;
  }
  return val;
}

template <typename T>
__device__ inline T WarpShuffleDown(T value, unsigned int offset);

// Reduce `subgroup_size` lanes worth of values into a single one and have the
// 0th lane of the subgroup return it.
__device__ inline uint32_t WarpSubgroupReduceSum(uint32_t value,
                                                 uint64_t subgroup_size) {
  for (unsigned int offset = 1; offset < subgroup_size; offset <<= 1) {
    value += WarpShuffleDown(value, offset);
  }
  return value;
}

template <typename T>
__device__ T WarpSubgroupReduceMin(T value, uint64_t subgroup_size) {
  value = FiniteOr(value, kInfinity<T>);

  for (unsigned int offset = 1; offset < subgroup_size; offset <<= 1) {
    value = std::min(value, WarpShuffleDown(value, offset));
  }
  return value;
}

template <typename T>
__device__ T WarpSubgroupReduceMax(T value, uint64_t subgroup_size) {
  value = FiniteOr(value, -kInfinity<T>);

  for (unsigned int offset = 1; offset < subgroup_size; offset <<= 1) {
    value = std::max(value, WarpShuffleDown(value, offset));
  }
  return value;
}

// Sum up a block worth of FloatCheckResults into a single one and have the 0th
// thread in the block return it.
__device__ inline FloatCheckResult BlockReduceResults(uint32_t tid,
                                                      FloatCheckResult value) {
  assert(kWarpSize == warpSize);
  static_assert(kBlockSize == kWarpSize * kMaxWarpsPerBlock);

  const size_t warp_idx = tid / kWarpSize;
  const size_t lane_idx = tid % kWarpSize;

  value.nan_count = WarpSubgroupReduceSum(value.nan_count, kWarpSize);
  value.inf_count = WarpSubgroupReduceSum(value.inf_count, kWarpSize);
  value.zero_count = WarpSubgroupReduceSum(value.zero_count, kWarpSize);
  value.min_value = WarpSubgroupReduceMin(value.min_value, kWarpSize);
  value.max_value = WarpSubgroupReduceMax(value.max_value, kWarpSize);

  __shared__ uint32_t scratch_nan[kMaxWarpsPerBlock];
  __shared__ uint32_t scratch_inf[kMaxWarpsPerBlock];
  __shared__ uint32_t scratch_zero[kMaxWarpsPerBlock];
  __shared__ double scratch_min[kMaxWarpsPerBlock];
  __shared__ double scratch_max[kMaxWarpsPerBlock];
  if (lane_idx == 0) {
    scratch_nan[warp_idx] = value.nan_count;
    scratch_inf[warp_idx] = value.inf_count;
    scratch_zero[warp_idx] = value.zero_count;
    scratch_min[warp_idx] = value.min_value;
    scratch_max[warp_idx] = value.max_value;
  }

  __syncthreads();
  // The first kMaxWarpsPerBlock lanes of first warp reduces the results from
  // all warps.
  if (warp_idx == 0) {
    const size_t scratch_idx = lane_idx % kMaxWarpsPerBlock;
    value.nan_count = scratch_nan[scratch_idx];
    value.inf_count = scratch_inf[scratch_idx];
    value.zero_count = scratch_zero[scratch_idx];
    value.min_value = scratch_min[scratch_idx];
    value.max_value = scratch_max[scratch_idx];

    value.nan_count = WarpSubgroupReduceSum(value.nan_count, kMaxWarpsPerBlock);
    value.inf_count = WarpSubgroupReduceSum(value.inf_count, kMaxWarpsPerBlock);
    value.zero_count =
        WarpSubgroupReduceSum(value.zero_count, kMaxWarpsPerBlock);
    value.min_value = WarpSubgroupReduceMin(value.min_value, kMaxWarpsPerBlock);
    value.max_value = WarpSubgroupReduceMax(value.max_value, kMaxWarpsPerBlock);
  } else {
    value.nan_count = 0;
    value.inf_count = 0;
    value.zero_count = 0;
    value.min_value = kInfinity<double>;
    value.max_value = -kInfinity<double>;
  }

  return value;
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
  result.min_value = kInfinity<double>;
  result.max_value = -kInfinity<double>;

  for (uint64_t i = tid; i < input_chunks; i += kBlockSize) {
    Chunk<T> values = chunked_input[i];
    for (const T value : values) {
      result.nan_count += IsNan(value);
      result.inf_count += IsInf(value);
      result.zero_count += IsZero(value);
      result.min_value = std::min(
          result.min_value, static_cast<double>(FiniteOr(value, kInfinity<T>)));
      result.max_value =
          std::max(result.max_value,
                   static_cast<double>(FiniteOr(value, -kInfinity<T>)));
    }
  }

  if (tid == 0 && chunked_input_size < block_input_size) {
    const size_t rest = block_input_size - chunked_input_size;
    for (uint64_t j = 0; j < rest; ++j) {
      const T value = block_input[chunked_input_size + j];
      result.nan_count += IsNan(value);
      result.inf_count += IsInf(value);
      result.zero_count += IsZero(value);
      result.min_value = std::min(
          result.min_value, static_cast<double>(FiniteOr(value, kInfinity<T>)));
      result.max_value =
          std::max(result.max_value,
                   static_cast<double>(FiniteOr(value, -kInfinity<T>)));
    }
  }

  return BlockReduceResults(tid, result);
}

__device__ inline FloatCheckResult ReduceResults(const FloatCheckResult* input,
                                                 uint64_t input_size) {
  const unsigned int tid = ThreadIdx();
  const auto [block_input, block_input_size] = GetBlockInput(input, input_size);

  FloatCheckResult result{};
  result.min_value = kInfinity<double>;
  result.max_value = -kInfinity<double>;

  for (uint64_t i = tid; i < input_size; i += kBlockSize) {
    const FloatCheckResult value = block_input[i];
    result.nan_count += value.nan_count;
    result.inf_count += value.inf_count;
    result.zero_count += value.zero_count;
    result.min_value = std::min(result.min_value, value.min_value);
    result.max_value = std::max(result.max_value, value.max_value);
  }

  // Now reduce a block worth of values into a single one.
  return BlockReduceResults(tid, result);
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

  FloatCheckResult result = CheckFloats(input, input_size, tmp_size);
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
    const uint32_t write_idx = AtomicIncSystem(&log_header->write_idx);
    if (write_idx < log_header->capacity) {
      log_entries[write_idx] =
          xla::gpu::BufferDebugFloatCheckEntry{entry_id, total};
    }
  }
}

}  // namespace stream_executor::gpu

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_FLOAT_CHECK_KERNEL_LIB_CU_H_
