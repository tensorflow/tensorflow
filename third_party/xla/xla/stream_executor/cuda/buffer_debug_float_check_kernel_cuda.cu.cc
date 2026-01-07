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

#include "cub/cub.cuh"
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "Eigen/Core"
#include "third_party/gpus/cuda/include/cuda/atomic"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/cuda/cuda_stream.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/tsl/platform/status_macros.h"

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

// Append the results from `FloatCheck` invocation to the buffer debug log.
__global__ void AppendFloatCheckResults(
    xla::gpu::FloatCheckResult* results,
    xla::gpu::BufferDebugLogEntryId* entry_ids, uint64_t num_entries,
    xla::gpu::BufferDebugLogHeader* log_header,
    xla::gpu::BufferDebugFloatCheckEntry* log_entries) {
  assert(BlockIdx() == 0);
  assert(ThreadIdx() == 0);

  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> log_write_idx(
      log_header->write_idx);
#if __CUDA_ARCH__ >= 600
  const uint32_t write_idx = log_write_idx.fetch_add(num_entries);
  for (int i = 0; i < num_entries; ++i) {
    log_entries[write_idx + i] = xla::gpu::BufferDebugFloatCheckEntry{
        entry_ids[i], results[i].nan_count, results[i].inf_count,
        results[i].zero_count};
  }
#else
  // Our toolchains generate a fetch_add PTX instructions with system scope,
  // which is not supported on pre-Pascal architectures.
  (void)results;
  (void)entry_ids;
  (void)num_entries;
  (void)log_entries;
  assert(false);
#endif
}

se::KernelLoaderSpec GetAppendFloatCheckResultsKernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateInProcessSymbolSpec(
      absl::bit_cast<void*>(&AppendFloatCheckResults),
      "AppendFloatCheckResultsKernel", arity);
}

template <typename T>
struct CheckFloat {
  __device__ FloatCheckResult operator()(const T input) {
    FloatCheckResult result{};
    result.nan_count += IsNan(input);
    result.inf_count += IsInf(input);
    result.zero_count += IsZero(input);
    return result;
  }
};

template <typename T>
CheckFloat<T> check_float{};

struct ReduceFloatCheckResults {
  __device__ FloatCheckResult operator()(FloatCheckResult a,
                                         FloatCheckResult b) {
    a.nan_count += b.nan_count;
    a.inf_count += b.inf_count;
    a.zero_count += b.zero_count;
    return a;
  }
};

ReduceFloatCheckResults reduce_float_check_results{};

template <typename T>
absl::Status CheckFloatsImpl(
    se::DeviceAddress<T> input,
    se::DeviceAddress<xla::gpu::FloatCheckResult> result, se::Stream* stream) {
  se::gpu::CudaStream* cuda_stream = dynamic_cast<se::gpu::CudaStream*>(stream);
  if (cuda_stream == nullptr) {
    return absl::InvalidArgumentError("Stream is not a CUDA stream");
  }

  size_t tmp_size;
  RETURN_IF_ERROR(se::cuda::ToStatus(
      cub::DeviceReduce::TransformReduce(
          nullptr, tmp_size, input.base(), result.base(), input.ElementCount(),
          reduce_float_check_results, check_float<T>, FloatCheckResult{},
          cuda_stream->stream_handle()),
      "required temp storage for TransformReduce failed"));

  se::DeviceAddressBase tmp = stream->parent()->Allocate(tmp_size);
  auto tmp_cleanup =
      absl::MakeCleanup([&]() { stream->parent()->Deallocate(&tmp); });

  return se::cuda::ToStatus(
      cub::DeviceReduce::TransformReduce(
          tmp.opaque(), tmp_size, input.base(), result.base(),
          input.ElementCount(), reduce_float_check_results, check_float<T>,
          FloatCheckResult{}, cuda_stream->stream_handle()),
      "TransformReduce failed");
}

}  // namespace

namespace stream_executor::gpu {

#define CHECK_FLOATS_CUB_IMPL(T)                            \
  template <>                                               \
  absl::Status CheckFloats<T>(                              \
      se::DeviceAddress<T> input,                           \
      se::DeviceAddress<xla::gpu::FloatCheckResult> result, \
      se::Stream * stream) {                                \
    return CheckFloatsImpl<T>(input, result, stream);       \
  }

CHECK_FLOATS_CUB_IMPL(float)
CHECK_FLOATS_CUB_IMPL(Eigen::bfloat16)

#undef CHECK_FLOATS_CUB_IMPL

}  // namespace stream_executor::gpu

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    BufferDebugAppendFloatCheckResultsKernel,
    se::gpu::BufferDebugAppendFloatCheckResultsKernel,
    se::cuda::kCudaPlatformId, GetAppendFloatCheckResultsKernelSpec);
