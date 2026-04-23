/* Copyright 2026 The OpenXLA Authors.

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
#include <cstddef>
#include <cstdint>  // IWYU pragma: keep

#include "cub/agent/agent_scan.cuh"
#include "cub/block/block_load.cuh"
#include "cub/block/block_reduce.cuh"
#include "cub/block/block_scan.cuh"
#include "cub/block/block_store.cuh"
#include "cub/device/device_scan.cuh"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda_fp16.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda_runtime.h"
#include "xla/stream_executor/cuda/cub_scan_kernel_cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::cuda {

namespace {

template <typename T, typename ScanOpT>
using MaxPolicyT = typename cub::detail::scan::policy_hub<
    /*InputValueT=*/T, /*OutputValueT=*/T,
    /*AccumT=*/T, /*OffsetT=*/int64_t,
    /*ScanOpT=*/ScanOpT>::MaxPolicy;

template <typename T, typename ScanOpT,
          typename ChainedPolicyT = MaxPolicyT<T, ScanOpT>>
__launch_bounds__(ChainedPolicyT::ActivePolicy::ScanPolicyT::BLOCK_THREADS)
    __global__ void BlockScanKernel(const T* d_in, T* d_out, int64_t n) {
  using ScanPolicyT = typename ChainedPolicyT::ActivePolicy::ScanPolicyT;

  constexpr int kBlockSize = ScanPolicyT::BLOCK_THREADS;
  constexpr int kVectorSize = ScanPolicyT::ITEMS_PER_THREAD;
  constexpr int kTileSize = kBlockSize * kVectorSize;

  using BlockLoad =
      cub::BlockLoad<T, kBlockSize, kVectorSize, ScanPolicyT::LOAD_ALGORITHM>;
  using BlockStore =
      cub::BlockStore<T, kBlockSize, kVectorSize, ScanPolicyT::STORE_ALGORITHM>;
  using BlockScan = cub::BlockScan<T, kBlockSize, ScanPolicyT::SCAN_ALGORITHM>;

  __shared__ union {
    typename BlockLoad::TempStorage load;
    typename BlockStore::TempStorage store;
  } storage;

  BlockLoad block_load(storage.load);
  BlockStore block_store(storage.store);
  BlockScan block_scan;

  d_in += blockIdx.x * n;
  d_out += blockIdx.x * n;

  auto accumulator = [sum = T{}, op = ScanOpT()](T value) mutable {
    T result = sum;
    sum = op(sum, value);
    return result;
  };
  for (int64_t tile_offset = 0; tile_offset < n; tile_offset += kTileSize) {
    int tile_size = std::min<int64_t>(n - tile_offset, kTileSize);

    T thread_data[kVectorSize];
    block_load.Load(d_in + tile_offset, thread_data, tile_size, T{});
    __syncthreads();

    block_scan.InclusiveSum(thread_data, thread_data, accumulator);

    block_store.Store(d_out + tile_offset, thread_data, tile_size);
    __syncthreads();
  }
}

// Kernel to perform inclusive sum of m rows of size n using one thread per row.
// Loads data into shared memory, performs in-place scan, and stores result to
// global memory.
template <typename T, typename ScanOpT, int kThreadsPerBlock>
__launch_bounds__(kThreadsPerBlock) __global__
    void ThreadScanKernel(const T* d_in, T* d_out, int64_t m, int64_t n) {
  extern __shared__ char smem_buf[];
  T* smem = reinterpret_cast<T*>(smem_buf);

  int tile_size = blockDim.x * n;
  int64_t block_offset = static_cast<int64_t>(tile_size) * blockIdx.x;
  tile_size = ullmin(tile_size, m * n - block_offset);

  d_in += block_offset;
  d_out += block_offset;

  for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
    smem[i] = d_in[i];
  }
  __syncthreads();

  if (threadIdx.x + blockIdx.x * blockDim.x < m) {
    T* it = smem + threadIdx.x * n;
    T sum = *it;
    ScanOpT scan_op;
    for (int i = 1; i < n; ++i) {
      ++it;
      sum = scan_op(sum, *it);
      *it = sum;
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
    d_out[i] = smem[i];
  }
}

template <typename T, typename ScanOpT>
absl::Status CubThreadScanDispatch(const T* d_in, T* d_out, int64_t row_length,
                                   int64_t column_length, CUstream stream) {
  constexpr int block_size = 256;
  auto* kernel = ThreadScanKernel<T, ScanOpT, block_size>;
  size_t shared_mem_bytes = block_size * row_length * sizeof(T);
  TF_RETURN_IF_ERROR(ToStatus(cudaFuncSetAttribute(
      reinterpret_cast<const void*>(kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes)));
  int grid_size = (column_length + block_size - 1) / block_size;
  cudaLaunchConfig_t config = {.gridDim = grid_size,
                               .blockDim = block_size,
                               .dynamicSmemBytes = shared_mem_bytes,
                               .stream = stream};
  return ToStatus(cudaLaunchKernelEx(&config, kernel, d_in, d_out,
                                     column_length, row_length));
}

template <typename T, typename ScanOpT>
absl::Status CubScanDispatch(void* d_temp_storage, size_t* temp_bytes,
                             const T* d_in, T* d_out, int64_t vector_length,
                             int64_t row_length, int64_t column_length,
                             bool is_reverse, CUstream stream) {
  if (is_reverse) {
    return absl::InvalidArgumentError("Only forward scan is supported.");
  }
  if (vector_length > 1) {
    return absl::InvalidArgumentError("Only vector_length = 1 is supported.");
  }

  if (d_in == nullptr) {
    *temp_bytes = 0;
    return absl::OkStatus();
  }

  // For many small scans, one thread per row is faster.
  // The threshold is based on sparse empirical data for H100.
  // If launching the thread scan fails, fall back to block scan.
  if (column_length >= 64 * row_length &&
      CubThreadScanDispatch<T, ScanOpT>(d_in, d_out, row_length, column_length,
                                        stream)
          .ok()) {
    return absl::OkStatus();
  }

  auto* kernel = BlockScanKernel<T, ScanOpT>;

  // CUB seems to require that the kernel matches the ptx version exactly.
  // E.g. when compiling for sm_70 and running on sm_75, MaxPolicyT::Invoke()
  // returns cudaErrorInvalidDeviceFunction. We therefore query the kernel's
  // max threads per block, which should match ScanPolicyT::BLOCK_THREADS
  // because we use that as __launch_bounds__.
  cudaFunction_t function;
  TF_RETURN_IF_ERROR(ToStatus(
      cudaGetFuncBySymbol(&function, reinterpret_cast<const void*>(kernel))));
  int block_size;
  TF_RETURN_IF_ERROR(ToStatus(cuFuncGetAttribute(
      &block_size, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function)));

  cudaLaunchConfig_t config = {
      .gridDim = column_length, .blockDim = block_size, .stream = stream};
  return ToStatus(cudaLaunchKernelEx(&config, kernel, d_in, d_out, row_length));
}

template <typename ScanOpT>
absl::Status CubScanDispatch(xla::PrimitiveType type, void* d_temp_storage,
                             size_t* temp_bytes, const void* d_in, void* d_out,
                             int64_t vector_length, int64_t row_length,
                             int64_t column_length, bool is_reverse,
                             CUstream stream) {
  auto impl = [&](auto value) {
    using T = decltype(value);
    return CubScanDispatch<T, ScanOpT>(
        d_temp_storage, temp_bytes, static_cast<const T*>(d_in),
        static_cast<T*>(d_out), vector_length, row_length, column_length,
        is_reverse, stream);
  };
  switch (type) {
    case xla::PrimitiveType::BF16:
      return impl(__nv_bfloat16{});
    case xla::PrimitiveType::F16:
      return impl(__half{});
    case xla::PrimitiveType::F32:
      return impl(float{});
    case xla::PrimitiveType::F64:
      return impl(double{});
    case xla::PrimitiveType::S8:
      return impl(int8_t{});
    case xla::PrimitiveType::S16:
      return impl(int16_t());
    case xla::PrimitiveType::S32:
      return impl(int32_t{});
    case xla::PrimitiveType::S64:
      return impl(int64_t{});
    case xla::PrimitiveType::U8:
      return impl(uint8_t{});
    case xla::PrimitiveType::U16:
      return impl(uint16_t());
    case xla::PrimitiveType::U32:
      return impl(uint32_t());
    case xla::PrimitiveType::U64:
      return impl(uint64_t{});
    default:
      return absl::InvalidArgumentError(
          "Unsupported element type for CUB scan");
  }
}

absl::Status CubScanDispatch(xla::PrimitiveType type, void* d_temp_storage,
                             size_t* temp_bytes, const void* d_in, void* d_out,
                             int64_t vector_length, int64_t row_length,
                             int64_t column_length, CubScanKind kind,
                             bool is_reverse, CUstream stream) {
  auto impl = [&](auto scan_op) {
    return CubScanDispatch<decltype(scan_op)>(
        type, d_temp_storage, temp_bytes, d_in, d_out, vector_length,
        row_length, column_length, is_reverse, stream);
  };
  switch (kind) {
    case CubScanKind::kSum:
      return impl(::cuda::std::plus<void>());
    default:
      return absl::InvalidArgumentError("Unsupported scan operation.");
  }
}

}  // namespace

absl::Status CubScanLaunchKernel(xla::PrimitiveType type, void* d_temp_storage,
                                 size_t temp_bytes, const void* d_in,
                                 void* d_out, int64_t vector_length,
                                 int64_t row_length, int64_t column_length,
                                 CubScanKind kind, bool is_reverse,
                                 CUstream stream) {
  return CubScanDispatch(type, d_temp_storage, &temp_bytes, d_in, d_out,
                         vector_length, row_length, column_length, kind,
                         is_reverse, stream);
}

absl::StatusOr<size_t> CubScanGetScratchSize(
    xla::PrimitiveType type, int64_t vector_length, int64_t row_length,
    int64_t column_length, CubScanKind kind, bool is_reverse) {
  size_t temp_bytes = 0;
  TF_RETURN_IF_ERROR(CubScanDispatch(type, nullptr, &temp_bytes, nullptr,
                                     nullptr, vector_length, row_length,
                                     column_length, kind, is_reverse, nullptr));
  return temp_bytes;
}

}  // namespace stream_executor::cuda
