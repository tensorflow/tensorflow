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

template <typename T, typename ScanOpT, typename ChainedPolicyT>
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

  using MaxPolicyT = typename cub::detail::scan::policy_hub<
      /*InputValueT=*/T, /*OutputValueT=*/T,
      /*AccumT=*/T, /*OffsetT=*/int64_t,
      /*ScanOpT=*/ScanOpT>::MaxPolicy;

  // CUB seems to require that the kernel matches the ptx version exactly.
  // E.g. when compiling for sm_70 and running on sm_75, MaxPolicyT::Invoke()
  // returns cudaErrorInvalidDeviceFunction. We therefore query the kernel's
  // max threads per block, which should match ScanPolicyT::BLOCK_THREADS
  // because we use that as __launch_bounds__.
  auto* kernel = BlockScanKernel<T, ScanOpT, MaxPolicyT>;
  cudaFunction_t function;
  TF_RETURN_IF_ERROR(ToStatus(
      cudaGetFuncBySymbol(&function, reinterpret_cast<const void*>(kernel))));
  int block_size;
  TF_RETURN_IF_ERROR(ToStatus(cuFuncGetAttribute(
      &block_size, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function)));
  cudaLaunchConfig_t config = {
      .gridDim = column_length, .blockDim = block_size, .stream = stream};
  TF_RETURN_IF_ERROR(
      ToStatus(cudaLaunchKernelEx(&config, kernel, d_in, d_out, row_length)));

  return absl::OkStatus();
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
