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

#include "cub/device/device_scan.cuh"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda_fp16.h"  // IWYU pragma: keep
#include "xla/stream_executor/cuda/cub_scan_kernel_cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::cuda {

namespace {

template <typename T>
absl::Status CubScanImpl(void* d_temp_storage, size_t& temp_bytes,
                         const T* d_in, T* d_out, int64_t vector_length,
                         int64_t row_length, int64_t column_length,
                         CubScanKind kind, bool is_reverse, CUstream stream) {
  if (kind != CubScanKind::kSum) {
    return absl::InvalidArgumentError("Only SUM scan kind is supported.");
  }
  if (is_reverse) {
    return absl::InvalidArgumentError("Only forward scan is supported.");
  }
  if (vector_length > 1) {
    return absl::InvalidArgumentError("Only vector_length = 1 is supported.");
  }

  if (d_in == nullptr) {
    return ToStatus(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_bytes, d_in, d_out, row_length, stream));
  }

  for (int64_t col = 0; col < column_length; ++col) {
    TF_RETURN_IF_ERROR(ToStatus(cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_bytes, d_in + col * row_length,
        d_out + col * row_length, row_length, stream)));
  }

  return absl::OkStatus();
}

absl::Status CubScanDispatch(xla::PrimitiveType type, void* d_temp_storage,
                             size_t& temp_bytes, const void* d_in, void* d_out,
                             int64_t vector_length, int64_t row_length,
                             int64_t column_length, CubScanKind kind,
                             bool is_reverse, CUstream stream) {
  auto impl = [&](auto value) {
    using T = decltype(value);
    return CubScanImpl<T>(d_temp_storage, temp_bytes,
                          static_cast<const T*>(d_in), static_cast<T*>(d_out),
                          vector_length, row_length, column_length, kind,
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

}  // namespace

absl::Status CubScanLaunchKernel(xla::PrimitiveType type, void* d_temp_storage,
                                 size_t temp_bytes, const void* d_in,
                                 void* d_out, int64_t vector_length,
                                 int64_t row_length, int64_t column_length,
                                 CubScanKind kind, bool is_reverse,
                                 CUstream stream) {
  return CubScanDispatch(type, d_temp_storage, temp_bytes, d_in, d_out,
                         vector_length, row_length, column_length, kind,
                         is_reverse, stream);
}

absl::StatusOr<size_t> CubScanGetScratchSize(
    xla::PrimitiveType type, int64_t vector_length, int64_t row_length,
    int64_t column_length, CubScanKind kind, bool is_reverse) {
  size_t temp_bytes = 0;
  TF_RETURN_IF_ERROR(CubScanDispatch(type, nullptr, temp_bytes, nullptr,
                                     nullptr, vector_length, row_length,
                                     column_length, kind, is_reverse, nullptr));
  return temp_bytes;
}

}  // namespace stream_executor::cuda
