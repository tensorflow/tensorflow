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
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/rocprim/device/device_scan.hpp"
#include "rocm/include/rocprim/device/device_segmented_scan.hpp"
#include "rocm/include/rocprim/functional.hpp"
#include "rocm/include/rocprim/iterator/counting_iterator.hpp"
#include "rocm/include/rocprim/iterator/transform_iterator.hpp"
#include "xla/stream_executor/rocm/cub_scan_kernel_rocm.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::rocm {

namespace {

struct ScaleByRowLength {
  int64_t row_length;
  __host__ __device__ int64_t operator()(int64_t i) const {
    return i * row_length;
  }
};

template <typename T>
absl::Status CubScanDispatch(void* d_temp_storage, size_t* temp_bytes,
                             const T* d_in, T* d_out, int64_t vector_length,
                             int64_t row_length, int64_t column_length,
                             CubScanKind kind, bool is_reverse,
                             hipStream_t stream) {
  if (kind != CubScanKind::kSum) {
    return absl::InvalidArgumentError("Only SUM scan kind is supported.");
  }
  if (is_reverse) {
    return absl::InvalidArgumentError("Only forward scan is supported.");
  }
  if (vector_length > 1) {
    return absl::InvalidArgumentError("Only vector_length = 1 is supported.");
  }

  if (column_length == 1) {
    // 1D: use rocprim device-wide scan for full GPU parallelism.
    return stream_executor::gpu::ToStatus(
        rocprim::inclusive_scan(d_temp_storage, *temp_bytes, d_in, d_out,
                                row_length, rocprim::plus<T>(), stream));
  }

  ScaleByRowLength scale{row_length};
  auto begin_offsets = rocprim::make_transform_iterator(
      rocprim::make_counting_iterator<int64_t>(0), scale);
  auto end_offsets = rocprim::make_transform_iterator(
      rocprim::make_counting_iterator<int64_t>(1), scale);
  return stream_executor::gpu::ToStatus(rocprim::segmented_inclusive_scan(
      d_temp_storage, *temp_bytes, d_in, d_out,
      static_cast<unsigned int>(column_length), begin_offsets, end_offsets,
      rocprim::plus<T>(), stream));
}

absl::Status CubScanDispatch(xla::PrimitiveType type, void* d_temp_storage,
                             size_t* temp_bytes, const void* d_in, void* d_out,
                             int64_t vector_length, int64_t row_length,
                             int64_t column_length, CubScanKind kind,
                             bool is_reverse, hipStream_t stream) {
  auto impl = [&](auto value) {
    using T = decltype(value);
    return CubScanDispatch<T>(d_temp_storage, temp_bytes,
                              static_cast<const T*>(d_in),
                              static_cast<T*>(d_out), vector_length, row_length,
                              column_length, kind, is_reverse, stream);
  };
  switch (type) {
    case xla::PrimitiveType::BF16:
      return impl(hip_bfloat16{});
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
                                 hipStream_t stream) {
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

}  // namespace stream_executor::rocm
