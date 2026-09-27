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

#ifndef XLA_STREAM_EXECUTOR_ROCM_CUB_SCAN_KERNEL_ROCM_H_
#define XLA_STREAM_EXECUTOR_ROCM_CUB_SCAN_KERNEL_ROCM_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::rocm {

enum class CubScanKind { kInvalid, kSum };

absl::Status CubScanLaunchKernel(xla::PrimitiveType type, void* d_temp_storage,
                                 size_t temp_bytes, const void* d_in,
                                 void* d_out, int64_t vector_length,
                                 int64_t row_length, int64_t column_length,
                                 CubScanKind kind, bool is_reverse,
                                 hipStream_t stream);

absl::StatusOr<size_t> CubScanGetScratchSize(xla::PrimitiveType type,
                                             int64_t vector_length,
                                             int64_t row_length,
                                             int64_t column_length,
                                             CubScanKind kind, bool is_reverse);

}  // namespace stream_executor::rocm

#endif  // XLA_STREAM_EXECUTOR_ROCM_CUB_SCAN_KERNEL_ROCM_H_
