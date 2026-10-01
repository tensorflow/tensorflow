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

#ifndef XLA_STREAM_EXECUTOR_GPU_DTYPE_CORE_INFO_H_
#define XLA_STREAM_EXECUTOR_GPU_DTYPE_CORE_INFO_H_

#include "absl/types/span.h"

namespace stream_executor {
namespace gpu {

// Instead of using base primitive types we use a simple description that maps
// to several primitive types at once. This way we can keep the types in the
// backend tables more abstract.
struct DTypeDescr {
  bool is_float;
  int bitwidth;
};

inline constexpr DTypeDescr kI8 = DTypeDescr{/*is_float=*/false, 8};
inline constexpr DTypeDescr kI32 = DTypeDescr{/*is_float=*/false, 32};

inline constexpr DTypeDescr kF4 = DTypeDescr{/*is_float=*/true, 4};
inline constexpr DTypeDescr kF6 = DTypeDescr{/*is_float=*/true, 6};
inline constexpr DTypeDescr kF8 = DTypeDescr{/*is_float=*/true, 8};
inline constexpr DTypeDescr kF16 = DTypeDescr{/*is_float=*/true, 16};
inline constexpr DTypeDescr kF32 = DTypeDescr{/*is_float=*/true, 32};
inline constexpr DTypeDescr kF64 = DTypeDescr{/*is_float=*/true, 64};

// Throughput of one execution unit for every primitive type matching `dtype`.
// Per unit rates only. Core count and base clock come from the
// DeviceDescription instead, since they vary between SKUs of one architecture.
struct DTypeCoreInfo {
  DTypeDescr dtype;
  int units_per_core;
  int ops_per_clock = 1;    // Note: FMA is considered 1 op.
  float clock_scale = 1.0;  // Ratio of clock rate of this unit vs base device.
};

// What a backend table holds for one architecture. Either span may be empty,
// and both point into the table itself, which has static storage duration.
struct CoreInfo {
  absl::Span<const DTypeCoreInfo> vector_infos;
  absl::Span<const DTypeCoreInfo> matrix_infos;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_DTYPE_CORE_INFO_H_
