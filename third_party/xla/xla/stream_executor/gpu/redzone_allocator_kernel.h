/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_KERNEL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
using ComparisonKernel = TypedKernel<DeviceMemory<uint8_t>, uint8_t, uint64_t,
                                     DeviceMemory<uint64_t>>;

// Returns a GPU kernel that checks a memory location for redzone patterns.
// Parameters are (buffer_address, redzone_pattern, buffer_length,
// mismatch_count_ptr). For each byte in buffer `[buffer_address :
// buffer_address
// + buffer_length]` that is not equal to `redzone_pattern`,
// `*mismatch_count_ptr` gets incremented by 1.
absl::StatusOr<const ComparisonKernel*> GetComparisonKernel(
    StreamExecutor* executor, GpuAsmOpts gpu_asm_opts);

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_REDZONE_ALLOCATOR_KERNEL_H_
