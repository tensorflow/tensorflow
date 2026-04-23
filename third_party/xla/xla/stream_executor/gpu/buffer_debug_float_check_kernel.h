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

#ifndef XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_FLOAT_CHECK_KERNEL_H_
#define XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_FLOAT_CHECK_KERNEL_H_

#include <cstdint>

#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/types.h"

namespace stream_executor::gpu {

// Counts the number of NaNs, Infs and zeros in a buffer of floats in parallel,
// and stores partially accumulated results in the FloatCheckResult array.
struct BufferDebugFloatCheckF32Kernel {
  using KernelType =
      TypedKernel<DeviceAddress<float>, uint64_t,
                  DeviceAddress<xla::gpu::FloatCheckResult>, uint64_t>;
};

// Counts the number of NaNs, Infs and zeros in a buffer of bfloat16s in
// parallel, and stores partially accumulated results in the FloatCheckResult
// array.
struct BufferDebugFloatCheckBf16Kernel {
  using KernelType =
      TypedKernel<DeviceAddress<Eigen::bfloat16>, uint64_t,
                  DeviceAddress<xla::gpu::FloatCheckResult>, uint64_t>;
};

// Counts the number of NaNs, Infs and zeros in a buffer of doubles in
// parallel, and stores partially accumulated results in the FloatCheckResult
// array.
struct BufferDebugFloatCheckF64Kernel {
  using KernelType =
      TypedKernel<DeviceAddress<double>, uint64_t,
                  DeviceAddress<xla::gpu::FloatCheckResult>, uint64_t>;
};

// Trait for a kernel that reduces the partially accumulated results from
// `BufferDebugFloatCheck{Bf16,F32,F64}Kernel` invocations and appends the
// result to the buffer debug log.
//
// This kernel MUST execute on a single thread block.
struct BufferDebugAppendReducedFloatCheckResultsKernel {
  using KernelType =
      TypedKernel<DeviceAddress<xla::gpu::FloatCheckResult>, uint64_t,
                  xla::gpu::BufferDebugLogEntryId,
                  DeviceAddress<xla::gpu::BufferDebugLogHeader>,
                  DeviceAddress<xla::gpu::BufferDebugFloatCheckEntry>>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_FLOAT_CHECK_KERNEL_H_
