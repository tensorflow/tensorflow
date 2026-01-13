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

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

namespace stream_executor::gpu {

// The implementation relies on cub:: templates which are CUDA specific. To
// keep this header platform-agnostic, provide forward declarations for
// template specializations that are going to be implemented in
// CUDA-specific code.
//
// And we need *some* definition to still compile on non-CUDA GPU platforms,
// even if the float check is not going to work.
template <typename T>
absl::Status CheckFloats(
    stream_executor::DeviceAddress<T> input,
    stream_executor::DeviceAddress<xla::gpu::FloatCheckResult> result,
    stream_executor::Stream* stream) {
  return absl::InternalError(
      "CheckFloats not implemented (missing explicit specialization?)");
}

#define CHECK_FLOATS_FORWARD_DECL(T)                                     \
  template <>                                                            \
  absl::Status CheckFloats<T>(                                           \
      stream_executor::DeviceAddress<T> input,                           \
      stream_executor::DeviceAddress<xla::gpu::FloatCheckResult> result, \
      stream_executor::Stream * stream);

CHECK_FLOATS_FORWARD_DECL(float)
CHECK_FLOATS_FORWARD_DECL(Eigen::bfloat16)

#undef CHECK_FLOATS_FORWARD_DECL

// Trait for a kernel that appends a range of `FloatCheckResult`s accompanied
// by their `BufferDebugLogEntryId`s to `BufferDebugLog`.
//
// The FloatCheckResult and BufferDebugLogEntryId arrays must have the same
// size, passed as the uint64_t parameter.
//
// FloatCheckResult can be calculated with CheckFloats functions above.
struct BufferDebugAppendFloatCheckResultsKernel {
  using KernelType =
      TypedKernel<DeviceAddress<xla::gpu::FloatCheckResult>,
                  DeviceAddress<xla::gpu::BufferDebugLogEntryId>, uint64_t,
                  DeviceAddress<xla::gpu::BufferDebugLogHeader>,
                  DeviceAddress<xla::gpu::BufferDebugFloatCheckEntry>>;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_FLOAT_CHECK_KERNEL_H_
