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
#include <cstddef>
#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

template <typename T>
absl::Status LaunchTypedKernel(
    se::Stream* stream, se::StreamExecutor* executor,
    const se::ThreadDim& thread_dims, const se::BlockDim& block_dims,
    se::DeviceMemoryBase input_buffer,
    const std::array<void*,
                     stream_executor::gpu::kMaxNumRaggedAllToAllOutputPtrs>&
        output_ptrs,
    se::DeviceMemoryBase input_offsets_buffer,
    se::DeviceMemoryBase send_sizes_buffer,
    se::DeviceMemoryBase output_offsets_buffer, int64_t num_updates_per_output,
    int64_t num_row_elements) {
  TF_ASSIGN_OR_RETURN(
      auto kernel, se::gpu::GpuKernelRegistry::GetGlobalRegistry()
                       .LoadKernel<se::gpu::RaggedAllToAllKernel<T>>(executor));

  return kernel.Launch(thread_dims, block_dims, stream, input_buffer,
                       output_ptrs, input_offsets_buffer, send_sizes_buffer,
                       output_offsets_buffer, num_updates_per_output,
                       num_row_elements);
}

}  // namespace

}  // namespace xla::gpu
