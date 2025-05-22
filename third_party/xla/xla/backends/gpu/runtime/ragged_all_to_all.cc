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

bool IsRaggedAllToAllKernelSupported(int64_t num_outputs,
                                     PrimitiveType element_type) {
  int bit_width = primitive_util::BitWidth(element_type);

  return num_outputs <= stream_executor::gpu::kMaxNumRaggedAllToAllOutputPtrs &&
         (bit_width == 8 || bit_width == 16 || bit_width == 32 ||
          bit_width == 64);
}

absl::Status RunRaggedAllToAllKernel(
    se::Stream* stream, PrimitiveType element_type,
    se::DeviceMemoryBase input_buffer,
    absl::Span<const se::DeviceMemoryBase> output_buffers,
    se::DeviceMemoryBase input_offsets_buffer,
    se::DeviceMemoryBase send_sizes_buffer,
    se::DeviceMemoryBase output_offsets_buffer, int64_t num_outputs,
    int64_t num_updates_per_output, int64_t num_input_rows,
    int64_t num_row_elements) {
  if (output_buffers.size() >
      stream_executor::gpu::kMaxNumRaggedAllToAllOutputPtrs) {
    return absl::InvalidArgumentError(
        "Number of output pointers exceeds the maximum supported number of "
        "output pointers.");
  }

  se::StreamExecutor* executor = stream->parent();
  static constexpr size_t kThreads = 128;
  static constexpr size_t kMaxBlocksPerUpdate = 1024;

  // blockIdx.x is the index of the update.
  int64_t num_blocks_x = num_updates_per_output * num_outputs;

  // blockIdx.y and threadIdx.x are used to iterate over the elements of the
  // update. Since the size of each update is not known at compile time, the
  // kernel assumes the worst case of `num_input_rows * num_row_elements`
  // elements per update and uses a loop up to `send_size * num_row_elements` to
  // terminate early.
  size_t num_blocks_y =
      std::min(CeilOfRatio<size_t>(num_input_rows * num_row_elements, kThreads),
               kMaxBlocksPerUpdate);

  se::ThreadDim thread_dims(kThreads, 1, 1);
  se::BlockDim block_dims(num_blocks_x, num_blocks_y, 1);

  std::array<void*, stream_executor::gpu::kMaxNumRaggedAllToAllOutputPtrs>
      output_ptrs;
  for (int64_t i = 0; i < output_buffers.size(); ++i) {
    output_ptrs[i] = output_buffers[i].opaque();
  }

  auto launch_kernel = [&](auto type) -> absl::Status {
    using T = decltype(type);
    return LaunchTypedKernel<T>(stream, executor, thread_dims, block_dims,
                                input_buffer, output_ptrs, input_offsets_buffer,
                                send_sizes_buffer, output_offsets_buffer,
                                num_updates_per_output, num_row_elements);
  };

  switch (xla::primitive_util::BitWidth(element_type)) {
    case 8:
      return launch_kernel(uint8_t{});
    case 16:
      return launch_kernel(uint16_t{});
    case 32:
      return launch_kernel(uint32_t{});
    case 64:
      return launch_kernel(uint64_t{});
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported element type: ",
          primitive_util::LowercasePrimitiveTypeName(element_type),
          " (bit width ", xla::primitive_util::BitWidth(element_type),
          ") for RaggedAllToAll kernel."));
  }
}
}  // namespace xla::gpu
