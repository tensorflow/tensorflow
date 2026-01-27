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

#include "xla/backends/gpu/runtime/ragged_all_to_all.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
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

template <int64_t kVectorSize>
absl::Status LaunchTypedKernel(
    se::Stream* stream, se::StreamExecutor* executor,
    const se::ThreadDim& thread_dims, const se::BlockDim& block_dims,
    se::DeviceAddressBase input_buffer,
    const std::array<void*,
                     stream_executor::gpu::kMaxNumRaggedAllToAllOutputPtrs>&
        output_ptrs,
    se::DeviceAddressBase input_offsets_buffer,
    se::DeviceAddressBase send_sizes_buffer,
    se::DeviceAddressBase output_offsets_buffer, int64_t num_updates_per_output,
    int64_t num_row_elements) {
  TF_ASSIGN_OR_RETURN(
      auto kernel,
      se::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<se::gpu::RaggedAllToAllKernel<kVectorSize>>(executor));

  return kernel.Launch(thread_dims, block_dims, stream, input_buffer,
                       output_ptrs, input_offsets_buffer, send_sizes_buffer,
                       output_offsets_buffer, num_updates_per_output,
                       num_row_elements);
}

}  // namespace

bool IsRaggedAllToAllKernelSupported(int64_t num_outputs,
                                     PrimitiveType element_type) {
  return num_outputs <= stream_executor::gpu::kMaxNumRaggedAllToAllOutputPtrs &&
         // Currently, the kernel doesn't support data types that are smaller
         // than 1 byte.
         primitive_util::BitWidth(element_type) % 8 == 0;
}

absl::Status RunRaggedAllToAllKernel(
    se::Stream* stream, PrimitiveType element_type,
    se::DeviceAddressBase input_buffer,
    absl::Span<const se::DeviceAddressBase> output_buffers,
    se::DeviceAddressBase input_offsets_buffer,
    se::DeviceAddressBase send_sizes_buffer,
    se::DeviceAddressBase output_offsets_buffer, int64_t num_outputs,
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

  int64_t num_vectorized_row_elements = num_row_elements;
  int64_t vector_size_bytes = xla::primitive_util::BitWidth(element_type) / 8;

  while (num_vectorized_row_elements % 2 == 0 && vector_size_bytes < 8) {
    num_vectorized_row_elements /= 2;
    vector_size_bytes *= 2;
  }

  int64_t num_updates_per_block = 1;
  int64_t num_block_clusters = num_updates_per_output;

  // Decide how many updates should each block process. In the kernel, N blocks
  // process N updates. This is done to reduce imbalance in data transfer per
  // block if updates happen to be unevenly distributed. The numbers were
  // chosen empirically in Sep 2025 and can change in the future.
  const int64_t max_num_updates_per_block =
      std::min<int64_t>(CeilOfRatio<int64_t>(num_input_rows, 16), 64);

  while (num_updates_per_block < max_num_updates_per_block &&
         num_block_clusters % 2 == 0) {
    num_block_clusters /= 2;
    num_updates_per_block *= 2;
  }

  se::ThreadDim thread_dims(kThreads, 1, 1);
  se::BlockDim block_dims(num_outputs, num_block_clusters,
                          num_updates_per_block);

  std::array<void*, stream_executor::gpu::kMaxNumRaggedAllToAllOutputPtrs>
      output_ptrs;
  for (int64_t i = 0; i < output_buffers.size(); ++i) {
    output_ptrs[i] = output_buffers[i].opaque();
  }

  auto launch_kernel = [&](auto type) -> absl::Status {
    using T = decltype(type);
    return LaunchTypedKernel<T::value>(
        stream, executor, thread_dims, block_dims, input_buffer, output_ptrs,
        input_offsets_buffer, send_sizes_buffer, output_offsets_buffer,
        num_updates_per_output, num_vectorized_row_elements);
  };

  switch (vector_size_bytes) {
    case 1:
      return launch_kernel(std::integral_constant<int64_t, 1>{});
    case 2:
      return launch_kernel(std::integral_constant<int64_t, 2>{});
    case 4:
      return launch_kernel(std::integral_constant<int64_t, 4>{});
    case 8:
      return launch_kernel(std::integral_constant<int64_t, 8>{});
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported element type: ",
          primitive_util::LowercasePrimitiveTypeName(element_type),
          " (bit width ", xla::primitive_util::BitWidth(element_type),
          ") for RaggedAllToAll kernel."));
  }
}
}  // namespace xla::gpu
