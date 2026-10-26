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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_device_kernel.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

struct RaggedAllToAllLaunchParams {
  int64_t vector_size_bytes;
  int64_t num_vectorized_row_elements;
  se::ThreadDim thread_dims;
  se::BlockDim block_dims;
};

absl::StatusOr<RaggedAllToAllLaunchParams> ComputeRaggedAllToAllLaunchParams(
    PrimitiveType element_type, int64_t num_outputs,
    int64_t num_updates_per_output, int64_t num_input_rows,
    int64_t num_row_elements) {
  static constexpr size_t kThreads = 128;

  int64_t num_vectorized_row_elements = num_row_elements;
  int64_t vector_size_bytes = xla::primitive_util::ByteWidth(element_type);
  while (num_vectorized_row_elements % 2 == 0 && vector_size_bytes < 8) {
    num_vectorized_row_elements /= 2;
    vector_size_bytes *= 2;
  }
  if (vector_size_bytes != 1 && vector_size_bytes != 2 &&
      vector_size_bytes != 4 && vector_size_bytes != 8) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported element type: ",
        primitive_util::LowercasePrimitiveTypeName(element_type),
        " (bit width ", xla::primitive_util::BitWidth(element_type),
        ") for RaggedAllToAll kernel."));
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

  return RaggedAllToAllLaunchParams{
      vector_size_bytes,
      num_vectorized_row_elements,
      se::ThreadDim(kThreads, 1, 1),
      se::BlockDim(num_outputs, num_block_clusters, num_updates_per_block),
  };
}

template <typename Fn>
auto WithVectorSize(int64_t vector_size_bytes, Fn&& fn)
    -> std::invoke_result_t<Fn, std::integral_constant<int64_t, 1>> {
  switch (vector_size_bytes) {
    case 1:
      return fn(std::integral_constant<int64_t, 1>{});
    case 2:
      return fn(std::integral_constant<int64_t, 2>{});
    case 4:
      return fn(std::integral_constant<int64_t, 4>{});
    case 8:
      return fn(std::integral_constant<int64_t, 8>{});
    default:
      return absl::InternalError("Unreachable vector_size_bytes");
  }
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
    stream_executor::gpu::RaggedAllToAllOutputPtrs output_ptrs,
    se::DeviceAddressBase input_offsets_buffer,
    se::DeviceAddressBase send_sizes_buffer,
    se::DeviceAddressBase output_offsets_buffer, int64_t num_outputs,
    int64_t num_updates_per_output, int64_t num_input_rows,
    int64_t num_row_elements) {
  ABSL_ASSIGN_OR_RETURN(RaggedAllToAllLaunchParams params,
                   ComputeRaggedAllToAllLaunchParams(
                       element_type, num_outputs, num_updates_per_output,
                       num_input_rows, num_row_elements));
  ABSL_ASSIGN_OR_RETURN(
      auto kernel,
      WithVectorSize(params.vector_size_bytes, [&](auto vector_size) {
        return se::gpu::GpuKernelRegistry::GetGlobalRegistry()
            .LoadKernel<
                se::gpu::RaggedAllToAllKernel<decltype(vector_size)::value>>(
                stream->parent());
      }));
  return kernel.Launch(
      params.thread_dims, params.block_dims, stream, input_buffer, output_ptrs,
      input_offsets_buffer, send_sizes_buffer, output_offsets_buffer,
      num_updates_per_output, params.num_vectorized_row_elements);
}

bool IsRaggedAllToAllWithSymmetricMemoryKernelSupported(
    PrimitiveType element_type) {
  // Currently, the kernel doesn't support data types that are smaller
  // than 1 byte.
  return primitive_util::BitWidth(element_type) % 8 == 0;
}

absl::StatusOr<RaggedAllToAllWithSymmetricMemoryKernelType>
LoadRaggedAllToAllWithSymmetricMemoryKernel(se::StreamExecutor* executor,
                                            PrimitiveType element_type,
                                            int64_t num_row_elements) {
  ABSL_ASSIGN_OR_RETURN(RaggedAllToAllLaunchParams params,
                   ComputeRaggedAllToAllLaunchParams(
                       element_type, /*num_outputs=*/1,
                       /*num_updates_per_output=*/1,
                       /*num_input_rows=*/1, num_row_elements));
  return WithVectorSize(params.vector_size_bytes, [&](auto vector_size) {
    return se::gpu::GpuKernelRegistry::GetGlobalRegistry()
        .LoadKernel<se::gpu::RaggedAllToAllWithSymmetricMemoryKernel<
            decltype(vector_size)::value>>(executor);
  });
}

absl::Status RunRaggedAllToAllWithSymmetricMemoryKernel(
    se::Stream* stream, RaggedAllToAllWithSymmetricMemoryKernelType& kernel,
    PrimitiveType element_type, se::DeviceAddressBase input_buffer,
    xla::SymmetricMemory* output_ptrs_symmetric_memory,
    size_t output_sym_offset, se::DeviceAddressBase input_offsets_buffer,
    se::DeviceAddressBase send_sizes_buffer,
    se::DeviceAddressBase output_offsets_buffer, int64_t num_outputs,
    int64_t num_updates_per_output, int64_t num_input_rows,
    int64_t num_row_elements) {
  ABSL_ASSIGN_OR_RETURN(RaggedAllToAllLaunchParams params,
                   ComputeRaggedAllToAllLaunchParams(
                       element_type, num_outputs, num_updates_per_output,
                       num_input_rows, num_row_elements));
  return kernel.Launch(
      params.thread_dims, params.block_dims, stream, input_buffer,
      output_ptrs_symmetric_memory, output_sym_offset, input_offsets_buffer,
      send_sizes_buffer, output_offsets_buffer, num_updates_per_output,
      params.num_vectorized_row_elements);
}

absl::StatusOr<const se::CommandBuffer::Command*>
RecordRaggedAllToAllWithSymmetricMemoryKernel(
    se::CommandBuffer* command_buffer,
    const RaggedAllToAllWithSymmetricMemoryKernelType& kernel,
    PrimitiveType element_type, se::DeviceAddressBase input_buffer,
    xla::SymmetricMemory* output_ptrs_symmetric_memory,
    size_t output_sym_offset, se::DeviceAddressBase input_offsets_buffer,
    se::DeviceAddressBase send_sizes_buffer,
    se::DeviceAddressBase output_offsets_buffer, int64_t num_outputs,
    int64_t num_updates_per_output, int64_t num_input_rows,
    int64_t num_row_elements,
    const se::CommandBuffer::Command* absl_nullable cmd,
    absl::Span<const se::CommandBuffer::Command* const> dependencies) {
  ABSL_ASSIGN_OR_RETURN(RaggedAllToAllLaunchParams params,
                   ComputeRaggedAllToAllLaunchParams(
                       element_type, num_outputs, num_updates_per_output,
                       num_input_rows, num_row_elements));
  if (cmd == nullptr) {
    return command_buffer->CreateLaunch(
        kernel, params.thread_dims, params.block_dims,
        /*cluster_dims=*/std::nullopt, dependencies, input_buffer,
        output_ptrs_symmetric_memory, output_sym_offset, input_offsets_buffer,
        send_sizes_buffer, output_offsets_buffer, num_updates_per_output,
        params.num_vectorized_row_elements);
  }
  ABSL_RETURN_IF_ERROR(command_buffer->UpdateLaunch(
      cmd, kernel, params.thread_dims, params.block_dims,
      /*cluster_dims=*/std::nullopt, input_buffer, output_ptrs_symmetric_memory,
      output_sym_offset, input_offsets_buffer, send_sizes_buffer,
      output_offsets_buffer, num_updates_per_output,
      params.num_vectorized_row_elements));
  return cmd;
}

namespace {

template <int64_t kVectorSize>
absl::Status LaunchDeviceKernel(
    se::Stream* stream, se::StreamExecutor* executor,
    const se::ThreadDim& thread_dims, const se::BlockDim& block_dims,
    xla::gpu::GpuDeviceCommunicator* dev_comm, xla::SymmetricMemory* send_win,
    xla::SymmetricMemory* recv_win, se::DeviceAddressBase input_offsets_buffer,
    se::DeviceAddressBase send_sizes_buffer,
    se::DeviceAddressBase output_offsets_buffer,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t input_buffer_offset_bytes, int64_t output_buffer_offset_bytes) {
  using KernelTrait = se::gpu::RaggedAllToAllDeviceKernel<kVectorSize>;

  ABSL_ASSIGN_OR_RETURN(auto kernel, se::gpu::GpuKernelRegistry::GetGlobalRegistry()
                                    .LoadKernel<KernelTrait>(executor));

  return kernel.Launch(thread_dims, block_dims, stream, dev_comm, send_win,
                       recv_win, input_offsets_buffer, send_sizes_buffer,
                       output_offsets_buffer, num_updates_per_replica,
                       num_row_elements, input_buffer_offset_bytes,
                       output_buffer_offset_bytes);
}

}  // namespace

absl::Status RunDeviceRaggedAllToAllKernel(
    se::Stream* stream, PrimitiveType element_type,
    xla::gpu::GpuDeviceCommunicator* dev_comm, xla::SymmetricMemory* send_win,
    xla::SymmetricMemory* recv_win, se::DeviceAddressBase input_offsets_buffer,
    se::DeviceAddressBase send_sizes_buffer,
    se::DeviceAddressBase output_offsets_buffer, int64_t num_ranks,
    int64_t num_updates_per_replica, int64_t num_row_elements,
    int64_t cta_count, int64_t input_buffer_offset_bytes,
    int64_t output_buffer_offset_bytes) {
  se::StreamExecutor* executor = stream->parent();
  static constexpr size_t kThreadsPerCta = 512;

  int64_t num_vectorized_row_elements = num_row_elements;
  int64_t vector_size_bytes = xla::primitive_util::ByteWidth(element_type);

  while (num_vectorized_row_elements % 2 == 0 && vector_size_bytes < 16) {
    num_vectorized_row_elements /= 2;
    vector_size_bytes *= 2;
  }

  se::ThreadDim thread_dims(kThreadsPerCta, 1, 1);
  se::BlockDim block_dims(cta_count, 1, 1);

  auto launch = [&](auto type) -> absl::Status {
    return LaunchDeviceKernel<decltype(type)::value>(
        stream, executor, thread_dims, block_dims, dev_comm, send_win, recv_win,
        input_offsets_buffer, send_sizes_buffer, output_offsets_buffer,
        num_updates_per_replica, num_vectorized_row_elements,
        input_buffer_offset_bytes, output_buffer_offset_bytes);
  };

  switch (vector_size_bytes) {
    case 1:
      return launch(std::integral_constant<int64_t, 1>{});
    case 2:
      return launch(std::integral_constant<int64_t, 2>{});
    case 4:
      return launch(std::integral_constant<int64_t, 4>{});
    case 8:
      return launch(std::integral_constant<int64_t, 8>{});
    case 16:
      return launch(std::integral_constant<int64_t, 16>{});
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported element type: ",
          primitive_util::LowercasePrimitiveTypeName(element_type),
          " (bit width ", xla::primitive_util::BitWidth(element_type),
          ") for device RaggedAllToAll kernel."));
  }
}

}  // namespace xla::gpu
