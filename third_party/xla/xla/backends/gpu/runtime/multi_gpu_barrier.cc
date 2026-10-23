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

#include "xla/backends/gpu/runtime/multi_gpu_barrier.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/ffi/api/record_api.h"
#include "xla/ffi/api/record_c_api.h"
#include "xla/ffi/record_ffi.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {
namespace se = ::stream_executor;

namespace {

// Returns a cached kernel for the given stream executor.
template <typename Kernel>
absl::StatusOr<typename Kernel::KernelType*> GetCachedKernel(
    se::StreamExecutor* executor) {
  // Pre-size to 16 slots so all standard single-host GPU topologies
  // (up to 16 GPUs, e.g. DGX H100/B200) hit the lock-free array fast path.
  static absl::NoDestructor<
      PerDeviceState<std::unique_ptr<typename Kernel::KernelType>>>
      kernel_per_device(/*num_devices=*/16);
  int device_ordinal = executor->device_ordinal();
  ABSL_RETURN_IF_ERROR(kernel_per_device->GetOrCreateAndInitialize(
      device_ordinal,
      [&](std::unique_ptr<typename Kernel::KernelType>* slot) -> absl::Status {
        ABSL_ASSIGN_OR_RETURN(
            auto new_kernel,
            stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
                .LoadKernel<Kernel>(executor));
        *slot = std::make_unique<typename Kernel::KernelType>(
            std::move(new_kernel));
        return absl::OkStatus();
      }));
  return kernel_per_device->Find(device_ordinal)->get();
}

absl::Status ValidateBarrierWithNcclInputs(
    int64_t num_devices, RankId rank,
    const xla::SymmetricMemory* symmetric_memory) {
  using Kernel = se::gpu::MultiGpuBarrierWithNcclKernel;
  TF_RET_CHECK(symmetric_memory != nullptr)
      << "Symmetric memory is required on rank " << rank.value();
  TF_RET_CHECK(num_devices <= Kernel::kMaxPeers)
      << "Number of participants exceeds "
         "MultiGpuBarrierWithNcclKernel::kMaxPeers";
  return absl::OkStatus();
}

}  // namespace

// Launches a cross-GPU barrier synchronization.
//
// This implements a decentralized peer-to-peer barrier synchronization:
// 1. Each device maintains a signal buffer array (one slot per peer) and a
//    local monotonic step counter.
// 2. During execution, a device writes to its designated slot in *every*
//    peer's signal buffer to indicate arrival.
// 3. The device then waits locally for all slots in its own signal buffer to
//    match the expected step value (confirming all peers have arrived).
//
// See MultiGpuBarrierKernel for more details.
absl::Status LaunchMultiGpuBarrier(
    stream_executor::Stream* stream, int64_t num_devices, RankId rank,
    const std::vector<stream_executor::DeviceAddressBase>& barrier_addresses,
    stream_executor::DeviceAddressBase local_barrier_signal_value) {
  using MultiGpuBarrierKernel = stream_executor::gpu::MultiGpuBarrierKernel;

  TF_RET_CHECK(num_devices <= MultiGpuBarrierKernel::kMaxPeers)
      << "Number of participants exceeds MultiGpuBarrierKernel::kMaxPeers";
  TF_RET_CHECK(barrier_addresses.size() == num_devices)
      << "Number of barrier addresses does not match number of peers";

  std::array<void*, MultiGpuBarrierKernel::kMaxPeers> signal_buffers;
  std::fill(signal_buffers.begin(), signal_buffers.end(), nullptr);

  for (int peer = 0; peer < num_devices; ++peer) {
    signal_buffers[peer] = barrier_addresses[peer].opaque();
  }

  ABSL_ASSIGN_OR_RETURN(MultiGpuBarrierKernel::KernelType * kernel,
                   GetCachedKernel<MultiGpuBarrierKernel>(stream->parent()));

  stream_executor::DeviceAddress<uint32_t> typed_sync_counter(
      local_barrier_signal_value);

  return kernel->Launch(
      stream_executor::ThreadDim(MultiGpuBarrierKernel::kMaxPeers, 1, 1),
      stream_executor::BlockDim(1, 1, 1), stream,
      static_cast<int64_t>(rank.value()), static_cast<int64_t>(num_devices),
      signal_buffers, typed_sync_counter);
}

absl::StatusOr<se::gpu::MultiGpuBarrierWithNcclKernel::KernelType>
LoadMultiGpuBarrierWithNcclKernel(se::StreamExecutor* executor) {
  TF_RET_CHECK(executor != nullptr) << "StreamExecutor is required";
  return se::gpu::GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<se::gpu::MultiGpuBarrierWithNcclKernel>(executor);
}

// See MultiGpuBarrierWithNcclKernel for more details.
absl::Status LaunchMultiGpuBarrierWithNccl(
    stream_executor::Stream* stream,
    se::gpu::MultiGpuBarrierWithNcclKernel::KernelType& kernel,
    int64_t num_devices, RankId rank, xla::SymmetricMemory* symmetric_memory,
    stream_executor::DeviceAddressBase local_barrier_signal_value) {
  using MultiGpuBarrierWithNcclKernel =
      stream_executor::gpu::MultiGpuBarrierWithNcclKernel;

  TF_RET_CHECK(static_cast<bool>(kernel)) << "Barrier kernel must be loaded";
  ABSL_RETURN_IF_ERROR(
      ValidateBarrierWithNcclInputs(num_devices, rank, symmetric_memory));

  stream_executor::DeviceAddress<uint32_t> typed_sync_counter(
      local_barrier_signal_value);

  return kernel.Launch(stream_executor::ThreadDim(
                           MultiGpuBarrierWithNcclKernel::kMaxPeers, 1, 1),
                       stream_executor::BlockDim(1, 1, 1), stream,
                       static_cast<int64_t>(rank.value()),
                       static_cast<int64_t>(num_devices), symmetric_memory,
                       typed_sync_counter);
}

absl::StatusOr<const XLA_FFI_Command*> RecordMultiGpuBarrierWithNccl(
    const se::gpu::MultiGpuBarrierWithNcclKernel::KernelType& kernel,
    xla::ffi::RecordContext& record_ctx, int64_t num_devices, RankId rank,
    xla::SymmetricMemory* symmetric_memory,
    stream_executor::DeviceAddressBase local_barrier_signal_value,
    const XLA_FFI_Command* absl_nullable cmd) {
  using MultiGpuBarrierWithNcclKernel =
      stream_executor::gpu::MultiGpuBarrierWithNcclKernel;

  TF_RET_CHECK(static_cast<bool>(kernel)) << "Barrier kernel must be loaded";
  ABSL_RETURN_IF_ERROR(
      ValidateBarrierWithNcclInputs(num_devices, rank, symmetric_memory));

  int64_t rank_val = rank.value();
  int64_t num_devices_val = num_devices;
  std::array<ffi::KernelArg, 4> kernel_args = {
      ffi::HostValue{&rank_val, sizeof(rank_val)},
      ffi::HostValue{&num_devices_val, sizeof(num_devices_val)},
      ffi::DevicePointer{symmetric_memory->PackKernelArg()},
      ffi::DevicePointer{local_barrier_signal_value.opaque()},
  };

  if (record_ctx.action() == ffi::RecordAction::kCreate) {
    XLA_FFI_LaunchDims launch_dims = {
        /*grid=*/{1, 1, 1},
        /*block=*/
        {static_cast<int32_t>(MultiGpuBarrierWithNcclKernel::kMaxPeers), 1, 1},
        /*cluster=*/{0, 0, 0}};
    return record_ctx.CreateLaunch(
        std::string(kernel->name()).c_str(),
        kernel->platform_specific_handle().kernel,
        /*kernel_size=*/0, ffi::SourceFormat::kFunctionPtr, launch_dims,
        /*shared_mem_bytes=*/0, /*uses_pdl=*/false, kernel_args);
  }

  TF_RET_CHECK(record_ctx.action() == ffi::RecordAction::kUpdate)
      << "Unexpected record action: " << static_cast<int>(record_ctx.action());
  TF_RET_CHECK(cmd != nullptr)
      << "Command pointer is required for RecordAction::kUpdate";
  ABSL_RETURN_IF_ERROR(record_ctx.UpdateLaunch(cmd, kernel_args));
  return cmd;
}

absl::StatusOr<const se::CommandBuffer::Command*> RecordMultiGpuBarrierWithNccl(
    se::CommandBuffer* command_buffer,
    const se::gpu::MultiGpuBarrierWithNcclKernel::KernelType& kernel,
    int64_t num_devices, RankId rank, xla::SymmetricMemory* symmetric_memory,
    se::DeviceAddressBase local_barrier_signal_value,
    const se::CommandBuffer::Command* absl_nullable cmd,
    absl::Span<const se::CommandBuffer::Command* const> dependencies) {
  using MultiGpuBarrierWithNcclKernel = se::gpu::MultiGpuBarrierWithNcclKernel;

  TF_RET_CHECK(command_buffer != nullptr) << "CommandBuffer is required";
  TF_RET_CHECK(static_cast<bool>(kernel)) << "Barrier kernel must be loaded";
  ABSL_RETURN_IF_ERROR(
      ValidateBarrierWithNcclInputs(num_devices, rank, symmetric_memory));

  const se::ThreadDim thread_dims(MultiGpuBarrierWithNcclKernel::kMaxPeers, 1,
                                  1);
  const se::BlockDim block_dims(1, 1, 1);
  se::DeviceAddress<uint32_t> typed_sync_counter(local_barrier_signal_value);

  if (cmd == nullptr) {
    return command_buffer->CreateLaunch(
        kernel, thread_dims, block_dims, /*cluster_dims=*/std::nullopt,
        dependencies, static_cast<int64_t>(rank.value()),
        static_cast<int64_t>(num_devices), symmetric_memory,
        typed_sync_counter);
  }
  ABSL_RETURN_IF_ERROR(command_buffer->UpdateLaunch(
      cmd, kernel, thread_dims, block_dims, /*cluster_dims=*/std::nullopt,
      static_cast<int64_t>(rank.value()), static_cast<int64_t>(num_devices),
      symmetric_memory, typed_sync_counter));
  return cmd;
}

size_t GetMultiGpuBarrierSignalBufferSize() {
  return stream_executor::gpu::MultiGpuBarrierWithNcclKernel::kMaxPeers *
         sizeof(uint32_t);
}

size_t GetMultiGpuBarrierSignalValueSize() { return sizeof(uint32_t); }

}  // namespace gpu
}  // namespace xla
