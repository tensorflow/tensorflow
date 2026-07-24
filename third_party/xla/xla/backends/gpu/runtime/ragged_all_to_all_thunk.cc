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

#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/gxl_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_kernel_api.h"
#include "xla/backends/gpu/runtime/collective_memory.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.pb.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/rendezvous.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace gpu {
namespace {

// RaggedAllToAll has 4 operands with ragged tensor metadata: input_offsets,
// send_sizes, output_offsets, and recv_sizes.
constexpr int64_t kNumRaggedMetadataOperands = 4;

struct RaggedAllToAllCommandState : CommandState {
  // MultiGpuBarrier: Device memory buffer for signal values (one per peer).
  // Peers write specific slots in this array to signal this device.
  se::ScopedDeviceAddress<uint8_t> barrier_signal_buffer;

  // MultiGpuBarrier: Device memory for the current local step counter.
  // This value is incremented locally by the kernel after every barrier.
  se::ScopedDeviceAddress<uint8_t> barrier_signal_value;
};

int64_t BarrierSignalBufferBytes() {
  return se::gpu::MultiGpuBarrierKernel::kMaxPeers * sizeof(uint32_t);
}

absl::Status ZeroBarrierSignalBuffers(
    se::Stream& stream, se::DeviceAddressBase barrier_signal_buffer,
    se::DeviceAddressBase barrier_signal_value) {
  RETURN_IF_ERROR(
      stream.MemZero(&barrier_signal_buffer, barrier_signal_buffer.size()));
  RETURN_IF_ERROR(
      stream.MemZero(&barrier_signal_value, barrier_signal_value.size()));
  return stream.BlockHostUntilDone();
}

absl::StatusOr<std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>>>
RendezvousRaggedAllToAllBuffers(
    int device_ordinal, RankId rank, const GpuCliqueKey& clique_key,
    absl::Span<DeviceBufferPair const> device_buffers,
    const se::DeviceAddressBase& barrier_signal_buffer) {
  const se::DeviceAddressBase& output_buffer =
      device_buffers[1].destination_buffer;
  return RendezvousResources(device_ordinal, rank, clique_key, output_buffer,
                             barrier_signal_buffer);
}

RaggedAllToAllConfig GetRaggedAllToAllConfig(
    const HloRaggedAllToAllInstruction* instr) {
  RaggedAllToAllConfig config;
  config.config = GetCollectiveConfig(instr, std::nullopt);

  const Shape& input_size_shape = instr->operand(2)->shape();
  config.num_total_updates = input_size_shape.dimensions(0);
  config.num_input_rows = instr->operand(0)->shape().dimensions(0);
  config.num_row_elements =
      ShapeUtil::ElementsIn(instr->shape()) / instr->shape().dimensions(0);

  config.one_shot_kernel_enabled =
      instr->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel();
  config.use_multi_gpu_barrier_with_nccl_in_one_shot_kernel =
      instr->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl();
  config.use_device_kernel =
      instr->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_ragged_all_to_all_use_device_kernel();

  config.collectives_mode = instr->GetModule()
                                ->config()
                                .debug_options()
                                .xla_gpu_ragged_all_to_all_mode();

  int64_t fast_interconnect_slice_size_override =
      instr->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_unsupported_override_fast_interconnect_slice_size();

  // proto3 doesn't distinguish between unset and zero, so we do the
  // check here.
  if (fast_interconnect_slice_size_override > 0) {
    config.fast_interconnect_slice_size_override =
        fast_interconnect_slice_size_override;
  }

  return config;
}

// Loads the offsets and sizes of the input and output ragged tensors from
// device memory.
//
// The parameter `ragged_metadata_allocs` is a vector of pointers to the buffers
// in the host memory allocated by StreamExecutor to copy data from the device
// memory.
absl::Status LoadRaggedTensorMetadata(
    se::Stream& stream, absl::Span<DeviceBufferPair const> buffers,
    int64_t num_total_updates,
    absl::Span<int64_t* const> ragged_metadata_allocs) {
  const uint64_t metadata_bytes =
      static_cast<uint64_t>(num_total_updates) * sizeof(int64_t);
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    const se::DeviceAddressBase& metadata_buffer = buffers[i + 2].source_buffer;
    TF_RET_CHECK(metadata_buffer.size() >= metadata_bytes)
        << "RaggedAllToAll metadata buffer " << i << " has "
        << metadata_buffer.size() << " bytes, expected at least "
        << metadata_bytes << " bytes";
    RETURN_IF_ERROR(stream.Memcpy(ragged_metadata_allocs[i], metadata_buffer,
                                  metadata_bytes));
  }

  // Wait for the copies to complete.
  if (absl::Status blocked = stream.BlockHostUntilDone(); !blocked.ok()) {
    return absl::InternalError(absl::StrFormat(
        "Failed to complete all kernels launched on stream %p: %s", &stream,
        blocked.message()));
  }

  return absl::OkStatus();
}

// Runs AllToAll on a buffer that contains ragged tensor metadata.
absl::Status RunAllToAllOnIndexBuffer(
    const se::DeviceAddressBase& source_buffer, int64_t num_updates_per_replica,
    const se::DeviceAddressBase& destination_buffer, PrimitiveType element_type,
    se::Stream& stream, Communicator& comm) {
  ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());
  auto* gpu_comm = absl::down_cast<GpuCommunicator*>(&comm);

  Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
    for (int peer = 0; peer < num_ranks; ++peer) {
      int64_t offset = peer * num_updates_per_replica;
      se::DeviceAddressBase send_slice =
          GpuCollectives::Slice(source_buffer, element_type, offset,
                                /*count=*/num_updates_per_replica);
      se::DeviceAddressBase recv_slice =
          GpuCollectives::Slice(destination_buffer, element_type, offset,
                                /*count=*/num_updates_per_replica);
      RETURN_IF_ERROR(gpu_comm->LaunchSend(send_slice, element_type,
                                           /*count=*/num_updates_per_replica,
                                           RankId(peer),
                                           GpuCollectives::On(stream)));
      RETURN_IF_ERROR(gpu_comm->LaunchRecv(recv_slice, element_type,
                                           /*count=*/num_updates_per_replica,
                                           RankId(peer),
                                           GpuCollectives::On(stream)));
    }
    return absl::OkStatus();
  });
  RETURN_IF_ERROR(future.Await());
  return stream.BlockHostUntilDone();
}

// Helper to launch the MultiGpuBarrierKernel.
absl::Status LaunchMultiGpuBarrier(
    se::Stream* stream, RankId rank, int64_t num_ranks,
    const std::vector<RaggedAllToAllRendezvousValue>& participants,
    se::DeviceAddressBase local_barrier_signal_value) {
  std::vector<se::DeviceAddressBase> barrier_peer_addresses;
  barrier_peer_addresses.reserve(participants.size());
  for (const auto& participant : participants) {
    barrier_peer_addresses.push_back(participant.barrier_signal_buffer);
  }
  return xla::gpu::LaunchMultiGpuBarrier(stream, num_ranks, rank,
                                         std::move(barrier_peer_addresses),
                                         local_barrier_signal_value);
}

absl::Status CheckRaggedAllToAllBounds(
    se::Stream& stream, int64_t num_total_updates, int64_t num_row_elements,
    const SymmetricMemory* output_sym_mem, size_t output_sym_offset,
    absl::Span<DeviceBufferPair const> buffers) {
  int device_ordinal = stream.parent()->device_ordinal();
  se::StreamExecutor* stream_executor = stream.parent();

  se::DeviceAddressBase input_buffer = buffers[0].source_buffer;
  PrimitiveType element_type = buffers[0].element_type;

  // Fetch buffers
  size_t copy_bytes = num_total_updates * sizeof(int64_t);
  se::DeviceAddressBase input_offsets_buffer = buffers[2].source_buffer;
  se::DeviceAddressBase send_sizes_buffer = buffers[3].source_buffer;
  se::DeviceAddressBase output_offsets_buffer = buffers[4].source_buffer;
  se::DeviceAddressBase recv_sizes_buffer = buffers[5].source_buffer;

  std::vector<int64_t> input_offsets_host(num_total_updates);
  std::vector<int64_t> send_sizes_host(num_total_updates);
  std::vector<int64_t> output_offsets_host(num_total_updates);
  std::vector<int64_t> recv_sizes_host(num_total_updates);

  RETURN_IF_ERROR(stream_executor->SynchronousMemcpyD2H(
      input_offsets_buffer, copy_bytes, input_offsets_host.data()));

  RETURN_IF_ERROR(stream_executor->SynchronousMemcpyD2H(
      send_sizes_buffer, copy_bytes, send_sizes_host.data()));

  RETURN_IF_ERROR(stream_executor->SynchronousMemcpyD2H(
      output_offsets_buffer, copy_bytes, output_offsets_host.data()));

  RETURN_IF_ERROR(stream_executor->SynchronousMemcpyD2H(
      recv_sizes_buffer, copy_bytes, recv_sizes_host.data()));

  int64_t max_read_index = 0;
  int64_t max_write_index = 0;
  int64_t total_recv_elements = 0;
  for (size_t i = 0; i < num_total_updates; ++i) {
    int64_t in_offset = input_offsets_host[i];
    int64_t out_offset = output_offsets_host[i];
    int64_t send_sz = send_sizes_host[i];
    int64_t recv_sz = recv_sizes_host[i];

    TF_RET_CHECK(in_offset >= 0 && out_offset >= 0)
        << "RaggedAllToAll: Negative offsets detected!";

    TF_RET_CHECK(send_sz >= 0 && recv_sz >= 0)
        << "RaggedAllToAll: Negative sizes detected!";

    max_read_index = std::max(max_read_index, in_offset + send_sz);
    max_write_index = std::max(max_write_index, out_offset + send_sz);
    total_recv_elements += recv_sz;
  }

  size_t element_width = primitive_util::ByteWidth(element_type);

  size_t max_read_bytes =
      static_cast<size_t>(max_read_index) * num_row_elements * element_width;

  size_t max_write_bytes =
      output_sym_offset +
      (static_cast<size_t>(max_write_index) * num_row_elements * element_width);

  size_t min_expected_recv_bytes = static_cast<size_t>(total_recv_elements) *
                                   num_row_elements * element_width;

  size_t actual_input_size = input_buffer.size();
  size_t actual_output_size = output_sym_mem->addr().size();

  // Check Out-of-Bounds Reads
  TF_RET_CHECK(max_read_bytes <= actual_input_size)
      << "RaggedAllToAll: READ violation detected! "
      << "Input read requires " << max_read_bytes
      << " bytes, but the input buffer is only " << actual_input_size
      << " bytes. Kernel launch aborted to prevent "
         "CUDA_ERROR_ILLEGAL_ADDRESS.";

  // Check Out-of-Bounds Writes
  TF_RET_CHECK(max_write_bytes <= actual_output_size)
      << "RaggedAllToAll: WRITE violation detected! "
      << "Output writes require " << max_write_bytes
      << " bytes, but the symmetric memory slice is only " << actual_output_size
      << " bytes. Kernel launch aborted to prevent "
         "CUDA_ERROR_ILLEGAL_ADDRESS.";

  // Check recv_sizes
  TF_RET_CHECK(min_expected_recv_bytes <= actual_output_size)
      << "RaggedAllToAll RECV size sanity check failed! "
      << "The user's recv_sizes tensor expects to receive a total of "
      << min_expected_recv_bytes
      << " bytes, but the local output buffer is only " << actual_output_size
      << " bytes. Downstream ops will overflow.";

  XLA_VLOG_DEVICE(5, device_ordinal)
      << "RaggedAllToAll: Bounds check passed. "
      << "max_read_bytes/actual_input_size: " << max_read_bytes << "/"
      << actual_input_size
      << ",  max_write_bytes/actual_output_size: " << max_write_bytes << "/"
      << actual_output_size << ",  min_expected_recv_bytes/actual_output_size: "
      << min_expected_recv_bytes << "/" << actual_output_size;

  return absl::OkStatus();
}

}  // namespace

RaggedAllToAllThunk::RaggedAllToAllThunk(
    ThunkInfo thunk_info, const HloRaggedAllToAllInstruction* instr,
    std::vector<CollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : RaggedAllToAllThunk(std::move(thunk_info), GetRaggedAllToAllConfig(instr),
                          std::move(buffers)) {}

RaggedAllToAllThunk::RaggedAllToAllThunk(
    ThunkInfo thunk_info, const RaggedAllToAllConfig& config,
    std::vector<CollectiveThunk::Buffer> buffers)
    : CollectiveThunk(Thunk::kRaggedAllToAll, thunk_info, std::move(buffers),
                      CommunicationId(0), config.collectives_mode),
      config_(config) {
  CHECK_EQ(config_.config.operand_element_type.size(), this->buffers().size());
}

/*static*/ absl::Status RaggedAllToAllThunk::CheckImplementable(
    const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kRaggedAllToAll));
    }

    if (!ShapeUtil::IsEffectivelyMostMajorDimension(instr->shape(), 0)) {
      return absl::UnimplementedError(absl::Substitute(
          "ragged-all-to-all must have the ragged dimension (0) in the most "
          "major position in the layout $0.",
          instr->shape().layout().ToString()));
    }

    if (instr->operand(2)->shape().element_type() != S64) {
      return absl::InvalidArgumentError(
          "RaggedAllToAllDecomposer only supports S64 offsets. Was "
          "`ragged-all-to-all-canonicalizer` pass executed?");
    }

    return absl::OkStatus();
  };
  return AddOpDescription<RaggedAllToAllThunk>(status(), instr, replica_count,
                                               partition_count);
}

/*static*/ CollectiveOpGroupMode RaggedAllToAllThunk::GetGroupMode(
    const HloRaggedAllToAllInstruction* instr) {
  return GetRaggedAllToAllConfig(instr).config.group_mode;
}

CollectiveCliqueRequests::CliqueRequirements
RaggedAllToAllThunk::GetCliqueRequirements(const GpuCliqueKey& clique_key) {
  CollectiveCliqueRequests::CliqueRequirements clique_reqs;
  if (UsesDeviceKernel()) {
    clique_reqs.dev_comm = DeviceKernelLsaDevCommRequirements();
  } else if (config_.use_multi_gpu_barrier_with_nccl_in_one_shot_kernel) {
    GpuDeviceCommunicator::Requirements req;
    req.lsa_barrier_count = 1;
    clique_reqs.dev_comm = req;
  }
  return clique_reqs;
}

absl::StatusOr<RaggedAllToAllStreamState*> RaggedAllToAllThunk::InitializeOnce(
    const InitializeParams& params) {
  se::StreamExecutor* executor = params.executor;
  {
    absl::MutexLock lock(mutex_);

    // If the stream state already exists, it means that the thunk has been
    // initialized for this executor.
    auto it = per_stream_states_.find(executor);
    if (it != per_stream_states_.end()) {
      return it->second.get();
    }
  }

  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, config_.config));
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);

  auto state = std::make_unique<RaggedAllToAllStreamState>(
      executor->device_ordinal(), rank.value(), std::move(clique_key));

  // Allocate temp buffers in the host memory to load the sizes and offsets of
  // ragged tensors from device memory.
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> alloc,
                     executor->HostMemoryAllocate(config_.num_total_updates *
                                                  sizeof(int64_t)));
    state->host_buffer_allocs.push_back(std::move(alloc));
  }

  const uint64_t output_offsets_buffer_bytes =
      static_cast<uint64_t>(config_.num_total_updates) * sizeof(int64_t);
  ASSIGN_OR_RETURN(
      state->output_offsets_device_buffer,
      params.buffer_allocations->memory_allocator()->Allocate(
          executor->device_ordinal(), output_offsets_buffer_bytes));

  if (output_offsets_buffer_bytes > 0 &&
      state->output_offsets_device_buffer.is_null()) {
    return absl::InternalError("Failed to allocate output offsets buffer.");
  }

  if (use_multi_gpu_barrier_with_nccl_in_one_shot_kernel() ||
      config_.use_device_kernel) {
    if (config_.fast_interconnect_slice_size_override.has_value()) {
      state->lsa_size = config_.fast_interconnect_slice_size_override.value();
    } else {
      GpuDeviceCommunicator::Requirements req;
      if (UsesDeviceKernel()) {
        req = DeviceKernelLsaDevCommRequirements();
      } else {
        req.lsa_barrier_count = 1;
      }
      ASSIGN_OR_RETURN(auto* dev_comm,
                       params.collective_cliques->GetDeviceComm(
                           state->clique_key, state->rank, req));

      state->lsa_size = dev_comm->lsa_size();
    }
  }
  XLA_VLOG_DEVICE(3, state->device_ordinal)
      << "lsa_size: "
      << (state->lsa_size.has_value() ? absl::StrCat(state->lsa_size.value())
                                      : "null");

  if (is_local(params.local_device_count) ||
      (state->lsa_size.has_value() &&
       state->lsa_size.value() == state->clique_key.num_devices())) {
    using MultiGpuBarrierKernel = se::gpu::MultiGpuBarrierKernel;

    ASSIGN_OR_RETURN(
        std::unique_ptr<se::MemoryAllocator> collective_allocator,
        executor->CreateMemoryAllocator(se::MemorySpace::kCollective));

    // We allocate kMaxPeers to be safe and avoid bounds issues, aligning with
    // the fixed-size kernel logic.
    ASSIGN_OR_RETURN(
        state->barrier_signal_buffer,
        collective_allocator->Allocate(BarrierSignalBufferBytes()));

    // This value acts as the local step counter.
    ASSIGN_OR_RETURN(state->barrier_signal_value,
                     collective_allocator->Allocate(sizeof(uint32_t)));

    ASSIGN_OR_RETURN(state->output_buffer_ptr_storage,
                     collective_allocator->Allocate(
                         MultiGpuBarrierKernel::kMaxPeers * sizeof(void*)));

    RETURN_IF_ERROR(ZeroBarrierSignalBuffers(
        *params.stream, state->barrier_signal_buffer->address(),
        state->barrier_signal_value->address()));

    ASSIGN_OR_RETURN(
        std::vector<DeviceBufferPair> device_buffers,
        ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                               config_.config.operand_element_type));
  }

  RaggedAllToAllStreamState* state_ptr = state.get();
  {
    absl::MutexLock lock(mutex_);
    per_stream_states_.emplace(executor, std::move(state));
  }
  return state_ptr;
}

absl::Status RaggedAllToAllThunk::Initialize(const InitializeParams& params) {
  RETURN_IF_ERROR(CollectiveThunk::Initialize(params));

  ASSIGN_OR_RETURN(RaggedAllToAllStreamState * state, InitializeOnce(params));

  // If the symmetric memory handlers are not initialized, initialize it.
  // This can happen in two scenarios:
  //   1. This is the first time the thunk is initialized.
  //   2. Original NCCL communicator was destroyed and symmetric memory handler
  //      was released. NCCL communicator can be destroyed in comm splitting
  //      process, but generally it should not change between executions, so
  //      it's safe to cache the symmetric handler.
  if (state->lsa_size.has_value() &&
      state->lsa_size.value() == state->clique_key.num_devices()) {
    ASSIGN_OR_RETURN(auto* comm, params.collective_cliques->GetComm(
                                     state->clique_key, state->rank));

    if (state->barrier_signal_symmetric_memory.Expired()) {
      ASSIGN_OR_RETURN(
          auto symmetric_memory,
          comm->CreateSymmetricMemory(state->barrier_signal_buffer->address()));

      ASSIGN_OR_RETURN(state->barrier_signal_symmetric_memory,
                       params.collective_cliques->Tie(
                           state->clique_key, std::move(symmetric_memory)));
    }
  } else if (is_local(params.local_device_count)) {
    // Rendezvous - Exchange output pointers and barrier signal buffers.
    ASSIGN_OR_RETURN(
        std::vector<DeviceBufferPair> device_buffers,
        ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                               config_.config.operand_element_type));

    ASSIGN_OR_RETURN(
        state->participants,
        RendezvousRaggedAllToAllBuffers(
            state->device_ordinal, state->rank, state->clique_key,
            device_buffers, state->barrier_signal_buffer->address()));
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> RaggedAllToAllThunk::Record(
    const ExecuteParams& execute_params, const RecordParams& record_params,
    RecordAction record_action, se::CommandBuffer* command_buffer) {
  se::StreamExecutor* executor = execute_params.stream->parent();

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "RaggedAllToAllThunk requires collective parameters and cliques");
  }

  TF_RET_CHECK(IsAllReplicasLocal(
      execute_params.collective_params->local_device_count,
      config_.config.replica_groups, config_.config.group_mode))
      << "RaggedAllToAllThunk: All replicas must be local for the one-shot "
         "kernel to work";

  ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                   GetCollectiveGpuCliqueKey(*execute_params.collective_params,
                                             config_.config));

  int device_ordinal = executor->device_ordinal();
  const std::optional<RankId> rank_opt =
      clique_key.rank(execute_params.collective_params->global_device_id);
  TF_RET_CHECK(rank_opt.has_value())
      << "RaggedAllToAllThunk::Record: Current device is not part of the "
         "clique";
  RankId rank = rank_opt.value();

  ASSIGN_OR_RETURN(
      bool peer_access_enabled,
      execute_params.collective_cliques->peer_access_enabled(clique_key));
  TF_RET_CHECK(peer_access_enabled)
      << "RaggedAllToAllThunk: Peer access must be enabled.";

  absl::Status state_status = absl::OkStatus();
  se::DeviceAddressAllocator* memory_allocator =
      execute_params.buffer_allocations->memory_allocator();
  RaggedAllToAllCommandState* cmd_state =
      record_params.state.GetOrCreate<RaggedAllToAllCommandState>(
          this, command_buffer,
          [&]() -> std::unique_ptr<RaggedAllToAllCommandState> {
            auto state = std::make_unique<RaggedAllToAllCommandState>();

            auto barrier_signal_buffer = memory_allocator->Allocate(
                device_ordinal, BarrierSignalBufferBytes());
            if (!barrier_signal_buffer.ok()) {
              state_status = barrier_signal_buffer.status();
              return nullptr;
            }
            state->barrier_signal_buffer = std::move(*barrier_signal_buffer);

            auto barrier_signal_value =
                memory_allocator->Allocate(device_ordinal, sizeof(uint32_t));
            if (!barrier_signal_value.ok()) {
              state_status = barrier_signal_value.status();
              return nullptr;
            }
            state->barrier_signal_value = std::move(*barrier_signal_value);

            if (state->barrier_signal_buffer.is_null() ||
                state->barrier_signal_value.is_null()) {
              state_status = absl::ResourceExhaustedError(
                  "Failed to allocate RaggedAllToAll barrier buffers");
              return nullptr;
            }

            state_status = ZeroBarrierSignalBuffers(
                *execute_params.stream, state->barrier_signal_buffer.cref(),
                state->barrier_signal_value.cref());
            if (!state_status.ok()) {
              return nullptr;
            }

            return state;
          });
  RETURN_IF_ERROR(state_status);
  TF_RET_CHECK(cmd_state != nullptr)
      << "Failed to get or create RaggedAllToAllCommandState";

  ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers(),
                             config_.config.operand_element_type));

  ASSIGN_OR_RETURN(
      std::unique_ptr<se::CommandBuffer> nested_cmd,
      se::TraceCommandBufferFactory::Create(
          executor, execute_params.command_buffer_trace_stream,
          [&](se::Stream* stream) -> absl::Status {
            ASSIGN_OR_RETURN(
                std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>>
                    participants,
                RendezvousRaggedAllToAllBuffers(
                    device_ordinal, rank, clique_key, device_buffers,
                    cmd_state->barrier_signal_buffer.cref()));

            return RunOneShotRaggedAllToAll(
                clique_key, *stream, rank,
                cmd_state->barrier_signal_buffer.cref(),
                cmd_state->barrier_signal_value.cref(),
                config_.num_total_updates, config_.num_input_rows,
                config_.num_row_elements, device_buffers, *participants);
          }));

  if (priority() != se::StreamPriority::Default) {
    RETURN_IF_ERROR(nested_cmd->SetPriority(priority()));
  }

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateChildCommand(*nested_cmd,
                                              create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(
        command_buffer->UpdateChildCommand(update->command, *nested_cmd));
    return update->command;
  }
  return Internal("Invalid record action");
}

bool RaggedAllToAllThunk::IsOneShotKernelSupported() const {
  if (config_.config.replica_groups.empty() ||
      config_.config.operand_element_type.empty()) {
    return false;
  }

  // In a collective, the number of ranks/outputs matches the size of the
  // replica group.
  int64_t num_outputs = config_.config.replica_groups[0].replica_ids_size();
  PrimitiveType element_type = config_.config.operand_element_type[0];

  return IsRaggedAllToAllKernelSupported(num_outputs, element_type);
}

absl::StatusOr<std::unique_ptr<RaggedAllToAllThunk>>
RaggedAllToAllThunk::FromProto(
    ThunkInfo thunk_info, const RaggedAllToAllThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  std::optional<int64_t> fast_interconnect_slice_size_override;
  if (thunk_proto.fast_interconnect_slice_size_override() > 0) {
    fast_interconnect_slice_size_override =
        thunk_proto.fast_interconnect_slice_size_override();
  }

  return std::make_unique<RaggedAllToAllThunk>(
      std::move(thunk_info),
      RaggedAllToAllConfig{
          config, thunk_proto.num_total_updates(), thunk_proto.num_input_rows(),
          thunk_proto.num_row_elements(), thunk_proto.one_shot_kernel_enabled(),
          thunk_proto.use_multi_gpu_barrier_with_nccl_in_one_shot_kernel(),
          thunk_proto.collectives_mode(), thunk_proto.use_device_kernel(),
          fast_interconnect_slice_size_override},
      std::move(buffers));
}

absl::StatusOr<ThunkProto> RaggedAllToAllThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  RaggedAllToAllThunkProto* thunk_proto =
      proto.mutable_ragged_all_to_all_thunk();

  for (const Buffer& buffer : buffers()) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();

  thunk_proto->set_num_total_updates(config_.num_total_updates);
  thunk_proto->set_num_input_rows(config_.num_input_rows);
  thunk_proto->set_num_row_elements(config_.num_row_elements);
  thunk_proto->set_one_shot_kernel_enabled(is_one_shot_kernel_enabled());
  thunk_proto->set_use_multi_gpu_barrier_with_nccl_in_one_shot_kernel(
      use_multi_gpu_barrier_with_nccl_in_one_shot_kernel());
  thunk_proto->set_collectives_mode(collectives_mode());
  thunk_proto->set_use_device_kernel(config_.use_device_kernel);
  thunk_proto->set_fast_interconnect_slice_size_override(
      config_.fast_interconnect_slice_size_override.value_or(0));

  return proto;
}

absl::Status RaggedAllToAllThunk::RunCollective(const ExecuteParams& params,
                                                const GpuCliqueKey& clique_key,
                                                se::Stream& stream,
                                                Communicator& comm) {
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params.buffer_allocations, buffers(),
                                          config_.config.operand_element_type));

  ASSIGN_OR_RETURN(bool peer_access_enabled,
                   params.collective_cliques->peer_access_enabled(clique_key));

  RaggedAllToAllStreamState* state = nullptr;
  {
    absl::MutexLock lock(mutex_);
    state = per_stream_states_[stream.parent()].get();
  }

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
  if (UsesDeviceKernel() && gpu_comm->SupportsDeviceComm() &&
      params.collective_memory != nullptr) {
    auto [input_sym, input_offset] =
        params.collective_memory->FindSymmetricMemory(
            clique_key, device_buffers[0].source_buffer);
    auto [output_sym, output_offset] =
        params.collective_memory->FindSymmetricMemory(
            clique_key, device_buffers[1].destination_buffer);

    if (input_sym != nullptr && output_sym != nullptr) {
      ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());
      ASSIGN_OR_RETURN(
          auto* lsa_dev_comm,
          params.collective_cliques->GetDeviceComm(
              clique_key, state->rank, DeviceKernelLsaDevCommRequirements()));

      const int64_t lsa_size = lsa_dev_comm->lsa_size();
      const bool has_remote_peers = lsa_size < num_ranks;
      if (has_remote_peers && !gpu_comm->SupportsGin()) {
        XLA_VLOG_DEVICE(3, state->device_ordinal)
            << "Device kernel skipped: lsa_size=" << lsa_size
            << " num_ranks=" << num_ranks << " requires GIN";
      } else {
        GpuDeviceCommunicator* dev_comm = lsa_dev_comm;
        const bool gin = has_remote_peers && gpu_comm->SupportsGin();
        if (has_remote_peers) {
          ASSIGN_OR_RETURN(dev_comm, params.collective_cliques->GetDeviceComm(
                                         clique_key, state->rank,
                                         DeviceKernelDevCommRequirements()));
        }

        const int64_t num_updates_per_replica =
            config_.num_total_updates / num_ranks;
        // Remote peers are reached via GIN puts; local peers via LSA copies.
        const int64_t num_active_updates =
            (gin ? num_ranks : lsa_size) * num_updates_per_replica;
        const int32_t cta_count = DeviceKernelLaunchCtaCount(
            stream.parent()->GetDeviceDescription().core_count(),
            num_active_updates);
        const PrimitiveType element_type = device_buffers[0].element_type;

        XLA_VLOG_DEVICE(3, state->device_ordinal)
            << "Device kernel: lsa_size=" << lsa_size
            << " num_ranks=" << num_ranks << " gin=" << gin
            << " cta_count=" << cta_count
            << " num_updates_per_replica=" << num_updates_per_replica
            << " num_row_elements=" << config_.num_row_elements
            << " element_type="
            << primitive_util::LowercasePrimitiveTypeName(element_type);

        return RunDeviceRaggedAllToAllKernel(
            &stream, element_type, dev_comm, input_sym, output_sym,
            device_buffers[2].source_buffer, device_buffers[3].source_buffer,
            device_buffers[4].source_buffer, num_ranks, num_updates_per_replica,
            config_.num_row_elements, cta_count,
            static_cast<int64_t>(input_offset),
            static_cast<int64_t>(output_offset));
      }
    }
  }

  FabricHomogeneity homogeneity =
      CheckFabricHomogeneity(stream.parent(), clique_key);
  // The fabric is "safe" unless we have confirmed a mismatch.
  // This allows H100/Borg (kUnknown) to still use one-shot kernels.
  // FabricInfo queries are unavailable with old driver (e.g. 535.xx).
  bool fabric_safe = (homogeneity != FabricHomogeneity::kHeterogeneous);

  if (is_one_shot_kernel_enabled() && fabric_safe) {
    if (use_multi_gpu_barrier_with_nccl_in_one_shot_kernel() &&
        IsRaggedAllToAllWithSymmetricMemoryKernelSupported(
            config_.config.operand_element_type[0]) &&
        state->lsa_size.has_value() &&
        state->lsa_size.value() == clique_key.num_devices()) {
      SymmetricMemory* output_sym_mem = nullptr;
      size_t output_sym_offset = 0;
      const BufferAllocation::Slice& out_slice =
          buffers()[1].destination_buffer.slice;
      std::tie(output_sym_mem, output_sym_offset) =
          params.collective_memory->FindSymmetricMemory(clique_key, out_slice);

      if (output_sym_mem == nullptr) {
        return Internal(
            "Symmetric memory not found for destination buffer slice [%s] "
            "in clique %v",
            out_slice.ToString(), clique_key);
      }
      return RunOneShotRaggedAllToAllWithNccl(
          clique_key, stream, state->rank,
          state->barrier_signal_symmetric_memory.Lock(),
          state->barrier_signal_value->address(), output_sym_mem,
          output_sym_offset, config_.num_total_updates, config_.num_input_rows,
          config_.num_row_elements, device_buffers);
    }

    if (state->participants != nullptr && IsOneShotKernelSupported() &&
        peer_access_enabled &&
        is_local(params.collective_params->local_device_count)) {
      return RunOneShotRaggedAllToAll(
          clique_key, stream, state->rank,
          state->barrier_signal_buffer
              ->address(),  // Buff peers write signals to
          state->barrier_signal_value
              ->address(),  // Local monotonic step counter
          config_.num_total_updates, config_.num_input_rows,
          config_.num_row_elements, device_buffers, *state->participants);
    }
  }

  // Get buffer allocs to load sizes and offsets of ragged tensors from device
  // memory.
  absl::InlinedVector<int64_t*, 8> ragged_metadata_allocs;
  ragged_metadata_allocs.reserve(kNumRaggedMetadataOperands);
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    ragged_metadata_allocs.push_back(reinterpret_cast<int64_t*>(
        state->host_buffer_allocs[i]->address().opaque()));
  }

  SymmetricMemory* output_sym_mem = nullptr;
  size_t output_base_offset = 0;
  if (use_symmetric_memory() && params.collective_memory) {
    auto [sym, offset] = params.collective_memory->FindSymmetricMemory(
        clique_key, device_buffers[1].destination_buffer);
    output_sym_mem = sym;
    output_base_offset = offset;
    XLA_VLOG_DEVICE(3, stream.parent()->device_ordinal())
        << "RaggedAllToAllThunk::RunCollective: FindSymmetricMemory "
        << "returned sym=" << (sym ? "non-null" : "NULL")
        << " offset=" << offset;
  }

  return RunRaggedAllToAll(config_.num_row_elements, config_.num_total_updates,
                           device_buffers, stream, comm, ragged_metadata_allocs,
                           state->output_offsets_device_buffer.cref(),
                           collectives_mode(), output_sym_mem,
                           output_base_offset, state->rank.value());
}

// Executes the rendezvous to exchange buffer addresses and barrier signal
// buffers.
absl::StatusOr<std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>>>
RendezvousResources(int device_ordinal, RankId rank,
                    const GpuCliqueKey& clique_key,
                    const se::DeviceAddressBase& output_buffer,
                    const se::DeviceAddressBase& barrier_signal_buffer) {
  int64_t num_ranks = clique_key.num_local_participants();

  RaggedAllToAllRendezvousValue rendezvous_value;
  rendezvous_value.rank = rank;
  rendezvous_value.output_buffer = output_buffer;
  rendezvous_value.barrier_signal_buffer = barrier_signal_buffer;

  auto rendezvous_fn =
      [](absl::Span<const RaggedAllToAllRendezvousValue* const> values) {
        std::vector<RaggedAllToAllRendezvousValue> values_copy;
        for (const auto& value : values) {
          values_copy.push_back(*value);
        }
        // Sort to make sure that values are in the same order as the devices
        // are ordered in the communicator.
        absl::c_sort(values_copy);
        return values_copy;
      };

  auto rendezvous_name = absl::StrFormat(
      "[%d] [rank=%v] One-shot ragged-all-to-all rendezvous, clique: %v",
      device_ordinal, rank, clique_key);

  return Rendezvous<std::vector<RaggedAllToAllRendezvousValue>>(
      rendezvous_name, clique_key, rendezvous_value, num_ranks, rendezvous_fn);
}

absl::Status RaggedAllToAllThunk::PrepareCollective(
    const PrepareParams& params, const GpuCliqueKey& clique_key) {
  if (config_.use_multi_gpu_barrier_with_nccl_in_one_shot_kernel ||
      use_symmetric_memory()) {
    // Request symmetric memory for the output buffer only.
    const Buffer& output_buffer = buffers()[1];
    RETURN_IF_ERROR(
        params.collective_memory_requests->RequestSymmetricAllocationSlice(
            clique_key, output_buffer.destination_buffer.slice));
  }

  if (UsesDeviceKernel()) {
    const Buffer& input_buf = buffers()[0];
    RETURN_IF_ERROR(
        params.collective_memory_requests->RequestSymmetricAllocation(
            clique_key, input_buf.source_buffer.slice.index()));

    RETURN_IF_ERROR(device_groups().status());
    CollectiveCliqueRequests::CliqueRequirements gin_reqs;
    gin_reqs.dev_comm = DeviceKernelDevCommRequirements();
    RETURN_IF_ERROR(params.collective_clique_requests->RequestClique(
        clique_key, *device_groups(), gin_reqs));
  }

  return absl::OkStatus();
}

absl::Status RunRaggedAllToAll(
    int64_t ragged_row_element_size, int64_t num_total_updates,
    const std::vector<DeviceBufferPair>& original_buffers, se::Stream& stream,
    Communicator& comm, absl::Span<int64_t* const> ragged_metadata_allocs,
    const se::DeviceAddressBase& output_offsets_device_buffer,
    CollectiveThunk::CollectivesMode collectives_mode,
    SymmetricMemory* output_symmetric_memory, size_t output_base_offset,
    int64_t rank) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Performing ragged-all-to-all from device ordinal: " << device_ordinal;
  ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
  if (gpu_comm->gxl_communicator() != nullptr) {
    GxlCommunicator* gxl_nccl_comm = gpu_comm->gxl_communicator();
    return gxl_nccl_comm->RunRaggedAllToAllGxl(
        &stream, original_buffers[0].element_type,
        original_buffers[0].source_buffer,
        original_buffers[1].destination_buffer,
        original_buffers[2].source_buffer, original_buffers[3].source_buffer,
        original_buffers[4].source_buffer, original_buffers[5].source_buffer,
        ragged_row_element_size, num_total_updates, rank);
  }

  std::vector<DeviceBufferPair> buffers = original_buffers;

  int64_t num_updates_per_replica = num_total_updates / num_ranks;

  bool use_put_path =
      collectives_mode == DebugOptions::COLLECTIVES_SYMMETRIC_MEMORY &&
      output_symmetric_memory != nullptr;

  if (!use_put_path) {
    // `output_offsets` of the RaggedAllToAll instruction are sharded in a way,
    // that `output_offset[i]` is an offset in the i-th peer output buffer. To
    // make it work for NCCL model with send/recv, we need to know offsets in
    // the local output buffer. To get the correct offsets we perform an
    // AllToAll on the output_offsets buffer.
    DeviceBufferPair& output_offsets_buffer_pair = buffers[4];
    RETURN_IF_ERROR(RunAllToAllOnIndexBuffer(
        output_offsets_buffer_pair.source_buffer, num_updates_per_replica,
        output_offsets_device_buffer, output_offsets_buffer_pair.element_type,
        stream, comm));
    output_offsets_buffer_pair.source_buffer = output_offsets_device_buffer;
  }

  RETURN_IF_ERROR(LoadRaggedTensorMetadata(stream, buffers, num_total_updates,
                                           ragged_metadata_allocs));

  const int64_t* input_offsets = ragged_metadata_allocs[0];
  const int64_t* send_sizes = ragged_metadata_allocs[1];
  const int64_t* output_offsets = ragged_metadata_allocs[2];

  if (use_put_path) {
    XLA_VLOG_DEVICE(3, device_ordinal)
        << "RunRaggedAllToAll: using Put+Signal path";
    PrimitiveType element_type = buffers[0].element_type;
    int64_t element_byte_width = primitive_util::ByteWidth(element_type);
    se::DeviceAddressBase input_buffer = buffers[0].source_buffer;

    Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
      for (int peer = 0; peer < num_ranks; ++peer) {
        for (int64_t i = 0; i < num_updates_per_replica; ++i) {
          const int64_t idx = peer * num_updates_per_replica + i;
          const int64_t send_count = send_sizes[idx] * ragged_row_element_size;

          const se::DeviceAddressBase send_slice = GpuCollectives::Slice(
              input_buffer, element_type,
              input_offsets[idx] * ragged_row_element_size, send_count);

          const size_t byte_offset =
              output_base_offset + output_offsets[idx] *
                                       ragged_row_element_size *
                                       element_byte_width;
          const size_t byte_count = send_count * element_byte_width;

          RETURN_IF_ERROR(gpu_comm->LaunchPut(
              send_slice, output_symmetric_memory, byte_offset, byte_count,
              RankId(peer), GpuCollectives::On(stream)));
        }
      }
      return absl::OkStatus();
    });
    RETURN_IF_ERROR(future.Await());

    GpuSignalDesc signal_desc(/*sig_idx=*/0, /*ctx=*/0);
    for (int peer = 0; peer < num_ranks; ++peer) {
      RETURN_IF_ERROR(comm.WaitSignal(RankId(peer),
                                      /*op_cnt=*/num_updates_per_replica,
                                      signal_desc, GpuCollectives::On(stream))
                          .Await());
    }

    return absl::OkStatus();
  }

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "RunRaggedAllToAll: using Send/Recv path";
  const int64_t* recv_sizes = ragged_metadata_allocs[3];

  Future<> future = gpu_comm->GroupExecute([&]() -> absl::Status {
    PrimitiveType element_type = buffers[0].element_type;

    se::DeviceAddressBase input_buffer = buffers[0].source_buffer;
    se::DeviceAddressBase output_buffer = buffers[1].destination_buffer;

    for (int64_t i = 0; i < num_updates_per_replica; ++i) {
      for (int peer = 0; peer < num_ranks; ++peer) {
        int64_t idx = peer * num_updates_per_replica + i;
        se::DeviceAddressBase send_slice =
            GpuCollectives::Slice(input_buffer, element_type,
                                  input_offsets[idx] * ragged_row_element_size,
                                  send_sizes[idx] * ragged_row_element_size);

        se::DeviceAddressBase recv_slice =
            GpuCollectives::Slice(output_buffer, element_type,
                                  output_offsets[idx] * ragged_row_element_size,
                                  recv_sizes[idx] * ragged_row_element_size);

        RETURN_IF_ERROR(gpu_comm->LaunchSend(
            send_slice, element_type, send_sizes[idx] * ragged_row_element_size,
            RankId(peer), GpuCollectives::On(stream)));

        RETURN_IF_ERROR(gpu_comm->LaunchRecv(
            recv_slice, element_type, recv_sizes[idx] * ragged_row_element_size,
            RankId(peer), GpuCollectives::On(stream)));
      }
    }

    return absl::OkStatus();
  });
  return future.Await();
}

struct CharFormatter {
  void operator()(std::string* out, char c) const {
    // Append the character to the output string
    absl::StrAppend(out, static_cast<int>(c));
  }
};

// Executes the RaggedAllToAll collective using a "One-Shot" kernel with
// explicit device-side synchronization.
// The execution flow is:
// Prerequisite: Rendezvous - Exchange output buffers and barrier signal
// buffers with peers.
// 1. Pre-Kernel Barrier: Wait until all peers are ready to receive data.
// 2. Execution: Run the RaggedAllToAll kernel (direct P2P writes).
// 3. Post-Kernel Barrier: Wait until all peers have finished writing.
absl::Status RunOneShotRaggedAllToAllWithNccl(
    const GpuCliqueKey& clique_key, se::Stream& stream, RankId rank,
    std::shared_ptr<xla::SymmetricMemory> barrier_signal_symmetric_memory,
    const se::DeviceAddressBase& barrier_signal_value,
    SymmetricMemory* output_sym_mem, size_t output_sym_offset,
    int64_t num_total_updates, int64_t num_input_rows, int64_t num_row_elements,
    absl::Span<DeviceBufferPair const> buffers) {
  int device_ordinal = stream.parent()->device_ordinal();
  const int64_t num_ranks = clique_key.num_devices();

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "RaggedAllToAll (One-Shot NCCL) STARTED. Rank: " << rank.value()
      << ", Total Updates: " << num_total_updates;

  PrimitiveType element_type = buffers[0].element_type;
  se::DeviceAddressBase input_buffer = buffers[0].source_buffer;
  se::DeviceAddressBase output_buffer = buffers[1].destination_buffer;
  XLA_VLOG_DEVICE(4, device_ordinal)
      << "Performing one-shot ragged-all-to-all "
         "with NCCL barrier. input buffer: ("
      << input_buffer.opaque() << ", size=" << input_buffer.size()
      << ") output buffer: (" << output_buffer.opaque()
      << ", size=" << output_buffer.size() << ")"
      << " output sym memory (handle=" << output_sym_mem
      << ", address=" << output_sym_mem->addr().opaque()
      << ", size=" << output_sym_mem->addr().size()
      << ", sym_offset=" << output_sym_offset << ")"
      << " barrier signal symmetric memory (handle="
      << barrier_signal_symmetric_memory.get()
      << ", address=" << barrier_signal_symmetric_memory->addr().opaque()
      << ", size=" << barrier_signal_symmetric_memory->addr().size() << ")";

  // 1. Barrier (Pre-Kernel)
  // Global synchronization before P2P writes.
  // Ensures that all peers have reached this point and their output buffers
  // are ready to receive data. This prevents the kernel from attempting to
  // write to a peer's memory before that peer has completed the rendezvous
  // setup.
  RETURN_IF_ERROR(xla::gpu::LaunchMultiGpuBarrierWithNccl(
      &stream, num_ranks, rank, barrier_signal_symmetric_memory.get(),
      barrier_signal_value));

  // 2. Execution of RunRaggedAllToAllKernel
  const int64_t num_updates_per_replica = num_total_updates / num_ranks;

  if (VLOG_IS_ON(5)) {
    RETURN_IF_ERROR(CheckRaggedAllToAllBounds(stream, num_total_updates,
                                              num_row_elements, output_sym_mem,
                                              output_sym_offset, buffers));
  }

  RETURN_IF_ERROR(RunRaggedAllToAllWithSymmetricMemoryKernel(
      &stream, element_type, input_buffer, output_sym_mem, output_sym_offset,
      buffers[2].source_buffer, buffers[3].source_buffer,
      buffers[4].source_buffer, num_ranks, num_updates_per_replica,
      num_input_rows, num_row_elements));

  // 3. Barrier (Post-Kernel)
  // Global synchronization to ensure data consistency.
  // We wait for all peers to signal completion.
  // This guarantees that all P2P writes to our output buffer are complete and
  // safe to consume.
  RETURN_IF_ERROR(xla::gpu::LaunchMultiGpuBarrierWithNccl(
      &stream, num_ranks, rank, barrier_signal_symmetric_memory.get(),
      barrier_signal_value));

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "RaggedAllToAll (One-Shot NCCL) FINISHED. Rank: " << rank.value();

  if (VLOG_IS_ON(6)) {
    RETURN_IF_ERROR(stream.BlockHostUntilDone());

    se::StreamExecutor* stream_executor = stream.parent();
    std::vector<char> input_buffer_host;
    input_buffer_host.resize(input_buffer.size());
    RETURN_IF_ERROR(stream_executor->SynchronousMemcpyD2H(
        input_buffer, input_buffer.size(), input_buffer_host.data()));
    XLA_VLOG_DEVICE(6, device_ordinal)
        << "Ragged-all-to-all with NCCL input buffer: "
        << absl::StrJoin(input_buffer_host, ",", CharFormatter{});

    std::vector<char> output_buffer_host;
    output_buffer_host.resize(output_buffer.size());
    RETURN_IF_ERROR(stream_executor->SynchronousMemcpyD2H(
        output_buffer, output_buffer.size(), output_buffer_host.data()));
    XLA_VLOG_DEVICE(6, device_ordinal)
        << "Ragged-all-to-all with NCCL output before kernel: "
        << absl::StrJoin(output_buffer_host, ",", CharFormatter{});
  }

  return absl::OkStatus();
}

// Executes the RaggedAllToAll collective using a "One-Shot" kernel with
// explicit device-side synchronization.
// The execution flow is:
// Prerequisite: Rendezvous - Exchange output buffers and barrier signal
// buffers with peers.
// 1. Pre-Kernel Barrier: Wait until all peers are ready to receive data.
// 2. Execution: Run the RaggedAllToAll kernel (direct P2P writes).
// 3. Post-Kernel Barrier: Wait until all peers have finished writing.
absl::Status RunOneShotRaggedAllToAll(
    const GpuCliqueKey& clique_key, se::Stream& stream, RankId rank,
    const se::DeviceAddressBase& barrier_signal_buffer,
    const se::DeviceAddressBase& barrier_signal_value,
    int64_t num_total_updates, int64_t num_input_rows, int64_t num_row_elements,
    absl::Span<DeviceBufferPair const> buffers,
    const std::vector<RaggedAllToAllRendezvousValue>& participants) {
  int device_ordinal = stream.parent()->device_ordinal();
  const int64_t num_ranks = clique_key.num_local_participants();

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Performing one-shot ragged-all-to-all rank: " << rank.value();

  PrimitiveType element_type = buffers[0].element_type;
  se::DeviceAddressBase input_buffer = buffers[0].source_buffer;

  // 1. Barrier (Pre-Kernel)
  // Global synchronization before P2P writes.
  // Ensures that all peers have reached this point and their output buffers are
  // ready to receive data. This prevents the kernel from attempting to write
  // to a peer's memory before that peer has completed the rendezvous setup.
  RETURN_IF_ERROR(LaunchMultiGpuBarrier(&stream, rank, num_ranks, participants,
                                        barrier_signal_value));

  // 2. Execution of RunRaggedAllToAllKernel
  const int64_t num_updates_per_replica = num_total_updates / num_ranks;

  stream_executor::gpu::RaggedAllToAllOutputPtrs output_ptrs;
  for (int i = 0; i < participants.size(); ++i) {
    output_ptrs[i] = participants[i].output_buffer.opaque();
  }

  RETURN_IF_ERROR(RunRaggedAllToAllKernel(
      &stream, element_type, input_buffer, output_ptrs,
      buffers[2].source_buffer, buffers[3].source_buffer,
      buffers[4].source_buffer, num_ranks, num_updates_per_replica,
      num_input_rows, num_row_elements));

  // 3. Barrier (Post-Kernel)
  // Global synchronization to ensure data consistency.
  // We wait for all peers to signal completion.
  // This guarantees that all P2P writes to our output buffer are complete and
  // safe to consume.
  RETURN_IF_ERROR(LaunchMultiGpuBarrier(&stream, rank, num_ranks, participants,
                                        barrier_signal_value));

  return absl::OkStatus();
}

bool RaggedAllToAllThunk::is_local(int device_count) const {
  return IsAllReplicasLocal(device_count, config_.config.replica_groups,
                            config_.config.group_mode);
}

}  // namespace gpu
}  // namespace xla
