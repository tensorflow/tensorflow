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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/rendezvous.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla {
namespace gpu {
namespace {

// RaggedAllToAll has 4 operands with ragged tensor metadata: input_offsets,
// send_sizes, output_offsets, and recv_sizes.
constexpr int64_t kNumRaggedMetadataOperands = 4;

RaggedAllToAllConfig GetRaggedAllToAllConfig(
    const HloRaggedAllToAllInstruction* instr) {
  RaggedAllToAllConfig config;
  config.config = GetCollectiveConfig(instr, std::nullopt);

  const Shape& input_size_shape = instr->operand(2)->shape();
  config.num_total_updates = input_size_shape.dimensions(0);
  config.num_input_rows = instr->operand(0)->shape().dimensions(0);
  config.num_row_elements =
      ShapeUtil::ElementsIn(instr->shape()) / instr->shape().dimensions(0);
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
    absl::Span<int64_t* const> ragged_metadata_allocs) {
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    TF_RETURN_IF_ERROR(stream.Memcpy(ragged_metadata_allocs[i],
                                     buffers[i + 2].source_buffer,
                                     buffers[i + 2].source_buffer.size()));
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
    const se::DeviceMemoryBase& source_buffer, int64_t num_updates_per_replica,
    const se::DeviceMemoryBase& destination_buffer, PrimitiveType element_type,
    se::Stream& stream, Communicator* comm) {
  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(comm);
  tsl::AsyncValueRef<Communicator::Event> event = gpu_comm->GroupExecute(
      [num_ranks, num_updates_per_replica, element_type, &source_buffer,
       &destination_buffer, &stream](GpuCommunicator* comm) -> absl::Status {
        for (int peer = 0; peer < num_ranks; ++peer) {
          int64_t offset = peer * num_updates_per_replica;
          se::DeviceMemoryBase send_slice =
              GpuCollectives::Slice(source_buffer, element_type, offset,
                                    /*count=*/num_updates_per_replica);
          se::DeviceMemoryBase recv_slice =
              GpuCollectives::Slice(destination_buffer, element_type, offset,
                                    /*count=*/num_updates_per_replica);
          TF_RETURN_IF_ERROR(comm->LaunchSend(send_slice, element_type,
                                              /*count=*/num_updates_per_replica,
                                              RankId(peer),
                                              GpuCollectives::On(stream)));
          TF_RETURN_IF_ERROR(comm->LaunchRecv(recv_slice, element_type,
                                              /*count=*/num_updates_per_replica,
                                              RankId(peer),
                                              GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  return stream.BlockHostUntilDone();
}

absl::Status RunRaggedAllToAll(
    int64_t ragged_row_element_size, int64_t num_total_updates,
    const std::vector<DeviceBufferPair>& original_buffers, se::Stream& stream,
    Communicator* comm, absl::Span<int64_t* const> ragged_metadata_allocs,
    const se::DeviceMemoryBase& output_offsets_device_buffer,
    bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "[" << device_ordinal
          << "] Performing ragged-all-to-all from device ordinal: "
          << device_ordinal;
  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm->NumRanks());

  std::vector<DeviceBufferPair> buffers = original_buffers;

  int64_t num_updates_per_replica = num_total_updates / num_ranks;

  // `output_offsets` of the RaggedAllToAll instruction are sharded in a way,
  // that `output_offset[i]` is an offset in the i-th peer output buffer. To
  // make it work for NCCL model with send/recv, we need to know offsets in the
  // local output buffer. To get the correct offsets we perform an AllToAll on
  // the output_offsets buffer.
  DeviceBufferPair& output_offsets_buffer_pair = buffers[4];
  TF_RETURN_IF_ERROR(RunAllToAllOnIndexBuffer(
      output_offsets_buffer_pair.source_buffer, num_updates_per_replica,
      output_offsets_device_buffer, output_offsets_buffer_pair.element_type,
      stream, comm));
  output_offsets_buffer_pair.source_buffer = output_offsets_device_buffer;

  TF_RETURN_IF_ERROR(
      LoadRaggedTensorMetadata(stream, buffers, ragged_metadata_allocs));

  const int64_t* input_offsets = ragged_metadata_allocs[0];
  const int64_t* send_sizes = ragged_metadata_allocs[1];
  const int64_t* output_offsets = ragged_metadata_allocs[2];
  const int64_t* recv_sizes = ragged_metadata_allocs[3];

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(comm);
  tsl::AsyncValueRef<Communicator::Event> event = gpu_comm->GroupExecute(
      [num_updates_per_replica, num_ranks, input_offsets, send_sizes,
       output_offsets, recv_sizes, ragged_row_element_size, &buffers,
       &stream](GpuCommunicator* comm) -> absl::Status {
        PrimitiveType element_type = buffers[0].element_type;

        se::DeviceMemoryBase input_buffer = buffers[0].source_buffer;
        se::DeviceMemoryBase output_buffer = buffers[1].destination_buffer;

        for (int64_t i = 0; i < num_updates_per_replica; ++i) {
          for (int peer = 0; peer < num_ranks; ++peer) {
            int64_t idx = peer * num_updates_per_replica + i;
            se::DeviceMemoryBase send_slice = GpuCollectives::Slice(
                input_buffer, element_type,
                input_offsets[idx] * ragged_row_element_size,
                send_sizes[idx] * ragged_row_element_size);

            se::DeviceMemoryBase recv_slice = GpuCollectives::Slice(
                output_buffer, element_type,
                output_offsets[idx] * ragged_row_element_size,
                recv_sizes[idx] * ragged_row_element_size);

            TF_RETURN_IF_ERROR(
                comm->LaunchSend(send_slice, element_type,
                                 send_sizes[idx] * ragged_row_element_size,
                                 RankId(peer), GpuCollectives::On(stream)));

            TF_RETURN_IF_ERROR(
                comm->LaunchRecv(recv_slice, element_type,
                                 recv_sizes[idx] * ragged_row_element_size,
                                 RankId(peer), GpuCollectives::On(stream)));
          }
        }

        return absl::OkStatus();
      });
  tsl::BlockUntilReady(event);
  if (event.IsError()) {
    return event.GetError();
  }
  return absl::OkStatus();
}

// Performs synchronization of all participating devices with CUDA events.
absl::Status SynchronizeWithCudaEvents(absl::string_view name,
                                       const GpuCliqueKey& clique_key,
                                       se::Stream& stream, RankId rank,
                                       absl::Span<se::Event* const> events) {
  se::Event* local_event = events[rank.value()];

  // Record that this device has reached the synchronization point. We do this
  // before the rendezvous to make sure that RecordEvent is called before
  // WaitFor on another stream.
  TF_RETURN_IF_ERROR(stream.RecordEvent(local_event));

  std::string rendezvous_key =
      absl::StrFormat("%s ragged-all-to-all for rank %d, clique %s", name,
                      rank.value(), clique_key.ToString());

  // Do a rendezvous to make sure that all host threads have added a command to
  // record the start event on the stream.
  TF_RETURN_IF_ERROR(Rendezvous(rendezvous_key, clique_key,
                                clique_key.num_local_participants()));

  // Wait for all devices to reach the event. This indicates that all output
  // buffers are ready for transfer.
  for (auto& event : events) {
    TF_RETURN_IF_ERROR(stream.WaitFor(event));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RaggedAllToAllStartThunk::RunMemCpyRaggedAllToAll(
    const GpuCliqueKey& clique_key, se::Stream& stream,
    const StreamState& state, absl::Span<DeviceBufferPair const> buffers,
    absl::Span<int64_t* const> ragged_metadata_allocs) {
  int device_ordinal = stream.parent()->device_ordinal();
  const RankId& rank = state.rank;
  const int64_t num_ranks = clique_key.num_local_participants();

  VLOG(3) << "[" << device_ordinal << "] Performing mem-copy-ragged-all-to-all";

  PrimitiveType element_type = buffers[0].element_type;

  se::DeviceMemoryBase input_buffer = buffers[0].source_buffer;

  TF_RETURN_IF_ERROR(
      LoadRaggedTensorMetadata(stream, buffers, ragged_metadata_allocs));

  const int64_t num_updates_per_replica = config_.num_total_updates / num_ranks;

  const int64_t* input_offsets = ragged_metadata_allocs[0];
  const int64_t* send_sizes = ragged_metadata_allocs[1];
  const int64_t* output_offsets = ragged_metadata_allocs[2];

  TF_RETURN_IF_ERROR(SynchronizeWithCudaEvents(
      "start memcpy", clique_key, stream, rank, state.start_events));

  // Transfer a slice of data to each peer's output buffer.
  for (int64_t i = 0; i < num_updates_per_replica; ++i) {
    for (int peer = 0; peer < num_ranks; ++peer) {
      int64_t idx = peer * num_updates_per_replica + i;
      se::DeviceMemoryBase send_slice =
          GpuCollectives::Slice(input_buffer, element_type,
                                input_offsets[idx] * config_.num_row_elements,
                                send_sizes[idx] * config_.num_row_elements);
      se::DeviceMemoryBase dst_slice =
          GpuCollectives::Slice(state.output_buffers[peer], element_type,
                                output_offsets[idx] * config_.num_row_elements,
                                send_sizes[idx] * config_.num_row_elements);
      TF_RETURN_IF_ERROR(
          stream.MemcpyD2D(&dst_slice, send_slice, send_slice.size()));
    }
  }

  return SynchronizeWithCudaEvents("finish memcpy", clique_key, stream, rank,
                                   state.end_events);
}

absl::Status RaggedAllToAllStartThunk::RunOneShotRaggedAllToAll(
    const GpuCliqueKey& clique_key, se::Stream& stream,
    const StreamState& state, absl::Span<DeviceBufferPair const> buffers) {
  int device_ordinal = stream.parent()->device_ordinal();
  const RankId& rank = state.rank;

  const int64_t num_ranks = clique_key.num_local_participants();

  VLOG(3) << "[" << device_ordinal
          << "] Performing one-shot ragged-all-to-all rank: " << rank.value();

  PrimitiveType element_type = buffers[0].element_type;

  se::DeviceMemoryBase input_buffer = buffers[0].source_buffer;

  TF_RETURN_IF_ERROR(SynchronizeWithCudaEvents(
      "start one-shot", clique_key, stream, rank, state.start_events));

  const int64_t num_updates_per_replica = config_.num_total_updates / num_ranks;

  TF_RETURN_IF_ERROR(RunRaggedAllToAllKernel(
      &stream, element_type, input_buffer, state.output_buffers,
      buffers[2].source_buffer, buffers[3].source_buffer,
      buffers[4].source_buffer, num_ranks, num_updates_per_replica,
      config_.num_input_rows, config_.num_row_elements));

  return SynchronizeWithCudaEvents("finish one-shot", clique_key, stream, rank,
                                   state.end_events);
}

RaggedAllToAllStartThunk::RaggedAllToAllStartThunk(
    ThunkInfo thunk_info, const HloRaggedAllToAllInstruction* instr,
    std::vector<CollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : CollectiveThunk(Thunk::kRaggedAllToAllStart, thunk_info,
                      IsGPUSyncCollective(*instr),
                      AsyncStreamKind::kCollective),
      config_(GetRaggedAllToAllConfig(instr)),
      buffers_(std::move(buffers)),
      p2p_memcpy_enabled_(p2p_memcpy_enabled),
      one_shot_kernel_enabled_(
          instr->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel()) {
  CHECK_EQ(config_.config.operand_count, buffers_.size());
}

/*static*/ absl::Status RaggedAllToAllStartThunk::CheckImplementable(
    const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
    int64_t partition_count) {
  auto status = [&instr]() -> absl::Status {
    for (HloInstruction* operand : instr->operands()) {
      Shape shape = operand->shape();
      TF_RETURN_IF_ERROR(IsValidOperand(shape, Thunk::kRaggedAllToAll));
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
  return AddOpDescription<RaggedAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode RaggedAllToAllStartThunk::GetGroupMode(
    const HloRaggedAllToAllInstruction* instr) {
  return GetRaggedAllToAllConfig(instr).config.group_mode;
}

absl::Status RaggedAllToAllStartThunk::Initialize(
    const InitializeParams& params) {
  TF_RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;

  se::StreamExecutor* executor = params.executor;

  TF_ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, config_.config));
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);

  StreamState* state = nullptr;
  {
    absl::MutexLock lock(&mutex_);

    // If the stream state already exists, it means that the thunk has been
    // initialized for this executor.
    if (per_stream_states_.contains(executor)) {
      return absl::OkStatus();
    }

    auto [map_iter, was_inserted] = per_stream_states_.emplace(
        executor, std::make_unique<StreamState>(executor->device_ordinal(),
                                                rank.value()));
    state = map_iter->second.get();
  }

  // Allocate temp buffers in the host memory to load the sizes and offsets of
  // ragged tensors from device memory.
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> alloc,
                        executor->HostMemoryAllocate(config_.num_total_updates *
                                                     sizeof(int64_t)));
    state->host_buffer_allocs.push_back(std::move(alloc));
  }

  state->output_offsets_device_buffer = se::DeviceMemoryHandle{
      executor,
      executor->Allocate(config_.num_total_updates * sizeof(int64_t))};

  if (state->output_offsets_device_buffer.memory().is_null()) {
    return absl::InternalError("Failed to allocate output offsets buffer.");
  }

  state->local_output_buffer = params.buffer_allocations->GetDeviceAddress(
      buffers_[1].destination_buffer);

  if (is_local()) {
    TF_ASSIGN_OR_RETURN(state->local_start_event, executor->CreateEvent());
    TF_ASSIGN_OR_RETURN(state->local_end_event, executor->CreateEvent());
  }

  auto completion_fn = [](absl::Span<const StreamState* const> states)
      -> std::vector<const StreamState*> {
    std::vector<const StreamState*> copy(states.begin(), states.end());
    // Sort by rank for stable order.
    absl::c_sort(copy,
                 [](const StreamState* const a, const StreamState* const b) {
                   return a->rank < b->rank;
                 });
    return copy;
  };
  std::string rendezvous_key =
      absl::StrFormat("Initializing ragged-all-to-all for device %d, clique %s",
                      state->device_ordinal, clique_key.ToString());
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<std::vector<const StreamState*>> rendezvous_values,
      Rendezvous<std::vector<const StreamState*>>(
          /*name=*/rendezvous_key, /*key=*/clique_key,
          /*value=*/*state,
          /*num_threads=*/clique_key.num_local_participants(), completion_fn));

  state->output_buffers.reserve(rendezvous_values->size());
  state->start_events.reserve(rendezvous_values->size());
  state->end_events.reserve(rendezvous_values->size());
  for (auto* rendezvous_state : *rendezvous_values) {
    state->output_buffers.push_back(rendezvous_state->local_output_buffer);
    state->start_events.push_back(rendezvous_state->local_start_event.get());
    state->end_events.push_back(rendezvous_state->local_end_event.get());
  }

  return absl::OkStatus();
}

bool RaggedAllToAllStartThunk::is_local() const {
  CHECK_NE(device_count_, -1);
  for (const auto& replica_group : config_.config.replica_groups) {
    const int64_t node_id = replica_group.replica_ids().at(0) / device_count_;
    if (!absl::c_all_of(replica_group.replica_ids(),
                        [this, node_id](const int64_t rank) {
                          return rank / device_count_ == node_id;
                        })) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<bool> RaggedAllToAllStartThunk::RunCollective(
    const ExecuteParams& params, se::Stream& stream,
    CommunicatorHandle comm_handle) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(params, buffers_,
                             config_.config.operand_element_type));

  TF_ASSIGN_OR_RETURN(int32_t num_ranks, comm_handle.comm->NumRanks());

  TF_ASSIGN_OR_RETURN(
      bool peer_access_enabled,
      params.collective_cliques->peer_access_enabled(comm_handle.clique_key));

  StreamState* state = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    state = per_stream_states_[stream.parent()].get();
  }

  bool should_use_one_shot_kernel =
      is_local() && one_shot_kernel_enabled_ && peer_access_enabled &&
      IsRaggedAllToAllKernelSupported(num_ranks,
                                      device_buffers[0].element_type);

  if (should_use_one_shot_kernel) {
    TF_RETURN_IF_ERROR(RunOneShotRaggedAllToAll(comm_handle.clique_key, stream,
                                                *state, device_buffers));
    return false;
  }

  // Get buffer allocs to load sizes and offsets of ragged tensors from device
  // memory.
  absl::InlinedVector<int64_t*, 8> ragged_metadata_allocs;
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    ragged_metadata_allocs.push_back(
        reinterpret_cast<int64_t*>(state->host_buffer_allocs[i]->opaque()));
  }

  if (should_use_memcpy()) {
    TF_RETURN_IF_ERROR(RunMemCpyRaggedAllToAll(comm_handle.clique_key, stream,
                                               *state, device_buffers,
                                               ragged_metadata_allocs));
    return false;
  }

  TF_RETURN_IF_ERROR(RunRaggedAllToAll(
      config_.num_row_elements, config_.num_total_updates, device_buffers,
      stream, comm_handle.comm, ragged_metadata_allocs,
      state->output_offsets_device_buffer.memory(),
      config_.config.use_symmetric_buffer));
  return true;
}

}  // namespace gpu
}  // namespace xla
