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
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/rendezvous.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "xla/tsl/platform/status_macros.h"

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
    RETURN_IF_ERROR(stream.Memcpy(ragged_metadata_allocs[i],
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
    const se::DeviceAddressBase& source_buffer, int64_t num_updates_per_replica,
    const se::DeviceAddressBase& destination_buffer, PrimitiveType element_type,
    se::Stream& stream, Communicator& comm) {
  ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute(
      [num_ranks, num_updates_per_replica, element_type, &source_buffer,
       &destination_buffer, &stream](GpuCommunicator* comm) -> absl::Status {
        for (int peer = 0; peer < num_ranks; ++peer) {
          int64_t offset = peer * num_updates_per_replica;
          se::DeviceAddressBase send_slice =
              GpuCollectives::Slice(source_buffer, element_type, offset,
                                    /*count=*/num_updates_per_replica);
          se::DeviceAddressBase recv_slice =
              GpuCollectives::Slice(destination_buffer, element_type, offset,
                                    /*count=*/num_updates_per_replica);
          RETURN_IF_ERROR(comm->LaunchSend(send_slice, element_type,
                                           /*count=*/num_updates_per_replica,
                                           RankId(peer),
                                           GpuCollectives::On(stream)));
          RETURN_IF_ERROR(comm->LaunchRecv(recv_slice, element_type,
                                           /*count=*/num_updates_per_replica,
                                           RankId(peer),
                                           GpuCollectives::On(stream)));
        }
        return absl::OkStatus();
      });
  RETURN_IF_ERROR(future.Await());
  return stream.BlockHostUntilDone();
}

absl::Status RunRaggedAllToAll(
    int64_t ragged_row_element_size, int64_t num_total_updates,
    const std::vector<DeviceBufferPair>& original_buffers, se::Stream& stream,
    Communicator& comm, absl::Span<int64_t* const> ragged_metadata_allocs,
    const se::DeviceAddressBase& output_offsets_device_buffer,
    bool use_symmetric_buffer) {
  int device_ordinal = stream.parent()->device_ordinal();
  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Performing ragged-all-to-all from device ordinal: " << device_ordinal;
  ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());

  std::vector<DeviceBufferPair> buffers = original_buffers;

  int64_t num_updates_per_replica = num_total_updates / num_ranks;

  // `output_offsets` of the RaggedAllToAll instruction are sharded in a way,
  // that `output_offset[i]` is an offset in the i-th peer output buffer. To
  // make it work for NCCL model with send/recv, we need to know offsets in the
  // local output buffer. To get the correct offsets we perform an AllToAll on
  // the output_offsets buffer.
  DeviceBufferPair& output_offsets_buffer_pair = buffers[4];
  RETURN_IF_ERROR(RunAllToAllOnIndexBuffer(
      output_offsets_buffer_pair.source_buffer, num_updates_per_replica,
      output_offsets_device_buffer, output_offsets_buffer_pair.element_type,
      stream, comm));
  output_offsets_buffer_pair.source_buffer = output_offsets_device_buffer;

  RETURN_IF_ERROR(
      LoadRaggedTensorMetadata(stream, buffers, ragged_metadata_allocs));

  const int64_t* input_offsets = ragged_metadata_allocs[0];
  const int64_t* send_sizes = ragged_metadata_allocs[1];
  const int64_t* output_offsets = ragged_metadata_allocs[2];
  const int64_t* recv_sizes = ragged_metadata_allocs[3];

  auto* gpu_comm = tsl::down_cast<GpuCommunicator*>(&comm);
  Future<> future = gpu_comm->GroupExecute(
      [num_updates_per_replica, num_ranks, input_offsets, send_sizes,
       output_offsets, recv_sizes, ragged_row_element_size, &buffers,
       &stream](GpuCommunicator* comm) -> absl::Status {
        PrimitiveType element_type = buffers[0].element_type;

        se::DeviceAddressBase input_buffer = buffers[0].source_buffer;
        se::DeviceAddressBase output_buffer = buffers[1].destination_buffer;

        for (int64_t i = 0; i < num_updates_per_replica; ++i) {
          for (int peer = 0; peer < num_ranks; ++peer) {
            int64_t idx = peer * num_updates_per_replica + i;
            se::DeviceAddressBase send_slice = GpuCollectives::Slice(
                input_buffer, element_type,
                input_offsets[idx] * ragged_row_element_size,
                send_sizes[idx] * ragged_row_element_size);

            se::DeviceAddressBase recv_slice = GpuCollectives::Slice(
                output_buffer, element_type,
                output_offsets[idx] * ragged_row_element_size,
                recv_sizes[idx] * ragged_row_element_size);

            RETURN_IF_ERROR(
                comm->LaunchSend(send_slice, element_type,
                                 send_sizes[idx] * ragged_row_element_size,
                                 RankId(peer), GpuCollectives::On(stream)));

            RETURN_IF_ERROR(
                comm->LaunchRecv(recv_slice, element_type,
                                 recv_sizes[idx] * ragged_row_element_size,
                                 RankId(peer), GpuCollectives::On(stream)));
          }
        }

        return absl::OkStatus();
      });
  return future.Await();
}

}  // namespace

RaggedAllToAllStartThunk::RaggedAllToAllStartThunk(
    ThunkInfo thunk_info, const HloRaggedAllToAllInstruction* instr,
    std::vector<CollectiveThunk::Buffer> buffers, bool p2p_memcpy_enabled)
    : RaggedAllToAllStartThunk(
          std::move(thunk_info), GetRaggedAllToAllConfig(instr),
          IsGPUSyncCollective(*instr)
              ? nullptr
              : std::make_shared<CollectiveThunk::AsyncEvents>(),
          std::move(buffers),
          instr->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel()) {}

RaggedAllToAllStartThunk::RaggedAllToAllStartThunk(
    ThunkInfo thunk_info, const RaggedAllToAllConfig& config,
    std::shared_ptr<AsyncEvents> async_events,
    std::vector<CollectiveThunk::Buffer> buffers, bool one_shot_kernel_enabled)
    : CollectiveThunk(Thunk::kRaggedAllToAllStart, thunk_info, async_events,
                      false),
      config_(config),
      buffers_(std::move(buffers)),
      one_shot_kernel_enabled_(one_shot_kernel_enabled) {
  CHECK_EQ(config_.config.operand_element_type.size(), buffers_.size());
}

// Executes the rendezvous before the kernel start.
// Inserts CUDA events into the stream to ensure that all devices have reached
// the start event before the kernel starts.
absl::Status RaggedAllToAllStartThunk::RendezvousBeforeKernelStart(
    const GpuCliqueKey& clique_key, se::Stream& stream,
    const StreamState& state) {
  int64_t num_ranks = clique_key.num_local_participants();
  const RankId& rank = state.rank;

  // Record that this device has started the memcpy ragged-all-to-all. We do
  // this before the rendezvous to make sure that RecordEvent is called before
  // WaitFor on another stream.
  RETURN_IF_ERROR(stream.RecordEvent(state.start_event.get()));

  std::string name =
      absl::StrFormat("start one-shot ragged-all-to-all for rank %d, clique %s",
                      rank.value(), clique_key.ToString());
  RETURN_IF_ERROR(Rendezvous(name, clique_key, num_ranks));

  // Wait for all devices to reach the start event. This indicates that all
  // output buffers are ready for transfer.
  for (const RendezvousValue& value : *state.rendezvous_values) {
    RETURN_IF_ERROR(stream.WaitFor(value.start_event));
  }

  return absl::OkStatus();
}

// Executes the rendezvous after the kernel finish. Waits for all devices to
// reach the end event.
absl::Status RaggedAllToAllStartThunk::RendezvousAfterKernelFinish(
    const GpuCliqueKey& clique_key, se::Stream& stream,
    const StreamState& state) {
  int64_t num_ranks = clique_key.num_local_participants();
  const RankId& rank = state.rank;

  // Record that this device has finished the memcpy ragged-all-to-all.
  RETURN_IF_ERROR(stream.RecordEvent(state.end_event.get()));

  // Do another rendezvous to make sure that we call RecordEvent for end_event
  // before WaitFor on another stream.
  std::string name = absl::StrFormat(
      "finish one-shot ragged-all-to-all for rank %d, clique %s", rank.value(),
      clique_key.ToString());
  RETURN_IF_ERROR(Rendezvous(name, clique_key, num_ranks));

  // Wait for all devices to reach the end event. This indicates that all
  // updates from other devices have arrived.
  for (const RendezvousValue& value : *state.rendezvous_values) {
    RETURN_IF_ERROR(stream.WaitFor(value.end_event));
  }

  return absl::OkStatus();
}

absl::Status RaggedAllToAllStartThunk::RunOneShotRaggedAllToAll(
    const GpuCliqueKey& clique_key, se::Stream& stream,
    const StreamState& state, absl::Span<DeviceBufferPair const> buffers) {
  int device_ordinal = stream.parent()->device_ordinal();
  const RankId& rank = state.rank;

  const int64_t num_ranks = clique_key.num_local_participants();

  XLA_VLOG_DEVICE(3, device_ordinal)
      << "Performing one-shot ragged-all-to-all rank: " << rank.value();

  PrimitiveType element_type = buffers[0].element_type;

  se::DeviceAddressBase input_buffer = buffers[0].source_buffer;

  RETURN_IF_ERROR(RendezvousBeforeKernelStart(clique_key, stream, state));

  const int64_t num_updates_per_replica = config_.num_total_updates / num_ranks;

  absl::InlinedVector<se::DeviceAddressBase, 4> output_ptrs;
  for (const RendezvousValue& value : *state.rendezvous_values) {
    output_ptrs.push_back(value.output_buffer);
  }

  RETURN_IF_ERROR(RunRaggedAllToAllKernel(
      &stream, element_type, input_buffer, output_ptrs,
      buffers[2].source_buffer, buffers[3].source_buffer,
      buffers[4].source_buffer, num_ranks, num_updates_per_replica,
      config_.num_input_rows, config_.num_row_elements));

  return RendezvousAfterKernelFinish(clique_key, stream, state);
}

/*static*/ absl::Status RaggedAllToAllStartThunk::CheckImplementable(
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
  return AddOpDescription<RaggedAllToAllStartThunk>(
      status(), instr, replica_count, partition_count);
}

/*static*/ CollectiveOpGroupMode RaggedAllToAllStartThunk::GetGroupMode(
    const HloRaggedAllToAllInstruction* instr) {
  return GetRaggedAllToAllConfig(instr).config.group_mode;
}

absl::Status RaggedAllToAllStartThunk::Initialize(
    const InitializeParams& params) {
  RETURN_IF_ERROR(CollectiveThunk::Initialize(params));
  device_count_ = params.local_device_count;

  ASSIGN_OR_RETURN(
      const GpuCliqueKey clique_key,
      GetCollectiveGpuCliqueKey(*params.collective_params, config_.config));
  const std::optional<RankId> rank =
      clique_key.rank(params.collective_params->global_device_id);
  int64_t num_ranks = clique_key.num_local_participants();

  se::StreamExecutor* executor = params.executor;

  StreamState* state = nullptr;
  {
    absl::MutexLock lock(mutex_);

    // If the stream state already exists, it means that the thunk has been
    // initialized for this executor.
    if (per_stream_states_.contains(executor)) {
      return absl::OkStatus();
    }

    auto [it, inserted] = per_stream_states_.emplace(
        executor, std::make_unique<StreamState>(executor->device_ordinal(),
                                                rank.value()));
    state = it->second.get();
  }

  // Allocate temp buffers in the host memory to load the sizes and offsets of
  // ragged tensors from device memory.
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> alloc,
                     executor->HostMemoryAllocate(config_.num_total_updates *
                                                  sizeof(int64_t)));
    state->host_buffer_allocs.push_back(std::move(alloc));
  }

  state->output_offsets_device_buffer = se::DeviceAddressHandle{
      executor,
      executor->Allocate(config_.num_total_updates * sizeof(int64_t))};

  if (state->output_offsets_device_buffer.memory().is_null()) {
    return absl::InternalError("Failed to allocate output offsets buffer.");
  }

  if (is_local() && clique_key.is_local()) {
    ASSIGN_OR_RETURN(state->start_event, executor->CreateEvent());
    ASSIGN_OR_RETURN(state->end_event, executor->CreateEvent());

    RendezvousValue rendezvous_value;
    rendezvous_value.rank = state->rank;
    rendezvous_value.output_buffer =
        params.buffer_allocations->GetDeviceAddress(
            buffers_[1].destination_buffer);
    rendezvous_value.start_event = state->start_event.get();
    rendezvous_value.end_event = state->end_event.get();

    std::string name =
        absl::StrFormat("initialize ragged-all-to-all for rank %d, clique %s",
                        rank->value(), clique_key.ToString());

    auto rendezvous_fn = [](absl::Span<const RendezvousValue* const> values) {
      std::vector<RendezvousValue> values_copy;
      for (const auto& value : values) {
        values_copy.push_back(*value);
      }
      // Sort to make sure that values are in the same order as the devices
      // are ordered in the communicator.
      absl::c_sort(values_copy);
      return values_copy;
    };

    ASSIGN_OR_RETURN(
        state->rendezvous_values,
        Rendezvous<std::vector<RendezvousValue>>(
            name, clique_key, rendezvous_value, num_ranks, rendezvous_fn,
            /*warn_stuck_timeout=*/absl::Seconds(20),
            /*terminate_timeout=*/absl::Seconds(40)));
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

absl::StatusOr<std::unique_ptr<RaggedAllToAllStartThunk>>
RaggedAllToAllStartThunk::FromProto(
    ThunkInfo thunk_info, const RaggedAllToAllStartThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    CollectiveThunk::AsyncEventsMap& async_events_map) {
  std::vector<CollectiveThunk::Buffer> buffers;
  buffers.reserve(thunk_proto.buffers_size());
  for (const CollectiveBufferProto& proto : thunk_proto.buffers()) {
    ASSIGN_OR_RETURN(
        CollectiveThunk::Buffer buffer,
        CollectiveThunk::Buffer::FromProto(proto, buffer_allocations));
    buffers.push_back(buffer);
  }

  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events;
  if (thunk_proto.has_async_events_unique_id()) {
    std::shared_ptr<CollectiveThunk::AsyncEvents>& events =
        async_events_map[AsyncEventsUniqueId{
            thunk_proto.async_events_unique_id()}];
    if (!events) {
      events = std::make_shared<CollectiveThunk::AsyncEvents>();
    }
    async_events = events;
  }

  CollectiveConfig config =
      CollectiveConfig::FromProto(thunk_proto.collective_config());

  return std::make_unique<RaggedAllToAllStartThunk>(
      std::move(thunk_info),
      RaggedAllToAllConfig{config, thunk_proto.num_total_updates(),
                           thunk_proto.num_input_rows(),
                           thunk_proto.num_row_elements()},
      async_events, std::move(buffers), thunk_proto.one_shot_kernel_enabled());
}

absl::StatusOr<ThunkProto> RaggedAllToAllStartThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  RaggedAllToAllStartThunkProto* thunk_proto =
      proto.mutable_ragged_all_to_all_start_thunk();

  std::optional<AsyncEventsUniqueId> async_events_id = GetAsyncEventsUniqueId();
  if (async_events_id.has_value()) {
    thunk_proto->set_async_events_unique_id(async_events_id->value());
  }

  for (const Buffer& buffer : buffers_) {
    ASSIGN_OR_RETURN(*thunk_proto->add_buffers(), buffer.ToProto());
  }

  *thunk_proto->mutable_collective_config() = config_.config.ToProto();

  thunk_proto->set_num_total_updates(config_.num_total_updates);
  thunk_proto->set_num_input_rows(config_.num_input_rows);
  thunk_proto->set_num_row_elements(config_.num_row_elements);
  thunk_proto->set_one_shot_kernel_enabled(one_shot_kernel_enabled_);

  return proto;
}

absl::StatusOr<bool> RaggedAllToAllStartThunk::RunCollective(
    const ExecuteParams& params, const GpuCliqueKey& clique_key,
    se::Stream& stream, Communicator& comm) {
  ASSIGN_OR_RETURN(std::vector<DeviceBufferPair> device_buffers,
                   ConvertToDeviceBuffers(params, buffers_,
                                          config_.config.operand_element_type));

  ASSIGN_OR_RETURN(int32_t num_ranks, comm.NumRanks());

  ASSIGN_OR_RETURN(bool peer_access_enabled,
                   params.collective_cliques->peer_access_enabled(clique_key));

  StreamState* state = nullptr;
  {
    absl::MutexLock lock(mutex_);
    state = per_stream_states_[stream.parent()].get();
  }

  bool should_use_one_shot_kernel =
      is_local() && one_shot_kernel_enabled_ && peer_access_enabled &&
      IsRaggedAllToAllKernelSupported(num_ranks,
                                      device_buffers[0].element_type);

  if (should_use_one_shot_kernel) {
    RETURN_IF_ERROR(
        RunOneShotRaggedAllToAll(clique_key, stream, *state, device_buffers));
    return false;
  }

  // Get buffer allocs to load sizes and offsets of ragged tensors from device
  // memory.
  absl::InlinedVector<int64_t*, 8> ragged_metadata_allocs;
  ragged_metadata_allocs.reserve(kNumRaggedMetadataOperands);
  for (int64_t i = 0; i < kNumRaggedMetadataOperands; ++i) {
    ragged_metadata_allocs.push_back(reinterpret_cast<int64_t*>(
        state->host_buffer_allocs[i]->address().opaque()));
  }

  RETURN_IF_ERROR(
      RunRaggedAllToAll(config_.num_row_elements, config_.num_total_updates,
                        device_buffers, stream, comm, ragged_metadata_allocs,
                        state->output_offsets_device_buffer.memory(),
                        config_.config.use_symmetric_buffer));
  return true;
}

}  // namespace gpu
}  // namespace xla
