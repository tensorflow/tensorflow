/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/collective_thunk.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/rendezvous.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

static constexpr int64_t kCollectiveMemorySpaceColor = 1;

bool IsTypeSupportedBy(PrimitiveType element_type, Thunk::Kind reduction_op) {
  switch (element_type) {
    case S8:
    case PRED:
    case U8:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
    case BF16:
    case C64:
    case C128:
      return true;
    case S16:
    case U16:
      // 16-bit integer reductions are not directly supported by NCCL and cannot
      // be implicitly converted into other 16-bit types like ncclFloat16 as
      // they involve actual computation and not just data movement.
    case F8E5M2:
    case F8E4M3FN:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      return !IsReductionCollective(reduction_op);
    default:
      return false;
  }
}

int64_t GetNumLocalParticipants(
    const CollectiveParams& params,
    const std::vector<GlobalDeviceId>& participants) {
  if (!params.global_device_id_map) {
    return participants.size();
  }

  std::vector<GlobalDeviceId> local_devices;
  local_devices.reserve(params.global_device_id_map->size());
  for (const auto& entry : *params.global_device_id_map) {
    local_devices.push_back(entry.second);
  }

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(local_devices, device_id);
  });
}

}  // namespace

// Returns if the collective communication operation is degenerate because all
// the groups formed by the operation are singleton. A given op can be
// degenerate under several conditions, corresponding to the modes supported
// in GetParticipatingDevices().
//   1. no channel id, use_global_device_ids = false:
//         degenerate if replica_groups are singleton, or groups empty and
//         replica_count == 1.
//   2. channel_id is set, use_global_device_ids = false:
//         degenerate if replica_groups are singleton and num_partitions == 1,
//         or groups empty and num_replicas == 1 && num_partitions == 1.
//   3. channel_id is set, use_global_device_ids = true (flattened-ids):
//         degenerate if replica_groups are singleton (groups cannot be empty).
//   4. no channel_id, no use_global_device_ids:
//         identical to 1.
//   5. channel_id is set, no use_global_device_ids:
//         degenerate if replica_groups are singleton or group emty and
//         num_partitions == 1 (since replica groups contain partition ids).
//
bool CollectiveConfig::IsDegenerate(int64_t replica_count,
                                    int64_t partition_count) const {
  bool groups_empty = replica_groups.empty();

  // check if all replica_groups are singleton. If not, then the operation is
  // not degenerate.
  bool all_groups_singleton =
      !groups_empty &&
      absl::c_all_of(replica_groups, [](const ReplicaGroup& group) {
        return group.replica_ids_size() == 1;
      });

  switch (group_mode) {
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA:
      return all_groups_singleton || (groups_empty && replica_count == 1);
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_PARTITION:
      return all_groups_singleton || (groups_empty && partition_count == 1);
    case CollectiveOpGroupMode::
        COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION:
      return (all_groups_singleton && partition_count == 1) ||
             (groups_empty && replica_count == 1 && partition_count == 1);
    case CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_FLATTENED_ID:
      CHECK(!groups_empty)
          << "replica groups cannot be empty if use_global_device_ids = true";
      return all_groups_singleton;
    default:
      CHECK(0) << "Invalid collective op mode";
      return false;
  }
}

CollectiveConfig GetCollectiveConfig(
    const HloInstruction* hlo, std::optional<bool> use_global_device_ids) {
  CollectiveConfig config;
  config.operand_element_type.reserve(hlo->operands().size());
  for (const HloInstruction* operand : hlo->operands()) {
    config.operand_element_type.push_back(operand->shape().element_type());
  }
  config.replica_groups = hlo->replica_groups();

  config.group_mode = GetCollectiveOpGroupMode(hlo->channel_id().has_value(),
                                               use_global_device_ids)
                          .value();

  config.use_symmetric_buffer =
      hlo->GetModule() &&
      hlo->GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_enable_nccl_symmetric_buffers();
  return config;
}

CollectiveThunk::CollectiveThunk(Kind kind, ThunkInfo thunk_info, bool is_sync,
                                 AsyncStreamKind stream_kind)
    : Thunk(kind, thunk_info),
      stream_kind_(stream_kind),
      async_events_(is_sync ? nullptr : std::make_shared<AsyncEvents>()) {}

absl::StatusOr<GpuCliqueKey> GetGpuCliqueKey(
    GpuCollectives* collectives, const CollectiveParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, AsyncStreamKind stream_kind,
    bool use_nccl) {
  GlobalDeviceId global_device_id = params.global_device_id;

  if (params.device_assn == nullptr) {
    return InvalidArgument(
        "Device assignment is null, but must be specified when running a "
        "collective thunk. If running multi-device HLO , make sure you're not "
        "using a tool designed for only one device like run_hlo_module.");
  }

  // Get the list of all devices that are participating in the collective
  // operation.
  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              replica_groups, group_mode));

  // Get grouping of participating devices.
  std::vector<std::vector<GlobalDeviceId>> participant_groups;
  if (use_nccl) {
    // If splitting is enabled, participating groups must match in order for a
    // clique to be reused from the cache. We can ignore the participating
    // groups otherwise.
    static const bool enable_nccl_comm_splitting =
        xla::GetDebugOptionsFromFlags().xla_gpu_enable_nccl_comm_splitting();
    if (enable_nccl_comm_splitting) {
      TF_ASSIGN_OR_RETURN(participant_groups,
                          GetParticipatingDevicesGroups(
                              *params.device_assn, replica_groups, group_mode));
    }

    if (collectives->IsGlobalConfig() &&
        (participants.size() != params.device_assn->replica_count())) {
      return InvalidArgument(
          "Partial replica groups are not allowed when using NCCL_COMM_ID "
          "environment configuration.");
    }
  }

  // Remove trivial group that contains all participants, as we do not want to
  // create two sets of communicator handles for these cases.
  if (participant_groups.size() == 1 && participant_groups[0] == participants) {
    participant_groups.clear();
  }

  int64_t num_local_participants =
      GetNumLocalParticipants(params, participants);

  absl::flat_hash_set<IncarnationId> unique_incarnations;
  if (params.incarnations) {
    for (GlobalDeviceId id : participants) {
      auto it = params.incarnations->find(id);
      if (it == params.incarnations->end()) {
        return FailedPrecondition("Incarnation for device %d not found",
                                  id.value());
      }
      unique_incarnations.insert(it->second);
    }
  }
  std::vector<IncarnationId> incarnations(unique_incarnations.begin(),
                                          unique_incarnations.end());
  absl::c_sort(incarnations);

  return GpuCliqueKey(std::move(participants), num_local_participants,
                      xla::gpu::IsP2PStreamKind(stream_kind),
                      std::move(participant_groups), GlobalDeviceId(-1),
                      incarnations);
}

absl::StatusOr<GpuCliqueKey> GetCollectiveGpuCliqueKey(
    const CollectiveParams& params, const CollectiveConfig& collective_config,
    bool use_nccl) {
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives,
                      CollectiveThunk::GetGpuCollectives(params));
  return GetGpuCliqueKey(collectives, params, collective_config.replica_groups,
                         collective_config.group_mode,
                         AsyncStreamKind::ASYNC_STREAM_KIND_COLLECTIVE,
                         use_nccl);
}

absl::StatusOr<CommunicatorHandle> GetComm(
    GpuCollectives* collectives, const CollectiveParams& params,
    const CollectiveCliques& collective_cliques,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, AsyncStreamKind stream_kind) {
  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(collectives, params, replica_groups,
                                      group_mode, stream_kind));

  std::optional<RankId> rank = clique_key.rank(params.global_device_id);
  TF_ASSIGN_OR_RETURN(Communicator * comm,
                      collective_cliques.GetComm(clique_key, *rank));

  return CommunicatorHandle(comm, std::move(clique_key));
}

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const Thunk::ExecuteParams& params,
    const std::vector<CollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types) {
  return ConvertToDeviceBuffers(params.buffer_allocations, buffers,
                                element_types);
}

absl::StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const BufferAllocations* buffer_allocations,
    const std::vector<CollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types) {
  if (buffers.size() != element_types.size()) {
    return FailedPrecondition("Mismatch in operand buffer counts.");
  }

  std::vector<DeviceBufferPair> device_buffers;
  device_buffers.reserve(buffers.size());
  for (int i = 0; i < buffers.size(); ++i) {
    device_buffers.emplace_back(DeviceBufferPair{
        element_types[i], buffers[i].element_count,
        buffer_allocations->GetDeviceAddress(buffers[i].source_buffer),
        buffer_allocations->GetDeviceAddress(buffers[i].destination_buffer),
        buffers[i].source_memory_space, buffers[i].destination_memory_space});
  }
  return device_buffers;
}

absl::Status MaybeRegisterBuffer(se::StreamExecutor* executor,
                                 const se::DeviceMemoryBase& buffer,
                                 Communicator* comm,
                                 bool use_symmetric_buffer) {
  TF_ASSIGN_OR_RETURN(auto range, executor->GetMemoryRange(buffer));
  VLOG(1) << "[" << executor->device_ordinal() << "] "
          << "Registering range: " << range.opaque()
          << " with size: " << range.size()
          << " for buffer: " << buffer.opaque()
          << " with size: " << buffer.size()
          << " is symmetric: " << (use_symmetric_buffer ? "true" : "false");
  // If the collective memory buffer is a slice of a larger preallocated buffer,
  // we need to register the entire preallocated buffer once.
  return comm->RegisterBufferOnce(range, executor->device_ordinal(),
                                  use_symmetric_buffer);
}

absl::Status MaybeRegisterBuffers(se::StreamExecutor* executor,
                                  const std::vector<DeviceBufferPair>& buffers,
                                  Communicator* comm,
                                  bool use_symmetric_buffer) {
  for (int i = 0; i < buffers.size(); ++i) {
    if (buffers[i].source_memory_space == kCollectiveMemorySpaceColor) {
      TF_RETURN_IF_ERROR(MaybeRegisterBuffer(executor, buffers[i].source_buffer,
                                             comm, use_symmetric_buffer));
    }
    if (buffers[i].destination_memory_space == kCollectiveMemorySpaceColor) {
      TF_RETURN_IF_ERROR(MaybeRegisterBuffer(
          executor, buffers[i].destination_buffer, comm, use_symmetric_buffer));
    }
  }
  return absl::OkStatus();
}

absl::Status CollectiveThunk::AsyncEvents::Initialize(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(mu_);
  if (events_.contains(executor)) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(auto event, executor->CreateEvent());

  events_.try_emplace(executor, std::move(event));
  return absl::OkStatus();
}

absl::StatusOr<se::Event*> CollectiveThunk::AsyncEvents::GetEvent(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(mu_);

  auto event = events_.find(executor);
  if (event == events_.end()) {
    return absl::InternalError(
        "Collective operation async completion event not initialized");
  }

  return event->second.get();
}

absl::Status CollectiveThunk::Prepare(const PrepareParams& params) {
  TF_RET_CHECK(params.collective_params != nullptr);
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(collectives, *params.collective_params,
                      config().replica_groups, config().group_mode,
                      GetAsyncStreamKind()));
  return params.clique_requests->RequestClique(clique_key);
}

absl::Status CollectiveThunk::Initialize(const InitializeParams& params) {
  if (async_events_) {
    TF_RETURN_IF_ERROR(async_events_->Initialize(params.executor));
  }
  return absl::OkStatus();
}

absl::Status CollectiveThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(1) << absl::StreamFormat(
      "[%d] Starting %s %s.", params.stream->parent()->device_ordinal(),
      IsAsync() ? "async" : "sync", Thunk::KindToString(kind()));
  AsyncStreamKind stream_kind = GetAsyncStreamKind();
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *params.collective_params,
              *params.collective_cliques, config().replica_groups,
              config().group_mode, stream_kind));
  se::StreamExecutor* executor = params.stream->parent();
  int64_t async_stream_idx = static_cast<int64_t>(stream_kind);

  bool is_first_rendezvous_needed = false;
  if (IsAsync()) {
    // Launch collective operation on an async stream.
    se::Stream& async_stream =
        *params.collective_params->async_streams.at(async_stream_idx);

    // Wait for main compute stream to make sure all buffers are ready.
    TF_RETURN_IF_ERROR(async_stream.WaitFor(params.stream));

    TF_ASSIGN_OR_RETURN(is_first_rendezvous_needed,
                        RunCollective(params, async_stream, comm_handle));

    // Record collective operation completion.
    TF_ASSIGN_OR_RETURN(se::Event * event, async_events_->GetEvent(executor));
    TF_RETURN_IF_ERROR(async_stream.RecordEvent(event));

  } else {
    // Launch collective operation on a main stream.
    TF_ASSIGN_OR_RETURN(is_first_rendezvous_needed,
                        RunCollective(params, *params.stream, comm_handle));
  }

  // After a first execution of this instance of collective operation do a
  // rendezvous with other participants to make sure that all of them allocated
  // required state (internal to NCCL) and ready to continue. Going too far
  // ahead on one rank leads to deadlocks in NCCL.
  if (is_first_rendezvous_needed &&
      !first_call_rendezvous_flag_.IsCompleted()) {
    GpuCliqueKey clique_key = comm_handle.clique_key;
    size_t num_local_participants = clique_key.num_local_participants();

    auto global_device_id = params.collective_params->global_device_id;
    RankId rank = clique_key.rank(global_device_id).value_or(RankId(-1));
    VLOG(1) << "[" << global_device_id.value()
            << "] Do a rendezvous after a first call to "
            << Thunk::KindToString(kind())
            << "; run_id=" << params.collective_params->run_id.ToInt()
            << "; num_local_participants=" << num_local_participants
            << "; rank=" << rank.value()
            << "; clique_key=" << clique_key.ToString();

    auto rendezvous_key = FirstCallRendezvousKey{std::move(clique_key)};
    auto rendezvous_name = absl::StrFormat(
        "first call to collective operation: kind=%s; run_id=%ld",
        Thunk::KindToString(kind()), params.collective_params->run_id.ToInt());

    const xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();

    TF_RETURN_IF_ERROR(Rendezvous(
        first_call_rendezvous_flag_, rendezvous_name, rendezvous_key,
        num_local_participants,
        /*warn_stuck_timeout=*/
        absl::Seconds(
            debug_options
                .xla_gpu_first_collective_call_warn_stuck_timeout_seconds()),
        /*terminate_timeout=*/
        absl::Seconds(
            debug_options
                .xla_gpu_first_collective_call_terminate_timeout_seconds())

            ));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<Communicator*>> CollectiveThunk::GetCommunicators(
    const ExecuteParams& params) const {
  AsyncStreamKind stream_kind = GetAsyncStreamKind();
  TF_ASSIGN_OR_RETURN(GpuCollectives * collectives, GetGpuCollectives(params));
  TF_ASSIGN_OR_RETURN(
      CommunicatorHandle comm_handle,
      GetComm(collectives, *params.collective_params,
              *params.collective_cliques, config().replica_groups,
              config().group_mode, stream_kind));
  return std::vector<Communicator*>{comm_handle.comm};
}

std::string CollectiveThunk::GetDeviceString(
    const CollectiveParams& collective_params) {
  GlobalDeviceId global_device_id = collective_params.global_device_id;
  DeviceAssignment::LogicalID logical_id =
      collective_params.device_assn->LogicalIdForDevice(global_device_id)
          .value();
  return absl::StrFormat("(r%d, p%d) : GlobalID %d, ord %d",
                         logical_id.replica_id, logical_id.computation_id,
                         global_device_id.value(),
                         collective_params.local_device_ordinal);
}

std::optional<AsyncEventsUniqueId> CollectiveThunk::GetAsyncEventsUniqueId()
    const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

CollectiveDoneThunk::CollectiveDoneThunk(
    Thunk::Kind kind, ThunkInfo thunk_info,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events,
    AsyncStreamKind async_stream_kind)
    : Thunk(kind, std::move(thunk_info)),
      async_events_(async_events),
      stream_kind_(async_stream_kind) {}

absl::Status CollectiveDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(se::Event * event, async_events_->GetEvent(executor));
  return params.stream->WaitFor(event);
}

absl::Status IsValidOperand(Shape shape, Thunk::Kind reduction_op) {
  if (!shape.IsArray()) {
    return absl::AbortedError(
        absl::StrFormat("input is not a dense array: %s",
                        shape.ToString(/*print_layout=*/true)));
  }
  if (!IsTypeSupportedBy(shape.element_type(), reduction_op)) {
    return absl::AbortedError(absl::StrFormat(
        "element type %s not suppored by NCCL",
        primitive_util::LowercasePrimitiveTypeName(shape.element_type())));
  }
  return absl::OkStatus();
}

std::optional<AsyncEventsUniqueId> CollectiveDoneThunk::GetAsyncEventsUniqueId()
    const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}
}  // namespace xla::gpu
