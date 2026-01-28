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
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/primitive_util.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/rendezvous.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

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

CollectiveConfigProto CollectiveConfig::ToProto() const {
  CollectiveConfigProto proto;

  proto.mutable_operand_element_type()->Assign(operand_element_type.begin(),
                                               operand_element_type.end());
  proto.mutable_replica_groups()->Assign(replica_groups.begin(),
                                         replica_groups.end());

  proto.set_group_mode(group_mode);
  proto.set_use_symmetric_buffer(use_symmetric_buffer);

  return proto;
}

CollectiveConfig CollectiveConfig::FromProto(
    const CollectiveConfigProto& proto) {
  CollectiveConfig config;

  config.operand_element_type.reserve(proto.operand_element_type_size());
  for (int element_type : proto.operand_element_type()) {
    config.operand_element_type.push_back(
        static_cast<PrimitiveType>(element_type));
  }

  config.replica_groups.assign(proto.replica_groups().begin(),
                               proto.replica_groups().end());

  config.group_mode = proto.group_mode();
  config.use_symmetric_buffer = proto.use_symmetric_buffer();
  return config;
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
                                 bool is_p2p)
    : Thunk(kind, thunk_info),
      async_events_(is_sync ? nullptr : std::make_shared<AsyncEvents>()),
      is_p2p_(is_p2p) {}

CollectiveThunk::CollectiveThunk(Kind kind, ThunkInfo thunk_info,
                                 std::shared_ptr<AsyncEvents> async_events,
                                 bool is_p2p)
    : Thunk(kind, thunk_info), async_events_(async_events), is_p2p_(is_p2p) {}

absl::StatusOr<GpuCliqueKey> GetCollectiveGpuCliqueKey(
    const CollectiveParams& params, const CollectiveConfig& collective_config,
    bool include_participant_groups) {
  return GetGpuCliqueKey(params, collective_config.replica_groups,
                         collective_config.group_mode,
                         include_participant_groups);
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
        buffer_allocations->GetDeviceAddress(buffers[i].source_buffer.slice),
        buffer_allocations->GetDeviceAddress(
            buffers[i].destination_buffer.slice),
        buffers[i].source_memory_space, buffers[i].destination_memory_space});
  }
  return device_buffers;
}

absl::Status MaybeRegisterBuffer(se::StreamExecutor* executor,
                                 const se::DeviceAddressBase& buffer,
                                 Communicator* comm,
                                 bool use_symmetric_buffer) {
  ASSIGN_OR_RETURN(auto range, executor->GetMemoryRange(buffer));
  XLA_VLOG_DEVICE(1, executor->device_ordinal())
      << "Registering range: " << range.opaque()
      << " with size: " << range.size() << " for buffer: " << buffer.opaque()
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
      RETURN_IF_ERROR(MaybeRegisterBuffer(executor, buffers[i].source_buffer,
                                          comm, use_symmetric_buffer));
    }
    if (buffers[i].destination_memory_space == kCollectiveMemorySpaceColor) {
      RETURN_IF_ERROR(MaybeRegisterBuffer(
          executor, buffers[i].destination_buffer, comm, use_symmetric_buffer));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<CollectiveBufferProto> CollectiveThunk::Buffer::ToProto() const {
  CollectiveBufferProto proto;
  proto.set_element_count(element_count);
  ASSIGN_OR_RETURN(*proto.mutable_source_buffer(), source_buffer.ToProto());
  ASSIGN_OR_RETURN(*proto.mutable_destination_buffer(),
                   destination_buffer.ToProto());
  proto.set_source_memory_space(source_memory_space);
  proto.set_destination_memory_space(destination_memory_space);
  return proto;
}

absl::StatusOr<CollectiveThunk::Buffer> CollectiveThunk::Buffer::FromProto(
    const CollectiveBufferProto& buffer_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  CollectiveThunk::Buffer res;
  res.element_count = buffer_proto.element_count();
  ASSIGN_OR_RETURN(
      res.source_buffer,
      ShapedSlice::FromProto(buffer_proto.source_buffer(), buffer_allocations));
  ASSIGN_OR_RETURN(res.destination_buffer,
                   ShapedSlice::FromProto(buffer_proto.destination_buffer(),
                                          buffer_allocations));
  res.source_memory_space = buffer_proto.source_memory_space();
  res.destination_memory_space = buffer_proto.destination_memory_space();
  return res;
}

absl::Status CollectiveThunk::AsyncEvents::Initialize(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(mu_);
  if (events_.contains(executor)) {
    return absl::OkStatus();
  }

  ASSIGN_OR_RETURN(auto event, executor->CreateEvent());

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
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, is_p2p_));
  return params.collective_clique_requests->RequestClique(clique_key);
}

absl::Status CollectiveThunk::Initialize(const InitializeParams& params) {
  if (async_events_) {
    RETURN_IF_ERROR(async_events_->Initialize(params.executor));
  }
  return absl::OkStatus();
}

absl::Status CollectiveThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(1) << absl::StreamFormat(
      "[%d] Starting %s %s.", params.stream->parent()->device_ordinal(),
      IsAsync() ? "async" : "sync", Thunk::KindToString(kind()));

  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, is_p2p_));

  ASSIGN_OR_RETURN(Communicator * comm,
                   params.collective_cliques->GetComm(
                       clique_key, params.collective_params->global_device_id));
  DCHECK(comm) << "Failed to get communicator for collective operation";

  se::StreamExecutor* executor = params.stream->parent();
  int64_t async_stream_idx = Thunk::execution_stream_id().value();

  bool is_first_rendezvous_needed = false;
  if (IsAsync()) {
    // Launch collective operation on an async stream.
    se::Stream& async_stream =
        *params.collective_params->async_streams.at(async_stream_idx);

    // Wait for main compute stream to make sure all buffers are ready.
    RETURN_IF_ERROR(async_stream.WaitFor(params.stream));

    ASSIGN_OR_RETURN(is_first_rendezvous_needed,
                     RunCollective(params, clique_key, async_stream, *comm));

    // Record collective operation completion.
    ASSIGN_OR_RETURN(se::Event * event, async_events_->GetEvent(executor));
    RETURN_IF_ERROR(async_stream.RecordEvent(event));
  } else {
    // Launch collective operation on a main stream.
    ASSIGN_OR_RETURN(is_first_rendezvous_needed,
                     RunCollective(params, clique_key, *params.stream, *comm));
  }

  // After a first execution of this instance of collective operation do a
  // rendezvous with other participants to make sure that all of them allocated
  // required state (internal to NCCL) and ready to continue. Going too far
  // ahead on one rank leads to deadlocks in NCCL.
  if (is_first_rendezvous_needed &&
      !first_call_rendezvous_flag_.IsCompleted()) {
    size_t num_local_participants = clique_key.num_local_participants();

    auto global_device_id = params.collective_params->global_device_id;
    RankId rank = clique_key.rank(global_device_id).value_or(RankId(-1));
    XLA_VLOG_DEVICE(1, global_device_id.value())
        << "Do a rendezvous after a first call to "
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

    RETURN_IF_ERROR(Rendezvous(
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
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, is_p2p_));
  ASSIGN_OR_RETURN(Communicator * comm,
                   params.collective_cliques->GetComm(
                       clique_key, params.collective_params->global_device_id));
  return std::vector<Communicator*>{comm};
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
                         collective_params.local_device_id.value());
}

std::optional<AsyncEventsUniqueId> CollectiveThunk::GetAsyncEventsUniqueId()
    const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

absl::StatusOr<CollectiveThunkProto> CollectiveThunk::ToCollectiveThunkProto()
    const {
  CollectiveThunkProto proto;

  std::optional<AsyncEventsUniqueId> async_events_id = GetAsyncEventsUniqueId();
  if (!async_events_id.has_value()) {
    return absl::FailedPreconditionError("AsyncEvents is not set.");
  }
  proto.set_async_events_unique_id(async_events_id->value());
  proto.set_thunk_kind(Thunk::KindToProto(kind()));

  return proto;
}

CollectiveDoneThunk::CollectiveDoneThunk(
    Thunk::Kind kind, ThunkInfo thunk_info,
    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events)
    : Thunk(kind, std::move(thunk_info)), async_events_(async_events) {}

absl::Status CollectiveDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::StreamExecutor* executor = params.stream->parent();
  ASSIGN_OR_RETURN(se::Event * event, async_events_->GetEvent(executor));
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

absl::StatusOr<ThunkProto> CollectiveDoneThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CollectiveDoneThunkProto* thunk_proto = proto.mutable_collective_done_thunk();

  std::optional<AsyncEventsUniqueId> async_events_id = GetAsyncEventsUniqueId();
  if (async_events_id.has_value()) {
    thunk_proto->set_async_events_unique_id(async_events_id->value());
  }
  thunk_proto->set_thunk_kind(Thunk::KindToProto(kind()));
  return proto;
}

absl::StatusOr<std::unique_ptr<CollectiveDoneThunk>>
CollectiveDoneThunk::FromProto(
    ThunkInfo thunk_info, const CollectiveDoneThunkProto& thunk_proto,
    CollectiveThunk::AsyncEventsMap& async_events_map) {
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

  ASSIGN_OR_RETURN(Thunk::Kind kind,
                   Thunk::KindFromProto(thunk_proto.thunk_kind()));
  return std::make_unique<CollectiveDoneThunk>(kind, std::move(thunk_info),
                                               async_events);
}

}  // namespace xla::gpu
