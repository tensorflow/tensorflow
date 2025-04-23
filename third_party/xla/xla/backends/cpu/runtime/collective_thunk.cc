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

#include "xla/backends/cpu/runtime/collective_thunk.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/cpu_cliques.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::string_view CollectiveThunk::CollectiveKindToString(CollectiveKind kind) {
  switch (kind) {
    case CollectiveKind::kAllGather:
      return "all-gather";
    case CollectiveKind::kAllReduce:
      return "all-reduce";
    case CollectiveKind::kAllToAll:
      return "all-to-all";
    case CollectiveKind::kCollectivePermute:
      return "collective-permute";
    case CollectiveKind::kReduceScatter:
      return "reduce-scatter";
  }
}

std::ostream& operator<<(std::ostream& os,
                         CollectiveThunk::CollectiveKind kind) {
  return os << CollectiveThunk::CollectiveKindToString(kind);
}

CollectiveThunk::CollectiveThunk(CollectiveKind collective_kind,
                                 Thunk::Info info, OpParams op_params,
                                 OpBuffers op_buffers, OpResources op_resources)
    : Thunk(Thunk::Kind::kCollective, info),
      op_params_(std::move(op_params)),
      op_buffers_(std::move(op_buffers)),
      op_resources_(std::move(op_resources)),
      collective_kind_(collective_kind) {}

Thunk::BufferUses CollectiveThunk::buffer_uses() const {
  BufferUses uses;
  uses.reserve(source_buffers().size() + destination_buffers().size());
  for (auto& source_buffer : source_buffers()) {
    uses.push_back(BufferUse::Read(source_buffer));
  }
  for (auto& destination_buffer : destination_buffers()) {
    uses.push_back(BufferUse::Write(destination_buffer));
  }
  return uses;
}

Thunk::ResourceUses CollectiveThunk::resource_uses() const {
  return {ResourceUse::Write(op_resources_.communicator_resource)};
}

bool CollectiveThunk::IsDataTypeSupportedByCollectiveReduce(
    PrimitiveType datatype) {
  switch (datatype) {
    case PRED:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case S64:
    case U64:
    case F16:
    case F32:
    case F64:
    case C64:
    case C128:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<CollectiveThunk::OpDeviceMemory>
CollectiveThunk::GetOpDeviceMemory(const ExecuteParams& params) {
  size_t num_srcs = source_buffers().size();
  size_t num_dsts = destination_buffers().size();
  DCHECK_EQ(num_srcs, num_dsts) << "Number of src and dst buffers must match";

  absl::InlinedVector<se::DeviceMemoryBase, 4> source_data(num_srcs);
  for (int i = 0; i < num_srcs; ++i) {
    TF_ASSIGN_OR_RETURN(
        source_data[i],
        params.buffer_allocations->GetDeviceAddress(source_buffer(i)));
  }

  absl::InlinedVector<se::DeviceMemoryBase, 4> destination_data(num_dsts);
  for (int i = 0; i < num_dsts; ++i) {
    TF_ASSIGN_OR_RETURN(
        destination_data[i],
        params.buffer_allocations->GetDeviceAddress(destination_buffer(i)));
  }

  return OpDeviceMemory{std::move(source_data), std::move(destination_data)};
}

absl::Duration CollectiveThunk::DefaultCollectiveTimeout() {
  return absl::Minutes(30);
}

absl::StatusOr<RendezvousKey> CollectiveThunk::GetRendezvousKey(
    const Thunk::CollectiveExecuteParams& params) {
  TF_RET_CHECK(params.device_assignment) << "Device assignment is null";

  const DeviceAssignment& device_assignment = *params.device_assignment;
  RendezvousKey::CollectiveOpKind op_kind = op_params_.has_channel_id
                                                ? RendezvousKey::kCrossModule
                                                : RendezvousKey::kCrossReplica;

  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode group_mode,
      GetCollectiveOpGroupMode(op_params_.has_channel_id,
                               op_params_.use_global_device_ids));

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participating_devices,
      GetParticipatingDevices(params.global_device_id, device_assignment,
                              op_params_.group, group_mode));

  int num_local_participants = participating_devices.size();
  return RendezvousKey{params.run_id, std::move(participating_devices),
                       num_local_participants, op_kind, op_params_.op_id};
}

absl::StatusOr<int32_t> CollectiveThunk::RankInGlobalDevices(
    const RendezvousKey& key, GlobalDeviceId device) {
  auto it = absl::c_find(key.global_devices, device);
  if (it == key.global_devices.end()) {
    return InvalidArgument(
        "Device %d not present in global devices %s.", device.value(),
        absl::StrJoin(key.global_devices, ", ",
                      [](std::string* out, GlobalDeviceId id) {
                        absl::StrAppend(out, id.value());
                      }));
  }
  return std::distance(key.global_devices.begin(), it);
}

tsl::AsyncValueRef<CollectiveThunk::ExecuteEvent>
CollectiveThunk::ExecuteWithCommunicator(
    const Thunk::CollectiveExecuteParams* params, Callback callback) {
  // Check that we have access to collectives interface implementation and
  // parameters that define our "position" in a collective clique.
  TF_RET_CHECK(params)
      << "Collective parameters are not set for collective operation";

  CpuCollectives* collectives = params->collectives;
  TF_RET_CHECK(collectives)
      << "Collectives interface is not set for collective operation";

  // Find out rendezvous key and rank in global devices for the current device.
  TF_ASSIGN_OR_RETURN(RendezvousKey key, GetRendezvousKey(*params));
  TF_ASSIGN_OR_RETURN(int32_t rank,
                      RankInGlobalDevices(key, params->global_device_id));

  VLOG(3) << absl::StreamFormat("  rank=%d, key=%s", rank, key.ToString());

  CpuCliqueKey clique_key(key.global_devices);
  TF_ASSIGN_OR_RETURN(
      Communicator * communicator,
      AcquireCommunicator(collectives, clique_key, RankId(rank)));

  return callback(key, *communicator);
}

const BufferAllocation::Slice& CollectiveThunk::source_buffer(
    int64_t index) const {
  return op_buffers_.source_buffers[index];
}

absl::Span<const BufferAllocation::Slice> CollectiveThunk::source_buffers()
    const {
  return op_buffers_.source_buffers;
}

const Shape& CollectiveThunk::source_shape(int64_t index) const {
  return op_buffers_.source_shapes[index];
}

const BufferAllocation::Slice& CollectiveThunk::destination_buffer(
    int64_t index) const {
  return op_buffers_.destination_buffers[index];
}

absl::Span<const BufferAllocation::Slice> CollectiveThunk::destination_buffers()
    const {
  return op_buffers_.destination_buffers;
}

const Shape& CollectiveThunk::destination_shape(int64_t index) const {
  return op_buffers_.destination_shapes[index];
}

}  // namespace xla::cpu
