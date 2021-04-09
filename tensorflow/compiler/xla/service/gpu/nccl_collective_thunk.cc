/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/nccl_collective_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"

namespace xla {
namespace gpu {

// This file runs collective ops (i.e. ops that communicate between multiple
// GPUs) using NCCL.
//
// Here's a high-level overview of how running an op works.
//
//  - Multiple threads call ExecuteOnStream.
//  - All threads that "go together" (i.e. are participating in the "same"
//    collective op) choose the same Rendezvous object from a global map.
//  - Once all threads have arrived at the Rendezvous, we know exactly which
//    GPUs are participating in the op, so we get or create a NcclClique
//    containing those GPUs.
//  - We perform the NCCL operation using the clique.

NcclCollectiveConfig::NcclCollectiveConfig() = default;
NcclCollectiveConfig::NcclCollectiveConfig(NcclCollectiveConfig&&) = default;
NcclCollectiveConfig::~NcclCollectiveConfig() = default;
NcclCollectiveConfig& NcclCollectiveConfig::operator=(NcclCollectiveConfig&&) =
    default;

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
bool NcclCollectiveConfig::IsDegenerate(int64_t replica_count,
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
    case CollectiveOpGroupMode::kCrossReplica:
      return all_groups_singleton || (groups_empty && replica_count == 1);
    case CollectiveOpGroupMode::kCrossPartition:
      return all_groups_singleton || (groups_empty && partition_count == 1);
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      return (all_groups_singleton && partition_count == 1) ||
             (groups_empty && replica_count == 1 && partition_count == 1);
    case CollectiveOpGroupMode::kFlattenedID:
      CHECK(!groups_empty)
          << "replica groups cannot be empty if use_global_device_ids = true";
      return all_groups_singleton;
    default:
      CHECK(0) << "Invalid collective op mode";
      return false;
  }
}

/* static */ bool NcclCollectiveThunk::NcclIsEnabled() {
#if XLA_ENABLE_XCCL
  return true;
#else
  return false;
#endif
}

Status NcclCollectiveThunk::ExecuteOnStream(const ExecuteParams& params) {
#if XLA_ENABLE_XCCL
  VLOG(1) << absl::StreamFormat("Starting %s.", ThunkKindToString(kind()));
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());

  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              config().replica_groups, config().group_mode));

  if (IsGlobalNcclConfig() &&
      (participants.size() != params.device_assn->replica_count())) {
    return InvalidArgument(
        "Partial replica groups are not allowed when using NCCL_COMM_ID "
        "environment configuration.");
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<LocalParticipant> local_participants,
      GetLocalParticipants(participants, params.gpu_global_device_ids));

  // Create the rendezvous for this collective operation.
  const RendezvousKey rendezvous_key(
      params.run_id, std::move(participants), local_participants.size(),
      config().collective_op_kind, config().op_id);
  VLOG(2) << GetDeviceString(params) << ": key " << rendezvous_key.ToString()
          << "\n";

  int device_ordinal = params.stream->parent()->device_ordinal();

  TF_ASSIGN_OR_RETURN(
      LockedNcclClique locked_clique,
      AcquireNcclClique(rendezvous_key, device_ordinal, params.stream,
                        local_participants, params.nccl_unique_id_callback));
  ncclComm_t comm =
      locked_clique.clique.GetCommForDeviceOrdinal(device_ordinal);

  se::StreamExecutor* executor = params.stream->parent();
  se::gpu::ScopedActivateExecutorContext scoped_context(executor);

  TF_RETURN_IF_ERROR(RunNcclCollective(params, comm));
  return Status::OK();
#else   // XLA_ENABLE_XCCL
  return Unimplemented(
      "NCCL support is not available: this binary was not built with a CUDA "
      "compiler, which is necessary to build the NCCL source library.");
#endif  // XLA_ENABLE_XCCL
}

std::string NcclCollectiveThunk::GetDeviceString(
    const ExecuteParams& params) const {
  int device_ordinal = params.stream->parent()->device_ordinal();
  GlobalDeviceId global_device_id = params.GetGlobalDeviceId().ValueOrDie();
  DeviceAssignment::LogicalID logical_id =
      params.device_assn->LogicalIdForDevice(global_device_id).ValueOrDie();
  return absl::StrFormat("(r%d, p%d) : GlobalID %d, ord %d",
                         logical_id.replica_id, logical_id.computation_id,
                         global_device_id.value(), device_ordinal);
}

bool IsTypeSupportedByNccl(PrimitiveType element_type) {
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
      return true;
    default:
      return false;
  }
}

}  // namespace gpu
}  // namespace xla
