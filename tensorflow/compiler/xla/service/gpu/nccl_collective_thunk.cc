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
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
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

NcclCollectiveConfig GetNcclCollectiveConfig(const HloInstruction* hlo,
                                             int64 replica_count) {
  NcclCollectiveConfig config;
  config.operand_count = hlo->operands().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    config.operand_element_type.push_back(
        hlo->operand(i)->shape().element_type());
  }
  config.replica_count = replica_count;
  config.replica_groups = hlo->replica_groups();

  if (hlo->channel_id().has_value()) {
    config.collective_op_kind = RendezvousKey::kCrossModule;
    config.op_id = *hlo->channel_id();
  } else {
    config.collective_op_kind = RendezvousKey::kCrossReplica;
    config.op_id = static_cast<int64>(hlo->GetModule()->unique_id());
  }
  return config;
}

/* static */ bool NcclCollectiveThunk::NcclIsEnabled() {
  return true;  // Skylark selects this source file if NCCL is enabled.
}

Status NcclCollectiveThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(1) << absl::StreamFormat("Starting %s.", ThunkKindToString(kind()));
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());

  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              config().replica_count, config().replica_groups));

  if (IsGlobalNcclConfig() && (participants.size() != config().replica_count)) {
    return InvalidArgument(
        "Partial replica groups are not allowed when using NCCL_COMM_ID "
        "environment configuration.");
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<LocalParticipant> local_participants,
      GetLocalParticipants(participants, params.gpu_global_device_ids));

  // Create the rendezvous for this collective operation.
  RendezvousKey rendezvous_key(params.run_id, std::move(participants),
                               local_participants.size(),
                               config().collective_op_kind, config().op_id);

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
}

}  // namespace gpu
}  // namespace xla
