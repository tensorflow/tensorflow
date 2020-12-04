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

#include "tensorflow/compiler/xla/service/gpu/nccl_all_reduce_thunk.h"

#include <chrono>  // NOLINT (required by TF interfaces)
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#endif
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"

namespace xla {
namespace gpu {

// This file runs collective ops (i.e. ops that communicate between multiple
// GPUs) using NCCL.  Currently only kAllReduce is implemented.
//
// Here's a high-level overview of how running an op works.
//
//  - Multiple threads call NcclAllReduceThunk::ExecuteOnStream.
//  - All threads that "go together" (i.e. are participating in the "same"
//    collective op) choose the same Rendezvous object from a global map.
//  - Once all threads have arrived at the Rendezvous, we know exactly which
//    GPUs are participating in the op, so we get or create a NcclClique
//    containing those GPUs.
//  - We perform the NCCL operation using the clique, then destroy the
//    Rendezvous.  The clique is cached, see below.
//
// Creating NCCL cliques is expensive, so we cache them.  Our policy is, a thunk
// keeps alive all cliques it's ever used.  When the thunk is destroyed, it
// releases its handle on the cliques, and cliques whose refcounts go to 0 are
// destroyed.

/* static */ bool NcclAllReduceThunk::NcclIsEnabled() {
  return true;  // Skylark selects this source file if NCCL is enabled.
}

// Extra data stored in NcclAllReduceThunk that we didn't want to expose in the
// header.  In particular, this stores the thunk's cache of all NcclCliques it's
// ever used.  This causes those cliques to stay alive as long as the thunk
// lives, which is how we avoid expensive reinitialization of NCCL cliques.
struct NcclAllReduceConfig::AuxData {
  tensorflow::mutex mu;
  absl::flat_hash_set<std::shared_ptr<NcclClique>> cliques TF_GUARDED_BY(mu);
};

NcclAllReduceConfig::NcclAllReduceConfig(NcclAllReduceConfig&&) = default;
NcclAllReduceConfig::~NcclAllReduceConfig() = default;

NcclAllReduceConfig GetNcclAllReduceConfig(const HloInstruction* instr,
                                           int64 replica_count) {
  NcclAllReduceConfig config;
  config.operand_count = instr->operands().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    config.operand_element_type.push_back(
        instr->operand(i)->shape().element_type());
  }
  config.replica_count = replica_count;
  config.replica_groups = instr->replica_groups();
  auto reduction_kind = MatchReductionComputation(instr->to_apply());
  CHECK(reduction_kind.has_value());
  config.reduction_kind = reduction_kind.value();

  if (instr->channel_id().has_value()) {
    config.collective_op_kind = RendezvousKey::kCrossModule;
    config.op_id = instr->channel_id().value();
  } else {
    config.collective_op_kind = RendezvousKey::kCrossReplica;
    config.op_id = static_cast<int64>(instr->GetModule()->unique_id());
  }
  config.aux_data = std::make_unique<NcclAllReduceConfig::AuxData>();
  return config;
}

/*static*/ bool NcclAllReduceThunk::CanImplement(const HloInstruction* crs) {
  auto operands_are_supported = [crs]() {
    return absl::c_all_of(crs->operands(), [](HloInstruction* operand) {
      return LayoutUtil::IsDenseArray(operand->shape()) &&
             ToNcclDataType(operand->shape().element_type()).ok();
    });
  };
  return MatchReductionComputation(crs->to_apply()).has_value() &&
         crs->IsCrossReplicaAllReduce() && operands_are_supported();
}

NcclAllReduceThunk::NcclAllReduceThunk(
    ThunkInfo thunk_info, NcclAllReduceConfig&& config,
    std::vector<NcclAllReduceThunk::Buffer> buffers)
    : Thunk(Thunk::kNcclAllReduce, thunk_info),
      config_(std::move(config)),
      buffers_(std::move(buffers)) {
  CHECK_EQ(config_.operand_count, buffers_.size());
}

// Figures out which devices (named by their replica-ids) are participating in
// the all-reduce subgroup that contains device_ordinal.
Status NcclAllReduceThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(1) << "Starting NcclAllReduceThunk.";
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());

  se::StreamExecutor* executor = params.stream->parent();
  int device_ordinal = executor->device_ordinal();
  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              config_.replica_count, config_.replica_groups));

  if (IsGlobalNcclConfig() && (participants.size() != config_.replica_count)) {
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
                               config_.collective_op_kind, config_.op_id);

  TF_ASSIGN_OR_RETURN(
      LockedNcclClique locked_clique,
      AcquireNcclClique(rendezvous_key, device_ordinal, params.stream,
                        local_participants, params.nccl_unique_id_callback));
  ncclComm_t comm =
      locked_clique.clique->GetCommForDeviceOrdinal(device_ordinal);

  VLOG(3) << "Performing all-reduce from device ordinal: " << device_ordinal;
  ncclRedOp_t reduction_kind = ToNcclReduction(config_.reduction_kind);

  se::gpu::ScopedActivateExecutorContext scoped_context(executor);
  cudaStream_t* cu_stream = reinterpret_cast<cudaStream_t*>(
      params.stream->implementation()->GpuStreamMemberHack());
  VLOG(3) << "Using stream pointer: " << cu_stream
          << " on device: " << device_ordinal;
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupStart());
  for (size_t i = 0; i < buffers_.size(); ++i) {
    const Buffer& buffer = buffers_[i];
    const void* send_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.source_buffer)
            .opaque();
    void* recv_buffer =
        params.buffer_allocations->GetDeviceAddress(buffer.destination_buffer)
            .opaque();
    TF_ASSIGN_OR_RETURN(ncclDataType_t datatype,
                        ToNcclDataType(config_.operand_element_type[i]));
    VLOG(3) << absl::StreamFormat(
        "Calling ncclAllReduce(send_buffer=%p, recv_buffer=%p, count=%d, "
        "comm=%p, stream=%p)",
        send_buffer, recv_buffer, buffer.element_count,
        static_cast<const void*>(comm), cu_stream);
    XLA_CUDA_RETURN_IF_ERROR(ncclAllReduce(send_buffer, recv_buffer,
                                           /*count=*/buffer.element_count,
                                           datatype,
                                           /*op=*/reduction_kind, comm,
                                           /*stream=*/*cu_stream));
  }
  XLA_CUDA_RETURN_IF_ERROR(ncclGroupEnd());

  VLOG(3) << "Done performing all-reduce for ordinal: " << device_ordinal;

  // Keep the clique we used alive for as long as this Thunk lives.  Creating
  // new NCCL cliques is expensive, and this is how we avoid thrashing them.
  {
    tensorflow::mutex_lock lock(config_.aux_data->mu);
    config_.aux_data->cliques.insert(std::move(locked_clique.clique));
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
