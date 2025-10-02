/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/ragged_all_to_all_multi_host_decomposer.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Exchanges the metadata between the hosts and computes the intra-host
// metadata.
//
// If `correct_offsets` is true, the offsets are corrected to account for the
// number of input rows in the combined ragged tensor. It's needed for
// `input_offsets`.
HloInstruction* GetIntraHostMetadata(
    HloRaggedAllToAllInstruction* ragged_all_to_all,
    HloInstruction* metadata_operand, HloComputation* computation,
    const std::vector<ReplicaGroup>& replica_groups,
    int64_t num_updates_per_replica, int64_t fast_interconnect_slice_size,
    int64_t num_hosts, bool correct_offsets) {
  Shape new_metadata_shape = ShapeUtil::MakeShape(
      metadata_operand->shape().element_type(),
      {num_hosts, fast_interconnect_slice_size, num_updates_per_replica});

  Shape new_metadata_transposed_shape = ShapeUtil::MakeShape(
      metadata_operand->shape().element_type(),
      {fast_interconnect_slice_size, num_hosts, num_updates_per_replica});

  HloInstruction* new_input_offsets = computation->AddInstruction(
      HloInstruction::CreateReshape(new_metadata_shape, metadata_operand));

  HloInstruction* new_local_metadata =
      computation->AddInstruction(HloInstruction::CreateAllToAll(
          /*shape=*/new_metadata_shape,
          /*operands=*/{new_input_offsets},
          /*device_list=*/CollectiveDeviceList(replica_groups),
          /*constrain_layout=*/false,
          /*channel_id=*/ragged_all_to_all->channel_id(),
          /*split_dimension=*/0));

  if (correct_offsets) {
    HloInstruction* iota =
        computation->AddInstruction(HloInstruction::CreateIota(
            /*shape=*/new_metadata_shape,
            /*iota_dimension=*/0));

    int64_t num_input_rows =
        ragged_all_to_all->operand(0)->shape().dimensions(0);

    HloInstruction* num_input_rows_constant =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int64_t>(num_input_rows)));

    HloInstruction* num_input_rows_constant_broadcast =
        computation->AddInstruction(HloInstruction::CreateBroadcast(
            /*shape=*/new_metadata_shape, num_input_rows_constant,
            /*broadcast_dimensions=*/{}));

    HloInstruction* input_offsets_offset =
        computation->AddInstruction(HloInstruction::CreateBinary(
            /*shape=*/new_metadata_shape, HloOpcode::kMultiply,
            /*lhs=*/iota, /*rhs=*/num_input_rows_constant_broadcast));

    new_local_metadata =
        computation->AddInstruction(HloInstruction::CreateBinary(
            /*shape=*/new_metadata_shape, HloOpcode::kAdd,
            /*lhs=*/new_local_metadata,
            /*rhs=*/input_offsets_offset));
  }

  HloInstruction* new_local_metadata_transposed =
      computation->AddInstruction(HloInstruction::CreateTranspose(
          /*shape=*/new_metadata_transposed_shape,
          /*operand=*/new_local_metadata,
          /*dimensions=*/{1, 0, 2}));

  HloInstruction* intra_host_metadata =
      computation->AddInstruction(HloInstruction::CreateReshape(
          metadata_operand->shape(), new_local_metadata_transposed));

  return intra_host_metadata;
}

absl::StatusOr<bool> DecomposeRaggedAllToAll(
    HloInstruction* hlo, HloComputation* computation, HloModule* module,
    int64_t fast_interconnect_slice_size) {
  auto* ragged_all_to_all = Cast<HloRaggedAllToAllInstruction>(hlo);

  auto replica_groups = ragged_all_to_all->replica_groups();

  // TODO(b/445380264): Support multiple replica groups.
  if (replica_groups.size() > 1) {
    return false;
  }

  // Replica groups can be empty in collective instruction. Empty replica groups
  // mean that all devices are participating in the collective. This semantics
  // is hard to handle in an HLO pass, because we don't have enough information
  // about the topology, so it's easier to skip this case. Note that this is not
  // a concert for production models, because Jax fills replica groups for all
  // collectives.
  if (replica_groups.empty()) {
    return false;
  }

  const auto& replica_ids = replica_groups[0].replica_ids();

  for (int i = 0; i < replica_ids.size(); ++i) {
    if (i != replica_ids[i]) {
      return false;
    }
  }

  HloInstruction* input_offsets = ragged_all_to_all->mutable_operand(2);

  int64_t num_updates_per_replica =
      input_offsets->shape().dimensions(0) / replica_ids.size();

  int64_t num_participating_devices = 0;
  for (auto& replica_group : replica_groups) {
    num_participating_devices += replica_group.replica_ids_size();
  }

  int64_t num_hosts =
      CeilOfRatio(num_participating_devices, fast_interconnect_slice_size);

  // All participating devices are in the same fast interconnect slice.
  if (num_hosts == 1) {
    return false;
  }

  // TODO(b/445380264): Support more than 2 hosts.
  if (num_hosts != 2) {
    return false;
  }

  std::vector<ReplicaGroup> inter_host_replica_groups(
      fast_interconnect_slice_size);
  std::vector<ReplicaGroup> intra_host_replica_groups(num_hosts);

  for (int i = 0; i < fast_interconnect_slice_size; ++i) {
    inter_host_replica_groups[i].add_replica_ids(i);
    inter_host_replica_groups[i].add_replica_ids(fast_interconnect_slice_size +
                                                 i);

    intra_host_replica_groups[0].add_replica_ids(i);
    intra_host_replica_groups[1].add_replica_ids(fast_interconnect_slice_size +
                                                 i);
  }

  std::vector<HloInstruction*> intra_host_metadata;

  HloInstruction* input_operand = ragged_all_to_all->mutable_operand(0);

  Shape new_input_shape = input_operand->shape();
  new_input_shape.set_dimensions(
      0, num_hosts * input_operand->shape().dimensions(0));

  HloInstruction* all_gather_input =
      computation->AddInstruction(HloInstruction::CreateAllGather(
          /*shape=*/new_input_shape,
          /*operands=*/{ragged_all_to_all->mutable_operand(0)},
          /*all_gather_dimension=*/0,
          /*device_list=*/CollectiveDeviceList(inter_host_replica_groups),
          /*constrain_layout=*/false,
          /*channel_id=*/ragged_all_to_all->channel_id(),
          /*use_global_device_ids=*/false));

  for (int i = 2; i < 6; ++i) {
    intra_host_metadata.push_back(GetIntraHostMetadata(
        ragged_all_to_all, ragged_all_to_all->mutable_operand(i), computation,
        inter_host_replica_groups, num_updates_per_replica,
        fast_interconnect_slice_size, num_hosts, /*correct_offsets=*/i == 2));
  }

  HloInstruction* new_ragged_all_to_all =
      computation->AddInstruction(HloInstruction::CreateRaggedAllToAll(
          /*shape=*/ragged_all_to_all->shape(),
          /*operands=*/
          {all_gather_input, ragged_all_to_all->mutable_operand(1),
           intra_host_metadata[0], intra_host_metadata[1],
           intra_host_metadata[2], intra_host_metadata[3]},
          /*replica_groups=*/intra_host_replica_groups,
          /*channel_id=*/ragged_all_to_all->channel_id()));

  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(hlo, new_ragged_all_to_all));

  return true;
}

absl::StatusOr<bool> RaggedAllToAllMultiHostDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (auto computation : module->computations(execution_threads)) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (HloPredicateIsNotOp<HloOpcode::kRaggedAllToAll>(hlo)) {
        continue;
      }

      if (hlo->operand(2)->shape().element_type() != S64) {
        return absl::InvalidArgumentError(
            "RaggedAllToAllDecomposer only supports S64 offsets. Was "
            "`ragged-all-to-all-canonicalizer` pass executed?");
      }

      TF_ASSIGN_OR_RETURN(
          bool result, DecomposeRaggedAllToAll(hlo, computation, module,
                                               fast_interconnect_slice_size_));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
