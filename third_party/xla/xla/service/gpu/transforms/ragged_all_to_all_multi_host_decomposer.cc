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
#include <iterator>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
using hlo_query::NextChannelId;

// Exchanges the metadata between the hosts and computes the intra-host
// metadata.
//
// If `correct_offsets` is true, the offsets are corrected to account for the
// number of input rows in the combined ragged tensor. It's needed for
// `input_offsets`.
HloInstruction* GetIntraHostMetadata(
    HloRaggedAllToAllInstruction* ragged_all_to_all,
    HloInstruction* metadata_operand, HloComputation* computation,
    absl::Span<ReplicaGroup const> replica_groups, int64_t num_hosts,
    int64_t num_devices_in_replica, bool correct_offsets) {
  int64_t num_devices_in_replica_per_host = num_devices_in_replica / num_hosts;

  int64_t num_updates_per_replica =
      metadata_operand->shape().dimensions(0) / num_devices_in_replica;

  Shape new_metadata_shape = ShapeUtil::MakeShape(
      metadata_operand->shape().element_type(),
      {num_hosts, num_devices_in_replica_per_host, num_updates_per_replica});

  Shape new_metadata_transposed_shape = ShapeUtil::MakeShape(
      metadata_operand->shape().element_type(),
      {num_devices_in_replica_per_host, num_hosts, num_updates_per_replica});

  HloInstruction* new_input_offsets = computation->AddInstruction(
      HloInstruction::CreateReshape(new_metadata_shape, metadata_operand));

  HloInstruction* new_local_metadata =
      computation->AddInstruction(HloInstruction::CreateAllToAll(
          /*shape=*/new_metadata_shape,
          /*operands=*/{new_input_offsets},
          /*device_list=*/CollectiveDeviceList(replica_groups),
          /*constrain_layout=*/false,
          /*channel_id=*/ragged_all_to_all->channel_id().has_value()
              ? std::make_optional(NextChannelId(*computation->parent()))
              : std::nullopt,
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

  // Replica groups can be empty in collective instruction. Empty replica groups
  // mean that all devices are participating in the collective. This semantics
  // is hard to handle in an HLO pass, because we don't have enough information
  // about the topology, so it's easier to skip this case. Note that this is not
  // a concert for production models, because Jax fills replica groups for all
  // collectives.
  if (replica_groups.empty()) {
    return false;
  }

  int64_t num_devices_in_replica = replica_groups[0].replica_ids_size();

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

  // Decompose the replica groups into inter-host and intra-host replica groups.
  // For example, if the original replica groups were:
  //   {{0, 2, 4, 6, 8, 10, 12, 14}, {1, 3, 5, 7, 9, 11, 13, 15}}
  // Then the inter-host replica groups would be:
  //   {{0, 8}, {2, 10}, {4, 12}, {6, 14}, {1, 9}, {3, 11}, {5, 13}, {7, 15}}}
  // And the intra-host replica groups would be:
  //   {{0, 2, 4, 6}, {8, 10, 12, 14}, {1, 3, 5, 7}, {9, 11, 13, 15}}
  absl::InlinedVector<ReplicaGroup, 8> intra_host_replica_groups;
  absl::InlinedVector<ReplicaGroup, 8> inter_host_replica_groups;

  for (const auto& replica_group : replica_groups) {
    absl::InlinedVector<int64_t, 8> replicas_per_host(num_hosts);

    absl::InlinedVector<ReplicaGroup, 8> intra_host_replica_group_split(
        num_hosts);
    for (int64_t replica_id : replica_group.replica_ids()) {
      int64_t host_id = replica_id / fast_interconnect_slice_size;

      intra_host_replica_group_split[host_id].add_replica_ids(replica_id);
      replicas_per_host[host_id]++;
    }

    // Check that each group has the same number of replicas per host.
    if (!absl::c_all_of(replicas_per_host,
                        [&](int64_t v) { return v == replicas_per_host[0]; })) {
      return false;
    }

    absl::c_copy(intra_host_replica_group_split,
                 std::back_inserter(intra_host_replica_groups));

    for (int64_t i = 0;
         i < intra_host_replica_group_split[0].replica_ids_size(); ++i) {
      ReplicaGroup inter_host_replica_group;

      inter_host_replica_group.mutable_replica_ids()->Reserve(num_hosts);
      for (int64_t host_id = 0; host_id < num_hosts; ++host_id) {
        inter_host_replica_group.add_replica_ids(
            intra_host_replica_group_split[host_id].replica_ids(i));
      }

      inter_host_replica_groups.push_back(inter_host_replica_group);
    }
  }

  std::vector<HloInstruction*> intra_host_metadata;

  HloInstruction* input_operand = ragged_all_to_all->mutable_operand(0);

  Shape new_input_shape = input_operand->shape();
  new_input_shape.set_dimensions(
      0, num_hosts * input_operand->shape().dimensions(0));

  // The collective can run in two modes: cross-replica and cross-partition. If
  // the original `ragged-all-to-all` has a channel id set, then it's a
  // cross-partition collective. In that case `all-gather` needs a channel_id
  // and `use_global_device_ids=true`.
  // Otherwise, when `ragged-all-to-all` has no channel id, it's a cross-replica
  // collective. In that case `all-gather` doesn't need a `channel_id` and
  // `use_global_device_ids` should be set to false.
  HloInstruction* all_gather_input =
      computation->AddInstruction(HloInstruction::CreateAllGather(
          /*shape=*/new_input_shape,
          /*operands=*/{ragged_all_to_all->mutable_operand(0)},
          /*all_gather_dimension=*/0,
          /*device_list=*/CollectiveDeviceList(inter_host_replica_groups),
          /*constrain_layout=*/false,
          /*channel_id=*/ragged_all_to_all->channel_id().has_value()
              ? std::make_optional(NextChannelId(*computation->parent()))
              : std::nullopt,
          /*use_global_device_ids=*/
          ragged_all_to_all->channel_id().has_value()));

  for (int i = 2; i < 6; ++i) {
    intra_host_metadata.push_back(GetIntraHostMetadata(
        ragged_all_to_all, ragged_all_to_all->mutable_operand(i), computation,
        inter_host_replica_groups, num_hosts, num_devices_in_replica,
        /*correct_offsets=*/i == 2));
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
