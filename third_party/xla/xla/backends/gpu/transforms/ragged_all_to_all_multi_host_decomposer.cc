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

#include "xla/backends/gpu/transforms/ragged_all_to_all_multi_host_decomposer.h"

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
#include "xla/array.h"
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

// Returns a permutation of the devices in the replica group such that devices
// on the same host are next to each other. The order of the devices within a
// host is preserved.
absl::InlinedVector<int64_t, 8> FindPermutation(
    const ReplicaGroup& replica_group, int64_t num_devices_per_host) {
  int64_t num_devices_in_replica = replica_group.replica_ids_size();

  absl::InlinedVector<int64_t, 8> permutation(num_devices_in_replica);
  absl::c_iota(permutation, 0);

  absl::c_stable_sort(permutation, [&](int64_t i, int64_t j) {
    int64_t host_i = replica_group.replica_ids(i) / num_devices_per_host;
    int64_t host_j = replica_group.replica_ids(j) / num_devices_per_host;
    return host_i < host_j;
    return replica_group.replica_ids(i) < replica_group.replica_ids(j);
  });
  return permutation;
}

// Returns a permutation of the devices in the replica groups such that devices
// on the same host are next to each other. Returns std::nullopt if the
// permutation is not the same for all replica groups.
std::optional<absl::InlinedVector<int64_t, 8>> FindReplicaGroupsPermutation(
    absl::Span<ReplicaGroup const> replica_groups,
    int64_t num_devices_per_host) {
  absl::InlinedVector<int64_t, 8> permutation =
      FindPermutation(replica_groups[0], num_devices_per_host);

  // Check that all replica groups have the same permutation. Operand
  // permutation doesn't not depend on the device id, so if permutations are
  // different, we can't rewrite the ragged-all-to-all.
  for (int64_t i = 1; i < replica_groups.size(); ++i) {
    auto replica_group_permutation =
        FindPermutation(replica_groups[i], num_devices_per_host);
    if (replica_group_permutation != permutation) {
      return std::nullopt;
    }
  }

  return permutation;
}

// Shuffle values in the hlo instruction based on the permutation.
HloInstruction* ShuffleMetadataOperandValues(
    HloInstruction* hlo, absl::Span<int64_t const> permutation) {
  // If the permutation is already sorted, then we don't need to shuffle.
  if (absl::c_is_sorted(permutation)) {
    return hlo;
  }

  HloComputation* computation = hlo->parent();

  PrimitiveType element_type = hlo->shape().element_type();
  int64_t num_elements = ShapeUtil::ElementsIn(hlo->shape());
  int64_t num_replicas = permutation.size();
  int64_t num_elements_per_replica = num_elements / permutation.size();
  Shape linear_shape = ShapeUtil::MakeShape(element_type, {num_elements});
  Shape gather_shape = ShapeUtil::MakeShape(
      element_type, {num_replicas, num_elements_per_replica});

  Array<int64_t> permutation_array({num_replicas, 1});
  for (int64_t i = 0; i < permutation.size(); ++i) {
    permutation_array(i, 0) = num_elements_per_replica * permutation[i];
  }

  auto permutation_constant =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateFromArray(permutation_array)));

  hlo = computation->AddInstruction(
      HloInstruction::CreateReshape(linear_shape, hlo));

  hlo = computation->AddInstruction(
      HloInstruction::CreateGather(gather_shape, hlo, permutation_constant,
                                   HloGatherInstruction::MakeGatherDimNumbers(
                                       /*offset_dims=*/{1},
                                       /*collapsed_slice_dims=*/{},
                                       /*start_index_map=*/{0},
                                       /*index_vector_dim=*/1),
                                   /*slice_sizes=*/{num_elements_per_replica},
                                   /*indices_are_sorted=*/false));

  return computation->AddInstruction(
      HloInstruction::CreateReshape(linear_shape, hlo));
}

// Corrects the offsets in the local metadata to account for the number of input
// rows in the combined ragged tensor.
HloInstruction* CorrectOffsets(int64_t offset, HloInstruction* local_metadata,
                               HloComputation* computation) {
  const Shape& shape = local_metadata->shape();

  HloInstruction* iota = computation->AddInstruction(
      HloInstruction::CreateIota(/*shape=*/shape, /*iota_dimension=*/0));

  HloInstruction* num_input_rows_constant = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int64_t>(offset)));

  HloInstruction* num_input_rows_constant_broadcast =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          /*shape=*/shape, num_input_rows_constant,
          /*broadcast_dimensions=*/{}));

  HloInstruction* input_offsets_offset =
      computation->AddInstruction(HloInstruction::CreateBinary(
          /*shape=*/shape, HloOpcode::kMultiply,
          /*lhs=*/iota, /*rhs=*/num_input_rows_constant_broadcast));

  return computation->AddInstruction(HloInstruction::CreateBinary(
      /*shape=*/shape, HloOpcode::kAdd,
      /*lhs=*/local_metadata,
      /*rhs=*/input_offsets_offset));
}

// Exchanges the metadata operands between the hosts and computes the intra-host
// metadata.
absl::InlinedVector<HloInstruction*, 4> GetIntraHostMetadata(
    HloRaggedAllToAllInstruction* ragged_all_to_all,
    HloComputation* computation, absl::Span<ReplicaGroup const> replica_groups,
    absl::Span<int64_t const> replica_groups_permutation, int64_t num_hosts,
    int64_t num_devices_in_replica) {
  int64_t num_devices_in_replica_per_host = num_devices_in_replica / num_hosts;

  absl::InlinedVector<HloInstruction*, 4> metadata_operands;
  metadata_operands.reserve(4);
  for (int i = 2; i < 6; ++i) {
    metadata_operands.push_back(ragged_all_to_all->mutable_operand(i));
    metadata_operands.back() = ShuffleMetadataOperandValues(
        metadata_operands.back(), replica_groups_permutation);
  }

  Shape metadata_operand_shape = metadata_operands[0]->shape();

  int64_t num_updates_per_replica =
      metadata_operand_shape.dimensions(0) / num_devices_in_replica;

  Shape new_metadata_shape = ShapeUtil::MakeShape(
      metadata_operand_shape.element_type(),
      {num_hosts, num_devices_in_replica_per_host, num_updates_per_replica});

  Shape new_metadata_transposed_shape = ShapeUtil::MakeShape(
      metadata_operand_shape.element_type(),
      {num_devices_in_replica_per_host, num_hosts, num_updates_per_replica});

  for (int64_t i = 0; i < metadata_operands.size(); ++i) {
    metadata_operands[i] =
        computation->AddInstruction(HloInstruction::CreateReshape(
            new_metadata_shape, metadata_operands[i]));
  }

  Shape all_to_all_shape =
      ShapeUtil::MakeShape(metadata_operand_shape.element_type(),
                           {num_hosts, num_devices_in_replica_per_host,
                            4 * num_updates_per_replica});

  HloInstruction* all_to_all_input =
      computation->AddInstruction(HloInstruction::CreateConcatenate(
          /*shape=*/all_to_all_shape,
          /*operands=*/metadata_operands,
          /*dimension=*/2));

  HloInstruction* all_to_all =
      computation->AddInstruction(HloInstruction::CreateAllToAll(
          /*shape=*/all_to_all_shape,
          /*operands=*/{all_to_all_input},
          /*device_list=*/CollectiveDeviceList(replica_groups),
          /*constrain_layout=*/false,
          /*channel_id=*/ragged_all_to_all->channel_id().has_value()
              ? std::make_optional(NextChannelId(*computation->parent()))
              : std::nullopt,
          /*split_dimension=*/0));

  for (int i = 0; i < metadata_operands.size(); ++i) {
    metadata_operands[i] =
        computation->AddInstruction(HloInstruction::CreateSlice(
            /*shape=*/new_metadata_shape,
            /*operand=*/all_to_all,
            /*start_indices=*/{0, 0, i * num_updates_per_replica},
            /*limit_indices=*/
            {num_hosts, num_devices_in_replica_per_host,
             (i + 1) * num_updates_per_replica},
            /*strides=*/{1, 1, 1}));
  }

  // Correct input offsets that need to be adjusted for the number of input
  // rows.
  metadata_operands[0] =
      CorrectOffsets(ragged_all_to_all->operand(0)->shape().dimensions(0),
                     metadata_operands[0], computation);

  for (int i = 0; i < metadata_operands.size(); ++i) {
    metadata_operands[i] =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            /*shape=*/new_metadata_transposed_shape,
            /*operand=*/metadata_operands[i],
            /*dimensions=*/{1, 0, 2}));
    metadata_operands[i] =
        computation->AddInstruction(HloInstruction::CreateReshape(
            metadata_operand_shape, metadata_operands[i]));
  }

  return metadata_operands;
}

// Decomposes a dispatch `ragged-all-to-all` collective into an inter-host
// `all-gather` and an intra-host `ragged-all-to-all`.
//
// Dispatch phase of MoE layer is characterized by the following properties:
//   - The input is dense and all or most of the rows are significant.
//   - The output is larger than the input, because we need to have a static
//   allocation that will accommodate all the possible rows.
// In case of dispatch phase, doing `all-gather` on inputs first is more
// efficient, because we're only transferring significant data with up to 2x
// overhead.
absl::StatusOr<bool> DecomposeDispatchRaggedAllToAll(
    HloRaggedAllToAllInstruction* ragged_all_to_all,
    HloComputation* computation,
    absl::Span<ReplicaGroup const> inter_host_replica_groups,
    absl::Span<ReplicaGroup const> intra_host_replica_groups,
    absl::Span<int64_t const> replica_groups_permutation, int64_t num_hosts,
    int64_t num_devices_in_replica) {
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

  absl::InlinedVector<HloInstruction*, 4> intra_host_metadata =
      GetIntraHostMetadata(
          ragged_all_to_all, computation, inter_host_replica_groups,
          replica_groups_permutation, num_hosts, num_devices_in_replica);

  HloInstruction* new_ragged_all_to_all =
      computation->AddInstruction(HloInstruction::CreateRaggedAllToAll(
          /*shape=*/ragged_all_to_all->shape(),
          /*operands=*/
          {all_gather_input, ragged_all_to_all->mutable_operand(1),
           intra_host_metadata[0], intra_host_metadata[1],
           intra_host_metadata[2], intra_host_metadata[3]},
          /*device_list=*/CollectiveDeviceList(intra_host_replica_groups),
          /*channel_id=*/ragged_all_to_all->channel_id()));

  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(ragged_all_to_all,
                                                     new_ragged_all_to_all));

  return true;
}

// Decomposes a combine `ragged-all-to-all` collective.
//
// Combine phase of MoE layer is characterized by the following properties:
//   - The input is larget than the output, because it contains rows distributed
//     by the dispatch phase.
//   - Most of the input rows are not significant, because it's padded to
//     accommodate all possible rows.
//   - The distribution of the significant rows depends on the runtime state of
//     the MoE layer, so we can't reason about it in an HLO rewrite pass.
//
// An `all-gather` as a first step would be inefficient in this case, because
// we would be transferring a lot of padding. An optimal way is to do
// `ragged-all-to-all` within the hosts to partially gather the significant data
// into smaller temporary buffer of output size. Exchange the data cross-host
// and the do another local `ragged-all-to-all` to the final output. This way we
// transfer more significant data with minimal padding with up to 2x overhead.
absl::StatusOr<bool> DecomposeCombineRaggedAllToAll(
    HloRaggedAllToAllInstruction* ragged_all_to_all,
    HloComputation* computation,
    absl::Span<ReplicaGroup const> inter_host_replica_groups,
    absl::Span<ReplicaGroup const> intra_host_replica_groups,
    absl::Span<int64_t const> replica_groups_permutation, int64_t num_hosts,
    int64_t num_devices_in_replica, int64_t num_participating_devices) {
  const Shape& metadata_operand_shape = ragged_all_to_all->operand(2)->shape();

  auto* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(
          ragged_all_to_all->operand(1)->shape().element_type())));

  Shape tmp_output_shape = ragged_all_to_all->shape();
  tmp_output_shape.set_dimensions(0,
                                  num_hosts * tmp_output_shape.dimensions(0));

  auto* zero_broadcast =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          /*shape=*/tmp_output_shape, zero, /*broadcast_dimensions=*/{}));

  int64_t num_devices_in_replica_per_host = num_devices_in_replica / num_hosts;

  int64_t num_updates_per_replica =
      ragged_all_to_all->operand(2)->shape().dimensions(0) /
      num_devices_in_replica;

  auto get_intra_host_metadata = [&](HloInstruction* metadata_operand,
                                     bool correct_offsets) {
    metadata_operand = ShuffleMetadataOperandValues(metadata_operand,
                                                    replica_groups_permutation);

    metadata_operand =
        computation->AddInstruction(HloInstruction::CreateReshape(
            /*shape=*/ShapeUtil::MakeShape(
                metadata_operand->shape().element_type(),
                {num_hosts, num_devices_in_replica_per_host,
                 num_updates_per_replica}),
            /*operand=*/metadata_operand));

    if (correct_offsets) {
      metadata_operand =
          CorrectOffsets(ragged_all_to_all->operand(1)->shape().dimensions(0),
                         metadata_operand, computation);
    }

    metadata_operand =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            /*shape=*/ShapeUtil::MakeShape(
                metadata_operand->shape().element_type(),
                {num_devices_in_replica_per_host, num_hosts,
                 num_updates_per_replica}),
            /*operand=*/metadata_operand,
            /*dimensions=*/{1, 0, 2}));

    return computation->AddInstruction(HloInstruction::CreateReshape(
        /*shape=*/metadata_operand_shape, /*operand=*/metadata_operand));
  };

  absl::InlinedVector<HloInstruction*, 4> intra_host_ragged_all_to_all_operands{
      ragged_all_to_all->mutable_operand(0),
      zero_broadcast,
      get_intra_host_metadata(ragged_all_to_all->mutable_operand(2),
                              /*correct_offsets=*/false),
      get_intra_host_metadata(ragged_all_to_all->mutable_operand(3),
                              /*correct_offsets=*/false),
      get_intra_host_metadata(ragged_all_to_all->mutable_operand(4),
                              /*correct_offsets=*/true),
      get_intra_host_metadata(ragged_all_to_all->mutable_operand(5),
                              /*correct_offsets=*/false),
  };

  HloInstruction* intra_host_ragged_all_to_all =
      computation->AddInstruction(HloInstruction::CreateRaggedAllToAll(
          /*shape=*/zero_broadcast->shape(),
          /*operands=*/intra_host_ragged_all_to_all_operands,
          /*device_list=*/CollectiveDeviceList(intra_host_replica_groups),
          /*channel_id=*/ragged_all_to_all->channel_id().has_value()
              ? std::make_optional(NextChannelId(*computation->parent()))
              : std::nullopt));

  HloInstruction* local_inputs =
      computation->AddInstruction(HloInstruction::CreateAllToAll(
          intra_host_ragged_all_to_all->shape(), {intra_host_ragged_all_to_all},
          /*device_list=*/CollectiveDeviceList(inter_host_replica_groups),
          /*constrain_layout=*/false,
          /*channel_id=*/ragged_all_to_all->channel_id().has_value()
              ? std::make_optional(NextChannelId(*computation->parent()))
              : std::nullopt,
          /*split_dimension=*/0));

  absl::InlinedVector<ReplicaGroup, 16> degenerated_replica_groups(
      num_participating_devices);
  for (int64_t i = 0; i < num_participating_devices; ++i) {
    degenerated_replica_groups[i].add_replica_ids(i);
  }

  HloInstruction* output_offsets = ragged_all_to_all->mutable_operand(4);

  output_offsets = computation->AddInstruction(HloInstruction::CreateReshape(
      /*shape=*/ShapeUtil::MakeShape(
          output_offsets->shape().element_type(),
          {num_devices_in_replica, num_updates_per_replica}),
      /*operand=*/output_offsets));

  output_offsets = computation->AddInstruction(HloInstruction::CreateAllToAll(
      /*shape=*/output_offsets->shape(),
      /*operands=*/{output_offsets},
      /*device_list=*/ragged_all_to_all->device_list(),
      /*constrain_layout=*/false,
      /*channel_id=*/ragged_all_to_all->channel_id().has_value()
          ? std::make_optional(NextChannelId(*computation->parent()))
          : std::nullopt,
      /*split_dimension=*/0));

  output_offsets = computation->AddInstruction(HloInstruction::CreateReshape(
      /*shape=*/metadata_operand_shape, /*operand=*/output_offsets));

  std::vector<HloInstruction*> local_ragged_all_to_all_operands = {
      local_inputs,   ragged_all_to_all->mutable_operand(1),
      output_offsets, ragged_all_to_all->mutable_operand(5),
      output_offsets, ragged_all_to_all->mutable_operand(5),
  };

  for (int i = 2; i < 6; ++i) {
    local_ragged_all_to_all_operands[i] = ShuffleMetadataOperandValues(
        local_ragged_all_to_all_operands[i], replica_groups_permutation);
  }

  HloInstruction* local_input_offsets =
      computation->AddInstruction(HloInstruction::CreateReshape(
          /*shape=*/ShapeUtil::MakeShape(
              output_offsets->shape().element_type(),
              {num_hosts, num_devices_in_replica_per_host,
               num_updates_per_replica}),
          /*operand=*/local_ragged_all_to_all_operands[2]));

  local_input_offsets =
      CorrectOffsets(ragged_all_to_all->operand(1)->shape().dimensions(0),
                     local_input_offsets, computation);

  local_ragged_all_to_all_operands[2] =
      computation->AddInstruction(HloInstruction::CreateReshape(
          /*shape=*/metadata_operand_shape, /*operand=*/local_input_offsets));

  HloInstruction* local_ragged_all_to_all =
      computation->AddInstruction(HloInstruction::CreateRaggedAllToAll(
          /*shape=*/ragged_all_to_all->shape(),
          /*operands=*/local_ragged_all_to_all_operands,
          /*device_list=*/CollectiveDeviceList(degenerated_replica_groups),
          /*channel_id=*/ragged_all_to_all->channel_id()));

  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(ragged_all_to_all,
                                                     local_ragged_all_to_all));

  return true;
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

  // Offsets and sizes in metadata operands are stored in the order of replica
  // groups. For example, if the replica groups are:
  //   {{0, 2, 4, 6, 1, 3, 5, 7}}
  // Then the offsets and sizes are stored in the order of
  //   [0, 2, 4, 6, 1, 3, 5, 7]
  // In the decomposition, we want to exchange all the intra-host metadata
  // between hosts. To do that we want to group the metadata by hosts. We
  // compute permutation that need to be performed on the metadata operand and
  // use gather to move values. After the shuffle, offsets and sizes will be
  // ordered as:
  //   [0, 2, 1, 3, 4, 6, 5, 7]
  auto replica_groups_permutation = FindReplicaGroupsPermutation(
      replica_groups, fast_interconnect_slice_size);
  // Empty value means that we can not find such permutation and the
  // ragged-all-to-all can not be decomposed.
  if (!replica_groups_permutation.has_value()) {
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

  int64_t num_input_rows = ragged_all_to_all->operand(0)->shape().dimensions(0);
  int64_t num_output_rows =
      ragged_all_to_all->operand(1)->shape().dimensions(0);

  if (num_input_rows > num_output_rows) {
    return DecomposeCombineRaggedAllToAll(
        ragged_all_to_all, computation, inter_host_replica_groups,
        intra_host_replica_groups, *replica_groups_permutation, num_hosts,
        num_devices_in_replica, num_participating_devices);
  }

  return DecomposeDispatchRaggedAllToAll(
      ragged_all_to_all, computation, inter_host_replica_groups,
      intra_host_replica_groups, *replica_groups_permutation, num_hosts,
      num_devices_in_replica);
}

absl::StatusOr<bool> RaggedAllToAllMultiHostDecomposer::RunImpl(
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
