/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/collective_decomposer_utils.h"

#include <functional>
#include <limits>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {

// Create the start indices for decompositing the given collective.
StatusOr<std::vector<HloInstruction *>>
CreateStartIndicesForCollectiveDecomposition(
    CollectiveOpGroupMode group_mode,
    absl::Span<const ReplicaGroup> replica_groups, const Shape &shard_shape,
    int64_t shard_dimension, HloComputation *computation,
    std::function<void(Shape &)> update_layout) {
  HloInstruction *zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(U32)));
  if (update_layout) {
    update_layout(*zero->mutable_shape());
  }
  std::vector<HloInstruction *> start_indices(shard_shape.rank(), zero);
  const Shape &scalar_shape = zero->shape();

  auto create_flattened_id = [&](HloInstruction *replica_index) {
    if (replica_index == zero) {
      // special case for 0 * num_partitions + partition_id
      return computation->AddInstruction(HloInstruction::CreatePartitionId());
    }
    const HloModuleConfig &config = computation->parent()->config();
    HloInstruction *partition_count =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<uint32_t>(config.num_partitions())));
    HloInstruction *mul = computation->AddInstruction(
        HloInstruction::CreateBinary(scalar_shape, HloOpcode::kMultiply,
                                     replica_index, partition_count));
    return computation->AddInstruction(HloInstruction::CreateBinary(
        scalar_shape, HloOpcode::kAdd, mul,
        computation->AddInstruction(HloInstruction::CreatePartitionId())));
  };

  HloInstruction *participant_id;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
      participant_id =
          computation->AddInstruction(HloInstruction::CreateReplicaId());
      break;
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      // For this mode, the replica groups contain replica_id's, but the
      // participant are replicas with the given replica_id across all
      // partitions (ordered in partition id order, see
      // GetParticipatingDevicesGroups). So replica group {0, 3} corresponds to
      // the participants {r0p0, r0p1, ..., r0pn, r3p0, r3p1, ... r3pn} where
      // number of partitions = n + 1. So the slice index for a given execution
      // instance can be computed by first computing its replica index (using
      // replica_id) and then accounting for partition_id:
      //    replica_index = map replica_id to index using the replica_groups.
      //    index = replica_index * num_partitions + partition_id;
      participant_id =
          computation->AddInstruction(HloInstruction::CreateReplicaId());
      break;
    case CollectiveOpGroupMode::kCrossPartition:
      participant_id =
          computation->AddInstruction(HloInstruction::CreatePartitionId());
      break;
    case CollectiveOpGroupMode::kFlattenedID:
      participant_id = create_flattened_id(
          computation->AddInstruction(HloInstruction::CreateReplicaId()));
      break;
  }

  auto is_trivial_group = [](absl::Span<const ReplicaGroup> replica_groups) {
    if (replica_groups.empty()) {
      return true;
    }
    if (replica_groups.size() == 1) {
      for (int64_t index = 0; index < replica_groups[0].replica_ids_size();
           ++index) {
        if (index != replica_groups[0].replica_ids(index)) {
          return false;
        }
      }
      return true;
    }
    return false;
  };

  HloInstruction *index;
  if (is_trivial_group(replica_groups)) {
    if (replica_groups.size() == 1 &&
        replica_groups[0].replica_ids_size() == 1) {
      // If there is a single replica group with a single ID, it has to be 0 and
      // the index therefore has to be 1
      TF_RET_CHECK(replica_groups[0].replica_ids(0) == 0);
      index = zero;
    } else {
      index = participant_id;
    }
  } else {
    size_t num_participants =
        replica_groups.size() * replica_groups.front().replica_ids_size();
    std::vector<uint32_t> index_values(num_participants,
                                       std::numeric_limits<uint32_t>::max());
    for (const ReplicaGroup &rg : replica_groups) {
      for (uint64_t idx = 0; idx < rg.replica_ids_size(); ++idx) {
        int64_t id = rg.replica_ids(idx);
        TF_RET_CHECK(index_values[id] == std::numeric_limits<uint32_t>::max());
        index_values[id] = idx;
      }
    }

    // create a u32 constant table of index values and use dynamic-slice to
    // index into it.
    HloInstruction *table =
        computation->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR1<uint32_t>(index_values)));
    if (update_layout) {
      update_layout(*table->mutable_shape());
    }
    HloInstruction *ds =
        computation->AddInstruction(HloInstruction::CreateDynamicSlice(
            ShapeUtil::MakeShape(U32, {1}), table, {participant_id}, {1}));
    if (update_layout) {
      update_layout(*ds->mutable_shape());
    }
    index = computation->AddInstruction(
        HloInstruction::CreateReshape(scalar_shape, ds));
  }

  // For cross-replica and partition mode, we need to scale the index (which is
  // the replica index) by num_partitions and add partition_id;
  if (group_mode == CollectiveOpGroupMode::kCrossReplicaAndPartition) {
    index = create_flattened_id(index);
  }

  // scale index by the shard size, which is the size of the shard_dimension.
  HloInstruction *scale = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<uint32_t>(
          shard_shape.dimensions(shard_dimension))));
  index = computation->AddInstruction(HloInstruction::CreateBinary(
      scalar_shape, HloOpcode::kMultiply, index, scale));
  start_indices[shard_dimension] = index;
  return start_indices;
}

}  // namespace xla
