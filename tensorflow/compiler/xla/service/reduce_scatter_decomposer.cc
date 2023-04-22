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

#include "tensorflow/compiler/xla/service/reduce_scatter_decomposer.h"

#include <sys/types.h>

#include <limits>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

namespace {

// Create the start indices for decompositing the given collective.
StatusOr<std::vector<HloInstruction *>>
CreateStartIndicesForCollectiveDecomposition(
    CollectiveOpGroupMode group_mode,
    absl::Span<const ReplicaGroup> replica_groups, const Shape &shard_shape,
    int64_t shard_dimension, HloComputation *computation) {
  HloInstruction *zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(U32)));
  std::vector<HloInstruction *> start_indices(shard_shape.rank(), zero);
  const Shape &scalar_shape = zero->shape();

  HloInstruction *participant_id;
  switch (group_mode) {
    case CollectiveOpGroupMode::kCrossReplica:
    case CollectiveOpGroupMode::kCrossReplicaAndPartition:
      participant_id =
          computation->AddInstruction(HloInstruction::CreateReplicaId());
      break;
    case CollectiveOpGroupMode::kCrossPartition:
      participant_id =
          computation->AddInstruction(HloInstruction::CreatePartitionId());
      break;
    case CollectiveOpGroupMode::kFlattenedID: {
      const HloModuleConfig &config = computation->parent()->config();
      HloInstruction *partition_count =
          computation->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32_t>(config.num_partitions())));
      HloInstruction *mul =
          computation->AddInstruction(HloInstruction::CreateBinary(
              scalar_shape, HloOpcode::kMultiply,
              computation->AddInstruction(HloInstruction::CreateReplicaId()),
              partition_count));
      participant_id = computation->AddInstruction(HloInstruction::CreateBinary(
          scalar_shape, HloOpcode::kAdd, mul,
          computation->AddInstruction(HloInstruction::CreatePartitionId())));
      break;
    }
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
    index = participant_id;
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
    HloInstruction *ds =
        computation->AddInstruction(HloInstruction::CreateDynamicSlice(
            ShapeUtil::MakeShape(U32, {1}), table, {participant_id}, {1}));
    index = computation->AddInstruction(
        HloInstruction::CreateReshape(scalar_shape, ds));
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
}  // namespace

StatusOr<bool> ReduceScatterDecomposer::Run(HloModule *module) {
  bool changed = false;
  int64 next_channel_id = hlo_query::NextChannelId(*module);

  for (HloComputation *computation : module->MakeNonfusionComputations()) {
    for (HloInstruction *instruction :
         computation->MakeInstructionPostOrder()) {
      auto *rs = DynCast<HloReduceScatterInstruction>(instruction);
      if (!rs || !rs->shape().IsArray()) {
        continue;
      }

      absl::optional<int64_t> channel_id;
      if (rs->channel_id()) {
        channel_id = next_channel_id++;
      }

      // Create an all-reduce
      HloInstruction *ar =
          computation->AddInstruction(HloInstruction::CreateAllReduce(
              rs->operand(0)->shape(), rs->operands(), rs->to_apply(),
              rs->replica_groups(), rs->constrain_layout(), channel_id,
              rs->use_global_device_ids()));
      // Create start indices for a dynamic slice to decompose the all-reduce
      // results.
      TF_ASSIGN_OR_RETURN(
          CollectiveOpGroupMode group_mode,
          GetCollectiveOpGroupMode(rs->channel_id().has_value(),
                                   rs->use_global_device_ids()));
      TF_ASSIGN_OR_RETURN(std::vector<HloInstruction *> start_indices,
                          CreateStartIndicesForCollectiveDecomposition(
                              group_mode, rs->replica_groups(), rs->shape(),
                              rs->scatter_dimension(), computation));

      HloInstruction *ds =
          computation->AddInstruction(HloInstruction::CreateDynamicSlice(
              rs->shape(), ar, start_indices, rs->shape().dimensions()));

      TF_RETURN_IF_ERROR(rs->ReplaceAllUsesWith(ds));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(rs));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
