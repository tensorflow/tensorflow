/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_gather_decomposer.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

// Creates a computation of x + y.
HloComputation* MakeBinaryAdd(PrimitiveType type, HloModule* module) {
  HloComputation::Builder sum_b("add");
  auto x = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(type, {}), "x"));
  auto y = sum_b.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(type, {}), "y"));
  if (type == PRED) {
    sum_b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), HloOpcode::kOr, x, y));
  } else {
    sum_b.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), HloOpcode::kAdd, x, y));
  }
  HloComputation* reduction = module->AddEmbeddedComputation(sum_b.Build());
  return reduction;
}

Status DecomposeAllGather(HloAllGatherInstruction* ag, HloComputation* comp) {
  const int64 shard_size =
      ag->operand(0)->shape().dimensions(ag->all_gather_dimension());
  const int64 ag_size = ag->shape().dimensions(ag->all_gather_dimension());
  TF_RET_CHECK(ag_size % shard_size == 0);
  int64 partition_count = ag_size / shard_size;
  auto zero = comp->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(ag->shape().element_type())));
  zero = comp->AddInstruction(
      HloInstruction::CreateBroadcast(ag->shape(), zero, {}));
  auto zero_index = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(U32)));
  std::vector<HloInstruction*> start_indices(ag->shape().rank(), zero_index);
  auto shard_id_from_subgroup = [&](HloInstruction* replica_or_global_id) {
    if (ag->replica_groups().empty()) {
      return replica_or_global_id;
    }
    if (ag->replica_groups().size() == 1) {
      // Whether the group is {1, 2, ..., N - 1}.
      bool trivial_group = true;
      for (int64 i = 0; i < ag->replica_groups()[0].replica_ids_size(); ++i) {
        if (ag->replica_groups()[0].replica_ids(i) != i) {
          trivial_group = false;
          break;
        }
      }
      if (trivial_group) {
        CHECK_EQ(partition_count, ag->replica_groups()[0].replica_ids_size());
        return replica_or_global_id;
      }
    }
    // Create a table of shard IDs for each replica_or_global_id, then slice it
    // using replica_or_global_id.
    std::vector<uint32> shard_ids(ag->replica_groups().size() *
                                  ag->replica_groups()[0].replica_ids_size());
    for (const auto& group : ag->replica_groups()) {
      for (int64 i = 0; i < group.replica_ids_size(); ++i) {
        shard_ids[group.replica_ids(i)] = i;
      }
    }
    auto id_table = comp->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR1<uint32>(shard_ids)));
    auto shard_id = comp->AddInstruction(HloInstruction::CreateDynamicSlice(
        ShapeUtil::MakeShape(U32, {1}), id_table, {replica_or_global_id}, {1}));
    shard_id = comp->AddInstruction(
        HloInstruction::CreateReshape(ShapeUtil::MakeShape(U32, {}), shard_id));
    return shard_id;
  };
  HloInstruction* shard_id;
  if (ag->channel_id().has_value()) {
    if (ag->use_global_device_ids()) {
      auto pid = comp->AddInstruction(HloInstruction::CreatePartitionId());
      auto rid = comp->AddInstruction(HloInstruction::CreateReplicaId());
      auto pcount = comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<uint32>(partition_count)));
      auto global_id = comp->AddInstruction(HloInstruction::CreateBinary(
          pid->shape(), HloOpcode::kAdd, pid,
          comp->AddInstruction(HloInstruction::CreateBinary(
              pid->shape(), HloOpcode::kMultiply, rid, pcount))));
      shard_id = shard_id_from_subgroup(global_id);
    } else {
      TF_RET_CHECK(!ag->replica_groups().empty());
      TF_RET_CHECK(ag->replica_groups()[0].replica_ids_size() == 1);
      shard_id = comp->AddInstruction(HloInstruction::CreatePartitionId());
    }
  } else {
    shard_id = shard_id_from_subgroup(
        comp->AddInstruction(HloInstruction::CreateReplicaId()));
  }
  start_indices[ag->all_gather_dimension()] =
      comp->AddInstruction(HloInstruction::CreateBinary(
          shard_id->shape(), HloOpcode::kMultiply, shard_id,
          comp->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<uint32>(shard_size)))));
  auto dus = comp->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      zero->shape(), zero, ag->mutable_operand(0), start_indices));
  auto ar = comp->AddInstruction(HloInstruction::CreateAllReduce(
      dus->shape(), {dus},
      MakeBinaryAdd(dus->shape().element_type(), comp->parent()),
      ag->replica_groups(),
      /*constrain_layout=*/ag->constrain_layout(), ag->channel_id(),
      ag->use_global_device_ids()));
  TF_RETURN_IF_ERROR(ag->ReplaceAllUsesWith(ar));
  TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(ag));
  return Status::OK();
}

StatusOr<bool> AllGatherDecomposer::Run(HloModule* module) {
  bool changed = false;
  for (auto comp : module->MakeNonfusionComputations()) {
    for (auto hlo : comp->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kAllGather) {
        continue;
      }
      auto ag = Cast<HloAllGatherInstruction>(hlo);
      if (should_decompose_(*ag)) {
        TF_RETURN_IF_ERROR(DecomposeAllGather(ag, comp));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
