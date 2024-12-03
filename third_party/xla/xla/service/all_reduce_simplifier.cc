/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/service/all_reduce_simplifier.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_replication_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<bool> AllReduceSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(
      auto replication,
      HloReplicationAnalysis::Run(module, /*cross_partition_spmd=*/false));
  std::vector<std::pair<HloInstruction*, int64_t>> all_reduces_to_replace;

  // Returns the number of participants in a replica group if all groups have
  // the same size, or -1 if they have different sizes.
  // Number of participants depends on the mode of the collective operation.
  auto get_participant_counts_for_replica_group =
      [](const HloInstruction* all_reduce) -> absl::StatusOr<int64_t> {
    const HloModuleConfig& config = all_reduce->GetModule()->config();
    TF_ASSIGN_OR_RETURN(
        CollectiveOpGroupMode group_mode,
        GetCollectiveOpGroupMode(all_reduce->channel_id().has_value(),
                                 Cast<HloAllReduceInstruction>(all_reduce)
                                     ->use_global_device_ids()));

    int64_t num_devices = config.num_partitions();
    int64_t num_replicas = config.replica_count();
    TF_ASSIGN_OR_RETURN(std::vector<int64_t> participant_counts,
                        GetPariticipantCountsForReplicaGroups(
                            num_replicas, num_devices,
                            all_reduce->replica_groups(), group_mode));
    if (participant_counts.empty()) {
      return -1;
    }
    if (!absl::c_all_of(participant_counts, [&](int64_t participant_count) {
          return participant_count == participant_counts[0];
        })) {
      return -1;
    }
    return participant_counts[0];
  };

  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      // AllGather and ReduceScatter with the same input and output shape
      if ((inst->opcode() == HloOpcode::kAllGather ||
           inst->opcode() == HloOpcode::kReduceScatter) &&
          ShapeUtil::Compatible(inst->shape(), inst->operand(0)->shape())) {
        changed = true;
        TF_RETURN_IF_ERROR(
            computation->ReplaceInstruction(inst, inst->mutable_operand(0)));
      }
    }
  }

  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (!inst->shape().IsArray()) {
        // We currently do not change tuple-shaped all-reduce.
        // Until XLA will support Token fed AllReduce(), the PyTorch client code
        // uses a fake data token (constant) which relies on this pass to not
        // optimize out (being fed within a tuple input).
        continue;
      }
      if (!inst->IsCrossReplicaAllReduce() && !inst->IsCrossModuleAllReduce()) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(int64_t group_size,
                          get_participant_counts_for_replica_group(inst));

      // We will not simplify this all reduce if any of the following is true:
      // 1. All group do not have the same size.
      //
      // 2. The AllReduce is not cross replica and the group size is not 1.
      // Since the replication analysis performed earlier is only for cross
      // replica spmd.
      //
      // 3. The AllReduce is not cross replica and the module is not using spmd.
      if (group_size == -1 ||
          (!inst->IsCrossReplicaAllReduce() && group_size != 1) ||
          (!inst->IsCrossReplicaAllReduce() &&
           !module->config().use_spmd_partitioning())) {
        continue;
      }
      if (replication->HloInstructionIsReplicatedAt(inst->operand(0), {}) ||
          group_size == 1) {
        all_reduces_to_replace.push_back({inst, group_size});
      }
    }
  }

  for (auto all_reduce_and_group_size : all_reduces_to_replace) {
    auto all_reduce = all_reduce_and_group_size.first;
    const int64_t replica_group_size = all_reduce_and_group_size.second;
    if (replica_group_size == 1) {
      TF_RETURN_IF_ERROR(all_reduce->parent()->ReplaceInstruction(
          all_reduce, all_reduce->mutable_operand(0)));
      changed = true;
      continue;
    }
    if (all_reduce->to_apply()->instruction_count() != 3 ||
        all_reduce->to_apply()->num_parameters() != 2) {
      continue;
    }
    HloInstruction* replacement;
    switch (all_reduce->to_apply()->root_instruction()->opcode()) {
      case HloOpcode::kAdd: {
        // Create the multiplier:
        //   broadcast(convert_to_matching_type(s32 group size))
        auto multiplier =
            all_reduce->parent()->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<int32_t>(replica_group_size)));
        if (all_reduce->shape().element_type() != S32) {
          multiplier = all_reduce->parent()->AddInstruction(
              HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(
                      multiplier->shape(), all_reduce->shape().element_type()),
                  multiplier));
        }
        if (all_reduce->shape().rank() > 0) {
          multiplier = all_reduce->parent()->AddInstruction(
              HloInstruction::CreateBroadcast(all_reduce->shape(), multiplier,
                                              {}));
        }
        replacement =
            all_reduce->parent()->AddInstruction(HloInstruction::CreateBinary(
                all_reduce->shape(), HloOpcode::kMultiply,
                all_reduce->mutable_operand(0), multiplier));
        break;
      }
      case HloOpcode::kMinimum:
      case HloOpcode::kMaximum:
      case HloOpcode::kOr:
      case HloOpcode::kAnd:
        replacement = all_reduce->mutable_operand(0);
        break;
      default:
        continue;
    }
    VLOG(2) << "Replacing " << all_reduce->ToString() << " with "
            << replacement->ToString();
    TF_RETURN_IF_ERROR(all_reduce->ReplaceAllUsesWith(replacement));
    changed = true;
  }
  return changed;
}

}  // namespace xla
