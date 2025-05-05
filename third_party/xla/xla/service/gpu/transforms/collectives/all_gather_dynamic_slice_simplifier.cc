/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/collectives/all_gather_dynamic_slice_simplifier.h"

#include <optional>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_opt_utils.h"
#include "xla/service/hlo_module_config.h"

namespace xla {
bool AllGatherDynamicSliceSimplifier::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (HloPredicateIsNotOp<HloOpcode::kDynamicSlice>(instruction)) {
    return false;
  }

  HloDynamicSliceInstruction* dynamic_slice =
      Cast<HloDynamicSliceInstruction>(instruction);
  HloInstruction* operand = dynamic_slice->mutable_operand(0);

  // Check if the operand is a reshape or all-gather instruction
  bool is_reshape = HloPredicateIsOp<HloOpcode::kReshape>(operand);
  bool is_all_gather = HloPredicateIsOp<HloOpcode::kAllGather>(operand);

  if (!is_reshape && !is_all_gather) {
    return false;
  }

  if (is_reshape && operand->operand(0)->opcode() != HloOpcode::kAllGather) {
    return false;
  }

  const HloModuleConfig& config = instruction->GetModule()->config();
  HloAllGatherInstruction* all_gather =
      is_reshape ? Cast<HloAllGatherInstruction>(operand->mutable_operand(0))
                 : Cast<HloAllGatherInstruction>(operand);

  std::optional<ReduceScatterSpec> spec = AllGatherDynamicSliceCancellation(
      all_gather, config.num_partitions(), config.replica_count(),
      config_.allow_multiple_split_dims, config_.allow_intervening_reshape,
      config_.min_rank, HloPredicateIsOp<HloOpcode::kPartitionId>,
      HloPredicateIsOp<HloOpcode::kReplicaId>,
      config_.allow_intervening_bitcast, config_.allow_multiple_users);

  return spec.has_value() &&
         spec->split_dim == all_gather->all_gather_dimension();
}

absl::StatusOr<HloInstruction*>
AllGatherDynamicSliceSimplifier::ExpandInstruction(
    HloInstruction* instruction) {
  HloDynamicSliceInstruction* dynamic_slice =
      Cast<HloDynamicSliceInstruction>(instruction);
  HloInstruction* operand = dynamic_slice->mutable_operand(0);

  if (HloPredicateIsNotOp<HloOpcode::kReshape>(operand)) {
    // dynamic-slice(all-gather) case
    return operand->mutable_operand(0);
  }

  // dynamic-slice(reshape(all-gather)) case
  HloReshapeInstruction* reshape = Cast<HloReshapeInstruction>(operand);
  HloAllGatherInstruction* all_gather =
      Cast<HloAllGatherInstruction>(reshape->mutable_operand(0));
  HloInstruction* all_gather_input = all_gather->mutable_operand(0);

  auto* new_reshape = instruction->parent()->AddInstruction(
      HloInstruction::CreateReshape(dynamic_slice->shape(), all_gather_input));
  return new_reshape;
}

}  // namespace xla
