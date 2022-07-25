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
#include "tensorflow/compiler/xla/service/collective_decomposer_utils.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
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
  TF_ASSIGN_OR_RETURN(CollectiveOpGroupMode group_mode,
                      GetCollectiveOpGroupMode(ag->channel_id().has_value(),
                                               ag->use_global_device_ids()));
  TF_ASSIGN_OR_RETURN(
      std::vector<HloInstruction*> start_indices,
      CreateStartIndicesForCollectiveDecomposition(
          group_mode, ag->replica_groups(), ag->operand(0)->shape(),
          ag->all_gather_dimension(), comp));

  auto zero = comp->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(ag->shape().element_type())));
  zero = comp->AddInstruction(
      HloInstruction::CreateBroadcast(ag->shape(), zero, {}));

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
  return OkStatus();
}

StatusOr<bool> AllGatherDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto comp : module->MakeNonfusionComputations(execution_threads)) {
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
