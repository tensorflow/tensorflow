/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/reduce_scatter_reassociate.h"

#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/all_reduce_key.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

// Returns if the given reduce-scatter instructions are compatible with each
// other. Note that since the given reduce-scatter instructions are connected
// to another instruction by a direct data flow edge, they must belong to the
// same domain. As a result, we don't need to include any domain information
// in the AllReduceKey to check compatibility.
//
// Note: AllReduceKey supports ReduceScatter as well.

bool AreCompatible(const HloReduceScatterInstruction *rs0,
                   const HloReduceScatterInstruction *rs1,
                   ReductionKind op_kind) {
  std::optional<AllReduceKey> key0 = GetAllReduceKey(rs0);
  std::optional<AllReduceKey> key1 = GetAllReduceKey(rs1);
  auto kind0 = MatchReductionComputation(rs0->to_apply());
  auto dims_match = rs0->scatter_dimension() == rs1->scatter_dimension();
  return key0 && key1 && kind0 && *key0 == *key1 && kind0 == op_kind &&
         dims_match;
}

}  // namespace

absl::StatusOr<bool> ReduceScatterReassociate::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  if (hlo_query::ContainsLayoutConstrainedCollective(
          *module, HloOpcode::kReduceScatter)) {
    VLOG(1)
        << "Skip ReduceScatterReassociate because the module contains reduce-"
           "scatter with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      std::optional<ReductionKind> kind = MatchReductionInstruction(inst);
      if (!kind || inst->operand(0)->opcode() != HloOpcode::kReduceScatter ||
          inst->operand(1)->opcode() != HloOpcode::kReduceScatter ||
          !inst->shape().IsArray()) {
        continue;
      }

      auto *rs0 = Cast<HloReduceScatterInstruction>(inst->mutable_operand(0));
      auto *rs1 = Cast<HloReduceScatterInstruction>(inst->mutable_operand(1));
      if (!AreCompatible(rs0, rs1, *kind)) {
        VLOG(2) << "Reduce-Scatter operations are not compatible, skipping";
        continue;
      }

      if (rs0->user_count() != 1 || rs1->user_count() != 1) {
        VLOG(2) << "Reduce-Scatter operations have > 1 users";
        continue;
      }
      TF_ASSIGN_OR_RETURN(auto rs0_annotation, GetSchedulingAnnotation(rs0));
      TF_ASSIGN_OR_RETURN(auto rs1_annotation, GetSchedulingAnnotation(rs1));
      if (rs0_annotation.has_value() && rs1_annotation.has_value() &&
          *rs0_annotation != *rs1_annotation) {
        VLOG(2) << "If two reduce scatters have different scheduling group do "
                   "not merge";
        continue;
      }

      // Found pattern op(rs(x), rs(y)). Transform it into rs(op(x,y)).
      HloInstruction *new_op =
          computation->AddInstruction(inst->CloneWithNewOperands(
              rs0->mutable_operand(0)->shape(),
              {rs0->mutable_operand(0), rs1->mutable_operand(0)}));
      HloInstruction *new_rs = computation->AddInstruction(
          rs0->CloneWithNewOperands(inst->shape(), {new_op}));
      // In case only one of the two instructions had a scheduling annotation,
      // delete the potential annotation.
      if (rs0_annotation.has_value() ^ rs1_annotation.has_value()) {
        RemoveSchedulingAnnotation(new_rs);
      }

      // Do not reuse channel_id from the existing instruction.
      if (new_rs->channel_id()) {
        new_rs->set_channel_id(next_channel_id++);
      }

      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(new_rs));
      // Note that RemoveInstructionAndUnusedOperands may not remove the 2
      // reduce-scatter operands of `inst` if they are not safe to remove
      // otherwise, so manually these instructions.
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(inst));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(rs0));
      if (rs0 != rs1) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(rs1));
      }
      changed = true;
    }
  }

  return changed;
}

}  // namespace xla
