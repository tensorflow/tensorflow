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

#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"

#include "tensorflow/compiler/xla/service/all_reduce_key.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace {

// Returns if the given all reduce instructions are compatible with each other.
// Note that since the given all-reduce instructions are connected to another
// instruction by a direct data flow edge, they must belong to the same domain.
// As a result, we don't need to include any domain information in the
// AllReduceKey to check compatibility.
bool AreCompatible(const HloAllReduceInstruction *ar0,
                   const HloAllReduceInstruction *ar1, ReductionKind op_kind) {
  std::optional<AllReduceKey> key0 = GetAllReduceKey(ar0);
  std::optional<AllReduceKey> key1 = GetAllReduceKey(ar1);
  auto kind0 = MatchReductionComputation(ar0->to_apply());
  return key0 && key1 && kind0 && *key0 == *key1 && kind0 == op_kind;
}

}  // namespace

StatusOr<bool> AllReduceReassociate::Run(HloModule *module) {
  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceReassociate because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations()) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      std::optional<ReductionKind> kind = MatchReductionInstruction(inst);
      if (!kind || inst->operand(0)->opcode() != HloOpcode::kAllReduce ||
          inst->operand(1)->opcode() != HloOpcode::kAllReduce ||
          !inst->shape().IsArray()) {
        continue;
      }

      auto *ar0 = Cast<HloAllReduceInstruction>(inst->mutable_operand(0));
      auto *ar1 = Cast<HloAllReduceInstruction>(inst->mutable_operand(1));
      if (!AreCompatible(ar0, ar1, *kind)) {
        VLOG(2) << "All-Reduce operations are not compatible, skipping";
        continue;
      }

      if (ar0->user_count() != 1 || ar1->user_count() != 1) {
        VLOG(2) << "All-Reduce operations have > 1 users";
        continue;
      }

      // Found pattern op(ar(x), ar(y)). Transform it into ar(op(x,y)).
      HloInstruction *new_op = computation->AddInstruction(
          inst->CloneWithNewOperands(inst->shape(), {ar0->mutable_operand(0),
                                                     ar1->mutable_operand(0)}));
      HloInstruction *new_ar = computation->AddInstruction(
          ar0->CloneWithNewOperands(inst->shape(), {new_op}));

      // Do not reuse channel_id from the existing instruction.
      if (new_ar->channel_id()) {
        new_ar->set_channel_id(next_channel_id++);
      }

      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(new_ar));
      // Note that RemoveInstructionAndUnusedOperands may not remove the 2
      // all-reduce operands of `inst` if they are not safe to remove otherwise,
      // so manually these instructions.
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(inst));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar0));
      if (ar0 != ar1) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar1));
      }
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
