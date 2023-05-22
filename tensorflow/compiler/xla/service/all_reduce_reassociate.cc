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

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/all_reduce_key.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/errors.h"

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

// Look-through some formatting operations that might be in front of the
// all-reduces we want to reassociate. Making sure the chain only has 1 user
// throughout.
HloAllReduceInstruction* LookThroughForAllReduce(
    HloInstruction* instr, const Literal& reduction_identity) {
  while (instr->opcode() != HloOpcode::kAllReduce) {
    if (instr->user_count() != 1) {
      return nullptr;
    }
    if (instr->opcode() != HloOpcode::kReshape &&
        instr->opcode() != HloOpcode::kPad &&
        instr->opcode() != HloOpcode::kSlice) {
      return nullptr;
    }
    if (instr->opcode() == HloOpcode::kPad) {
      if (!instr->operand(1)->IsConstant()) {
        return nullptr;
      }
      if (instr->operand(1)->literal() != reduction_identity) {
        return nullptr;
      }
    }
    instr = instr->mutable_operand(0);
  }
  if (instr->user_count() != 1) {
    return nullptr;
  }
  return Cast<HloAllReduceInstruction>(instr);
}

// Because we can look through pads its possible that reassociating the
// all-reduce makes us reduce over more than the sum of the two unpadded
// individual all-reduces. Check that's not the case.
bool ReassociateAllReduceIsProfitable(HloInstruction* ar0, HloInstruction* ar1,
                                      HloInstruction* reassociated_inst) {
  int64_t pre_reassociated_size = ShapeUtil::ElementsIn(ar0->shape());
  if (ar0 != ar1) {
    pre_reassociated_size += ShapeUtil::ElementsIn(ar1->shape());
  }
  return pre_reassociated_size >=
         ShapeUtil::ElementsIn(reassociated_inst->shape());
}

}  // namespace

StatusOr<bool> AllReduceReassociate::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceReassociate because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      // Check if the instruction we want to reassociate with will match any
      // valid all-reduce reduction function. Save the ReductionKind object for
      // later.
      std::optional<ReductionKind> kind = MatchReductionInstruction(inst);
      if (!kind) {
        continue;
      }
      std::optional<Literal> reduction_identity =
          GetReductionIdentity(*kind, inst->shape().element_type());
      // Unsupported reduction type.
      if (!reduction_identity) {
        continue;
      }
      // Find LHS all-reduce.
      HloAllReduceInstruction* ar0 = LookThroughForAllReduce(
          inst->mutable_operand(0), *reduction_identity);
      if (ar0 == nullptr) {
        continue;
      }
      // Find RHS all-reduce.
      HloAllReduceInstruction* ar1 = LookThroughForAllReduce(
          inst->mutable_operand(1), *reduction_identity);
      if (ar1 == nullptr) {
        continue;
      }
      if (!inst->shape().IsArray()) {
        continue;
      }
      // Because we look through pads it might not be profitable to actually
      // reassociate if reassociating makes us all-reduce more values.
      if (!ReassociateAllReduceIsProfitable(ar0, ar1, inst)) {
        continue;
      }
      if (!AreCompatible(ar0, ar1, *kind)) {
        VLOG(2) << "All-Reduce operations are not compatible, skipping";
        continue;
      }
      VLOG(2) << "Reassociated:";
      VLOG(2) << "\tAR0: " << ar0->opcode();
      VLOG(2) << "\tAR1: " << ar1->opcode();

      // Found pattern op(ar(x), ar(y)). Transform it into ar(op(x,y)).
      TF_RETURN_IF_ERROR(ar0->ReplaceAllUsesWith(ar0->mutable_operand(0)));
      TF_RETURN_IF_ERROR(ar1->ReplaceAllUsesWith(ar1->mutable_operand(0)));
      auto op_users = inst->users();
      HloInstruction* new_ar = computation->AddInstruction(
          ar0->CloneWithNewOperands(inst->shape(), {inst}));

      // Do not reuse channel_id from the existing instruction.
      if (new_ar->channel_id()) {
        new_ar->set_channel_id(next_channel_id++);
      }

      TF_RETURN_IF_ERROR(inst->ReplaceUsesWith(op_users, new_ar));
      // Note that RemoveInstructionAndUnusedOperands may not remove the 2
      // all-reduce operands of `inst` if they are not safe to remove otherwise,
      // so manually these instructions.
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
