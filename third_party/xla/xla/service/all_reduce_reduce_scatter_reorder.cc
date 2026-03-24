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

#include "xla/service/all_reduce_reduce_scatter_reorder.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

namespace {
// Returns true if
// 1. `inst` is an all-reduce instruction.
// 2. `inst` has a single user.
// 3. The user is a reduce-scatter instruction.
// 4. The all-reduce and reduce-scatter have the same reduction type.
bool IsAllReduceReduceScatter(const HloInstruction* ar) {
  if (ar->opcode() != HloOpcode::kAllReduce) {
    return false;
  }
  if (ar->user_count() != 1) {
    return false;
  }
  const HloInstruction* rs = ar->users().front();
  if (rs->opcode() != HloOpcode::kReduceScatter) {
    return false;
  }
  if (MatchReductionComputation(ar->to_apply()) &&
      MatchReductionComputation(ar->to_apply()) !=
          MatchReductionComputation(rs->to_apply())) {
    return false;
  }
  return true;
}

// Before: operand -> old_ar -> old_rs -> users
// After:  operand -> new_rs -> new_ar -> users
//
// old_rs, new_rs, and new_ar share the same shape, while old_ar has a larger
// shape.
absl::Status ReorderAllReduceReduceScatter(HloInstruction* old_ar) {
  HloComputation* computation = old_ar->parent();
  HloInstruction* old_rs = old_ar->users().front();
  HloInstruction* new_rs = computation->AddInstruction(
      old_rs->CloneWithNewOperands(old_rs->shape(), old_ar->operands()));
  HloInstruction* new_ar = computation->AddInstruction(
      old_ar->CloneWithNewOperands(old_rs->shape(), {new_rs}));
  TF_RETURN_IF_ERROR(old_rs->ReplaceUsesWith(old_rs->users(), new_ar));

  TF_RETURN_IF_ERROR(computation->RemoveInstruction(old_rs));
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(old_ar));
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<bool> AllReduceReduceScatterReorder::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (IsAllReduceReduceScatter(inst)) {
        TF_RETURN_IF_ERROR(ReorderAllReduceReduceScatter(inst));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
