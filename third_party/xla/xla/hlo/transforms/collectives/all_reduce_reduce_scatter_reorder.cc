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

#include "xla/hlo/transforms/collectives/all_reduce_reduce_scatter_reorder.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"

namespace xla {

namespace {
// Returns true if:
// 1. `ar` is a single-operand all-reduce without control dependencies and with
//    a single user.
// 2. The user `rs` is a single-operand reduce-scatter without control
//    dependencies.
// 3. `ar` and `rs` share the same recognized reduction kind and collective
//    group mode.
// 4. `ar` and `rs` have non-empty, orthogonal replica groups with matching
//    group sizes.
bool IsAllReduceReduceScatter(const HloInstruction* ar) {
  if (ar->opcode() != HloOpcode::kAllReduce || ar->user_count() != 1 ||
      ar->operand_count() != 1 || ar->HasControlDependencies()) {
    return false;
  }
  const HloInstruction* rs = ar->users().front();
  const auto ar_reduction = MatchReductionComputation(ar->to_apply());
  if (rs->opcode() != HloOpcode::kReduceScatter || rs->operand_count() != 1 ||
      rs->HasControlDependencies() || ar->replica_groups().empty() ||
      rs->replica_groups().empty() || !ar_reduction.has_value() ||
      ar_reduction != MatchReductionComputation(rs->to_apply())) {
    return false;
  }
  absl::StatusOr<CollectiveOpGroupMode> ar_mode = GetCollectiveOpGroupMode(ar);
  if (!ar_mode.ok() || ar_mode != GetCollectiveOpGroupMode(rs) ||
      (*ar_mode == CollectiveOpGroupMode::
                       COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA_AND_PARTITION &&
       ar->GetModule()->config().num_partitions() > 1)) {
    return false;
  }
  for (const ReplicaGroup& g : ar->replica_groups()) {
    if (g.replica_ids_size() != rs->replica_groups().size()) {
      return false;
    }
  }
  for (const ReplicaGroup& g : rs->replica_groups()) {
    if (g.replica_ids_size() != ar->replica_groups().size()) {
      return false;
    }
  }
  return ReplicaGroupsOrthogonal(ar->replica_groups(), rs->replica_groups());
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
  ABSL_RETURN_IF_ERROR(old_rs->ReplaceUsesWith(old_rs->users(), new_ar));

  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(old_rs));
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(old_ar));
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
        ABSL_RETURN_IF_ERROR(ReorderAllReduceReduceScatter(inst));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
