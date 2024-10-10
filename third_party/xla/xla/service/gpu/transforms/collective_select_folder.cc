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

#include "xla/service/gpu/transforms/collective_select_folder.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using SourceTargetPair = std::pair<int64_t, int64_t>;
using SourceTargetPairs = std::vector<SourceTargetPair>;

struct FoldableSelect {
  Comparison::Direction cmp_direction;
  int64_t constant_id;
  CollectiveOpGroupMode collective_mode;
  HloInstruction* true_operand;
  HloInstruction* false_operand;
};

// Returns handy references to %constant, %true_operand, %false_operand of the
//   `select(broadcast(compare(current_id, constant)), true_operand,
//       false_operand)`
// or
//    select(compare(current_id, constant), true_operand, false_operand)`
std::optional<FoldableSelect> MatchFoldableSelect(HloInstruction* select) {
  if (select->opcode() != HloOpcode::kSelect) {
    return std::nullopt;
  }

  // Select may have broadcast.
  const HloInstruction* compare_candidate = select->operand(0);
  if (compare_candidate->opcode() != HloOpcode::kCompare) {
    compare_candidate = compare_candidate->operand(0);
  }
  if (compare_candidate->opcode() != HloOpcode::kCompare) {
    return std::nullopt;
  }

  const HloCompareInstruction* compare =
      DynCast<HloCompareInstruction>(compare_candidate);

  const HloInstruction* id_op = compare->operand(0);
  CollectiveOpGroupMode mode;
  if (id_op->opcode() == HloOpcode::kReplicaId) {
    mode = CollectiveOpGroupMode::kCrossReplica;
  } else if (id_op->opcode() == HloOpcode::kPartitionId) {
    mode = CollectiveOpGroupMode::kCrossPartition;
  } else {
    return std::nullopt;
  }

  if (compare->operand(1)->opcode() != HloOpcode::kConstant) {
    return std::nullopt;
  }

  int64_t id_value =
      compare->operand(1)->literal().GetFirstInteger().value_or(-1);

  return FoldableSelect{compare->direction(), id_value, mode,
                        select->mutable_operand(1), select->mutable_operand(2)};
}

std::optional<bool> StaticallyEvaluatePredicateForAllSourceIDs(
    FoldableSelect select_match, SourceTargetPairs pairs) {
  // If there are no pairs, the predicate is undefined.
  auto it = pairs.begin();
  if (it == pairs.end()) return std::nullopt;

  // Evaluate the select predicate for the first source target pair.
  assert(select_match.cmp_direction == Comparison::Direction::kEq ||
         select_match.cmp_direction == Comparison::Direction::kNe);
  auto predicate = [select_match](const SourceTargetPair& pair) {
    int64_t src_id = pair.first;
    return select_match.cmp_direction == Comparison::Direction::kEq
               ? src_id == select_match.constant_id
               : src_id != select_match.constant_id;
  };
  bool result_candidate = predicate(*it++);

  // Check that the result is the same for all source target pairs. If not,
  // we have a contradiction and cannot statically evaluate the predicate. We
  // return std::nullopt in this case.
  while (it != pairs.end()) {
    if (result_candidate != predicate(*it++)) return std::nullopt;
  }

  // The predicate statically evaluates to the same value for all source target
  // pairs.
  return result_candidate;
}

// Recognizes the pattern and update if applicable.
absl::StatusOr<bool> TryFoldColectivePermuteOfSelect(HloInstruction* inst) {
  // Root op must be a collective-permute.
  HloCollectivePermuteInstruction* cp =
      DynCast<HloCollectivePermuteInstruction>(inst);
  if (!cp) return false;

  // Operand must be a foldable select, i.e. a select op that this pass'
  // analysis supports.
  std::optional<FoldableSelect> select_match =
      MatchFoldableSelect(inst->mutable_operand(0));
  if (!select_match) return false;

  // We have to maintain integrity of relationship between the predicate, which
  // is based on partition or replica ID, and the collevtive mode of the
  // collective-permute op.
  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode collective_mode,
      GetCollectiveOpGroupMode(cp->channel_id().has_value(),
                               /*use_global_device_ids=*/std::nullopt));
  if (collective_mode != select_match->collective_mode) return false;

  // We can only actually fold the select if we can evaluate the predicate
  // statically to a known value for all relevant source IDs.
  std::optional<bool> predicate_value =
      StaticallyEvaluatePredicateForAllSourceIDs(*select_match,
                                                 cp->source_target_pairs());
  if (!predicate_value.has_value()) return false;

  // Fold select and forward the correct operand.
  HloInstruction* new_operand = *predicate_value ? select_match->true_operand
                                                 : select_match->false_operand;
  TF_RETURN_IF_ERROR(cp->ReplaceOperandWith(0, new_operand));
  return true;
}

}  // namespace

absl::StatusOr<bool> CollectiveSelectFolder::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations()) {
    for (HloInstruction* inst : comp->instructions()) {
      TF_ASSIGN_OR_RETURN(bool local_changed,
                          TryFoldColectivePermuteOfSelect(inst));
      changed |= local_changed;
    }
  }
  return changed;
}

}  // namespace xla
