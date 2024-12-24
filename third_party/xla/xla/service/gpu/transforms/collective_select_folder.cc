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

#include "absl/algorithm/container.h"
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
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
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

const HloInstruction* FindInnerScalarOp(const HloInstruction* inst) {
  while (inst->opcode() == HloOpcode::kConvert ||
         inst->opcode() == HloOpcode::kBroadcast) {
    inst = inst->operand(0);
  }
  return inst;
}

// Matches foldable select ops that we can analyse and returns handy references
// to %constant, %true_operand, %false_operand of the op. Matches, e.g.,
//
// ```
// select(
//     broadcast(compare(convert(partition-id()), constant)),
//     true_operand,
//     false_operand)
// ```
//
// or
//
// ```
// select(
//     compare(replica-id(), constant),
//     true_operand,
//     false_operand)
// ```
std::optional<FoldableSelect> MatchFoldableSelect(HloInstruction* select) {
  if (HloPredicateIsNotOp<HloOpcode::kSelect>(select)) {
    return std::nullopt;
  }

  // Match select predicate.
  const HloInstruction* predicate_candidate =
      FindInnerScalarOp(select->operand(0));
  const HloCompareInstruction* compare =
      DynCast<HloCompareInstruction>(predicate_candidate);
  if (compare == nullptr) {
    return std::nullopt;
  }
  if (compare->direction() != Comparison::Direction::kEq &&
      compare->direction() != Comparison::Direction::kNe) {
    return std::nullopt;
  }

  // Find replica-id or partition-id op and constant op, swap if needed.
  const HloInstruction* id_op = FindInnerScalarOp(compare->operand(0));
  const HloInstruction* constant_op = FindInnerScalarOp(compare->operand(1));
  if (HloPredicateIsNotOp<HloOpcode::kConstant>(constant_op)) {
    std::swap(id_op, constant_op);
  }

  // Match replica-id or partition-id.
  CollectiveOpGroupMode collective_mode;
  if (HloPredicateIsOp<HloOpcode::kReplicaId>(id_op)) {
    collective_mode = CollectiveOpGroupMode::kCrossReplica;
  } else if (HloPredicateIsOp<HloOpcode::kPartitionId>(id_op)) {
    collective_mode = CollectiveOpGroupMode::kCrossPartition;
  } else {
    return std::nullopt;
  }

  // Match constant.
  if (HloPredicateIsNotOp<HloOpcode::kConstant>(constant_op) ||
      !ShapeUtil::IsScalar(constant_op->shape())) {
    return std::nullopt;
  }
  std::optional<int64_t> constant_id = constant_op->literal().GetFirstInteger();
  if (!constant_id.has_value()) {
    return std::nullopt;
  }
  return FoldableSelect{compare->direction(), *constant_id, collective_mode,
                        select->mutable_operand(1), select->mutable_operand(2)};
}

std::optional<bool> StaticallyEvaluatePredicateForAllSourceIDs(
    FoldableSelect select_match, SourceTargetPairs pairs) {
  // If there are no pairs, the predicate is undefined.
  if (pairs.empty()) return std::nullopt;

  // Evaluate the select predicate for the first source target pair.
  CHECK(select_match.cmp_direction == Comparison::Direction::kEq ||
        select_match.cmp_direction == Comparison::Direction::kNe);
  auto select_predicate_eval = [&select_match](const SourceTargetPair& pair) {
    int64_t src_id = pair.first;
    return select_match.cmp_direction == Comparison::Direction::kEq
               ? src_id == select_match.constant_id
               : src_id != select_match.constant_id;
  };
  bool result_candidate = select_predicate_eval(pairs.front());

  // Check that the result is the same for all source target pairs. If not,
  // we have a contradiction and cannot statically evaluate the predicate. We
  // return std::nullopt in this case.
  if (!absl::c_all_of(pairs, [&](const SourceTargetPair& it) -> bool {
        return result_candidate == select_predicate_eval(it);
      })) {
    return std::nullopt;
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
  if (cp == nullptr) return false;
  VLOG(3) << "Try folding collective-permute(*) at " << cp->ToShortString();

  // Operand must be a foldable select, i.e. a select op that this pass'
  // analysis supports.
  std::optional<FoldableSelect> select_match =
      MatchFoldableSelect(inst->mutable_operand(0));
  VLOG(3) << (select_match.has_value() ? "Matched" : "Did not match")
          << " foldable select at " << cp->ToShortString();
  if (!select_match.has_value()) return false;

  // We have to maintain integrity of relationship between the predicate, which
  // is based on partition or replica ID, and the collective mode of the
  // collective-permute op.
  TF_ASSIGN_OR_RETURN(
      CollectiveOpGroupMode collective_mode,
      GetCollectiveOpGroupMode(cp->channel_id().has_value(),
                               /*use_global_device_ids=*/std::nullopt));
  bool collective_mode_is_compatible =
      collective_mode == select_match->collective_mode;
  VLOG(3) << "Collective mode "
          << (collective_mode_is_compatible ? "is" : "is not")
          << " compatible with select predicate";
  if (!collective_mode_is_compatible) return false;

  // We can only actually fold the select if we can evaluate the predicate
  // statically to a known value for all relevant source IDs.
  std::optional<bool> predicate_value =
      StaticallyEvaluatePredicateForAllSourceIDs(*select_match,
                                                 cp->source_target_pairs());
  if (!predicate_value.has_value()) {
    VLOG(3) << "Static evaluation of the predicate failed";
    return false;
  }
  VLOG(3) << "Static evaluation of the predicate yields " << *predicate_value;

  // Fold select and forward the correct operand.
  HloInstruction* new_operand = *predicate_value ? select_match->true_operand
                                                 : select_match->false_operand;
  TF_RETURN_IF_ERROR(cp->ReplaceOperandWith(0, new_operand));
  VLOG(3) << "Successfully folded select op away";
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
