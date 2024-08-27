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
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

struct SelectPredInfo {
  const bool is_replica_select;
  const int64_t constant;
  HloInstruction* true_operand;
  HloInstruction* false_operand;
};

// Returns handy references to %constant, %true_operand, %false_operand
// assuming following pattern:
//   %partition = partition-id()
//   %constant = constant(?)
//   %compare = compare(%partition, %constant), direction=EQ
//   %broadcast =  broadcast(%compare), dimensions={}
//   %select = select(%broadcast, %true_operand, %false_operand)
SelectPredInfo GetPredSelectInfo(HloInstruction* select) {
  if (select->opcode() != HloOpcode::kSelect ||
      select->operand(0)->opcode() != HloOpcode::kBroadcast ||
      select->operand(0)->operand(0)->opcode() != HloOpcode::kCompare) {
    return SelectPredInfo(false);
  }

  const HloCompareInstruction* compare =
      Cast<HloCompareInstruction>(select->operand(0)->operand(0));

  bool is_replica_or_partition_compare =
      (compare->operand(0)->opcode() == HloOpcode::kReplicaId ||
       compare->operand(0)->opcode() == HloOpcode::kPartitionId) &&
      compare->direction() == Comparison::Direction::kEq &&
      compare->operand(1)->opcode() == HloOpcode::kConstant;

  if (!is_replica_or_partition_compare) return SelectPredInfo(false);

  int64_t id_value =
      compare->operand(1)->literal().GetFirstInteger().value_or(-1);

  return SelectPredInfo(true, id_value, select->mutable_operand(1),
                        select->mutable_operand(2));
}

// Pattern recognizer for the optimization.
bool IsApplicable(HloInstruction* data_rcv_select) {
  if (data_rcv_select->opcode() != HloOpcode::kSelect ||
      data_rcv_select->operand(1)->opcode() != HloOpcode::kCollectivePermute ||
      data_rcv_select->operand(2)->opcode() != HloOpcode::kCollectivePermute)
    return false;

  SelectPredInfo rcv_data_select_info = GetPredSelectInfo(data_rcv_select);
  if (!rcv_data_select_info.is_replica_select) return false;

  HloCollectivePermuteInstruction* bkw_cp =
      Cast<HloCollectivePermuteInstruction>(rcv_data_select_info.true_operand);

  auto orig_permute_data = bkw_cp->mutable_operand(0);
  // Capture actual data routed based on the select.
  SelectPredInfo snd_data_select_info = GetPredSelectInfo(orig_permute_data);
  if (!snd_data_select_info.is_replica_select) return false;

  // We are optimizing  cp must only have one pair and its send device
  // must be the same as the constant in the compare.
  if (bkw_cp->source_target_pairs().size() != 1 ||
      snd_data_select_info.constant != bkw_cp->source_target_pairs()[0].first)
    return false;

  return true;
}

absl::Status FoldCollectiveSelect(HloComputation* computation,
                                  HloInstruction* rcv_select) {
  auto bkw_cp =
      Cast<HloCollectivePermuteInstruction>(rcv_select->mutable_operand(1));
  auto fwd_cp =
      Cast<HloCollectivePermuteInstruction>(rcv_select->mutable_operand(2));
  auto orig_permute_data = bkw_cp->mutable_operand(0);
  SelectPredInfo cp_select_info = GetPredSelectInfo(orig_permute_data);
  auto bkw_data = cp_select_info.true_operand;
  auto fwd_data = cp_select_info.false_operand;
  TF_RETURN_IF_ERROR(bkw_cp->ReplaceOperandWith(0, bkw_data));
  TF_RETURN_IF_ERROR(fwd_cp->ReplaceOperandWith(0, fwd_data));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CollectiveSelectFolder::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsApplicable(instruction)) {
        TF_RETURN_IF_ERROR(FoldCollectiveSelect(computation, instruction));
        changed = true;
        //  We expect only one such optimization opportunity in a computation
        // TODO (b/359348622): Further generalize selects feeding into pair of
        // collective permutes and handle many such cases in a computation.
        break;
      }
    }
  }
  return changed;
}

}  // namespace xla
