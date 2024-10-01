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
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using SourceTargetPair = std::pair<int64_t, int64_t>;
using SourceTargetPairs = std::vector<SourceTargetPair>;

struct SelectPredInfo {
  int64_t constant;
  Comparison::Direction direction;
  HloOpcode device_id_type;  // kReplicaId or kPartitionId
  HloInstruction* true_operand;
  HloInstruction* false_operand;
};

// Returns handy references to %constant, %true_operand, %false_operand of the
// select(broadcast(compare(current_device_id, constant)), true_operand,
// false_operand)
// or
// select(compare(current_device_id, constant), true_operand,
// false_operand)
std::optional<SelectPredInfo> GetPredSelectInfo(HloInstruction* select) {
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

  if ((compare->operand(0)->opcode() != HloOpcode::kReplicaId &&
       compare->operand(0)->opcode() != HloOpcode::kPartitionId) ||
      compare->operand(1)->opcode() != HloOpcode::kConstant) {
    return std::nullopt;
  }

  int64_t id_value =
      compare->operand(1)->literal().GetFirstInteger().value_or(-1);

  return SelectPredInfo{id_value, compare->direction(),
                        compare->operand(0)->opcode(),
                        select->mutable_operand(1), select->mutable_operand(2)};
}

bool IsUniqueSource(int64_t device_id, const SourceTargetPairs& pairs) {
  if (pairs.size() == 1 && pairs[0].first == device_id) return true;
  return false;
}

bool IsNotPresentInSource(int64_t device_id, const SourceTargetPairs& pairs) {
  return absl::c_none_of(
      pairs, [device_id](const auto& pair) { return pair.first == device_id; });
}

inline absl::StatusOr<bool> update(HloInstruction* cp, HloInstruction* data) {
  TF_RETURN_IF_ERROR(cp->ReplaceOperandWith(0, data));
  return true;
}

// We have to maintain integrity of relationship between partition/replica
// and collective-permute's channel_id.
// That is we can only fold select when
// 1. cp has channel_id and condition is based on partition_id
// 2. cp has no channel_id and condition is based on replica_id
// See enum class CollectiveOpGroupMode for details.
bool IsShardingConsistent(HloCollectivePermuteInstruction* cp,
                          HloOpcode device_id_type) {
  auto id = cp->channel_id();
  return (device_id_type == HloOpcode::kPartitionId && id.has_value()) ||
         (device_id_type == HloOpcode::kReplicaId && !id.has_value());
}

// Recognizes the pattern and update if applicable.
absl::StatusOr<bool> TryFoldSelect(HloInstruction* in) {
  if (in->opcode() != HloOpcode::kCollectivePermute) return false;
  auto select_info_opt = GetPredSelectInfo(in->mutable_operand(0));
  if (!select_info_opt.has_value()) return false;
  auto select_info = select_info_opt.value();

  HloCollectivePermuteInstruction* cp =
      Cast<HloCollectivePermuteInstruction>(in);
  if (!IsShardingConsistent(cp, select_info.device_id_type)) return false;

  int64_t device_id = select_info.constant;
  SourceTargetPairs pairs = cp->source_target_pairs();

  if (select_info.direction == Comparison::Direction::kEq) {
    if (IsUniqueSource(device_id, pairs)) {
      return update(cp, select_info.true_operand);
    } else if (IsNotPresentInSource(device_id, pairs)) {
      return update(cp, select_info.false_operand);
    }
  }

  if (select_info.direction == Comparison::Direction::kNe) {
    if (IsNotPresentInSource(device_id, pairs)) {
      return update(cp, select_info.true_operand);
    } else if (IsUniqueSource(device_id, pairs)) {
      return update(cp, select_info.false_operand);
    }
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> CollectiveSelectFolder::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, TryFoldSelect(instruction));
      changed |= local_changed;
    }
  }
  return changed;
}

}  // namespace xla
