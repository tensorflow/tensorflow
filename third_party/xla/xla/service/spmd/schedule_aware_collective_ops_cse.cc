/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/spmd/schedule_aware_collective_ops_cse.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

// Returns if an instructions adds only degenerate dimensions to the shape of
// the input, like going from [X,Y] to [1,X,Y,1].
bool IsAddingOnlyDegenerateDimensions(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kBitcast &&
      inst->opcode() != HloOpcode::kReshape) {
    return false;
  }
  const Shape& in_shape = inst->operand(0)->shape();
  const Shape& out_shape = inst->shape();
  return ShapeUtil::ElementsIn(in_shape) == ShapeUtil::ElementsIn(out_shape) &&
         ShapeUtil::DimensionsUnmodifiedByReshape(in_shape, out_shape).size() ==
             in_shape.dimensions_size();
}

// Passthrough reshapes or bitcasts adding only degenerate hdimensions to some
// shape.
const HloInstruction* PassthroughDegenerateAddingReshapes(
    const HloInstruction* inst) {
  while (IsAddingOnlyDegenerateDimensions(inst)) {
    inst = inst->operand(0);
  }
  return inst;
}

bool ShouldConsiderSchedule(HloInstruction* hlo) {
  return hlo->opcode() != HloOpcode::kCollectivePermute;
}

HloInstruction* MayConsiderCollective(HloInstruction* hlo, bool for_replicas) {
  auto chan_instr = DynCast<HloChannelInstruction>(hlo);
  if (!chan_instr) {
    return nullptr;
  }
  if (for_replicas == chan_instr->channel_id().has_value()) {
    return nullptr;
  }
  if (hlo->opcode() == HloOpcode::kCollectivePermute) {
    return hlo;
  }
  auto coll = DynCast<HloCollectiveInstruction>(hlo);
  if (!coll) {
    return nullptr;
  }
  if (coll->constrain_layout()) {
    return nullptr;
  }
  if (coll->opcode() == HloOpcode::kAllGather) {
    return coll;
  }
  // Consider broadcast -> dynamic-update-slice -> all-reduce as all-gather.
  if (coll->opcode() == HloOpcode::kAllReduce && coll->shape().IsArray()) {
    auto operand = coll->operand(0);
    return operand->opcode() == HloOpcode::kDynamicUpdateSlice &&
                   operand->operand(0)->opcode() == HloOpcode::kBroadcast
               ? coll
               : nullptr;
  }
  return nullptr;
}

absl::StatusOr<bool> RunOnComputation(HloComputation* comp, bool for_replicas,
                                      int64_t distance_threshold) {
  // We consider estimate the live ranges of all-gathers by comparing their
  // users' distance to the root, e.g., height.
  bool changed = false;
  absl::flat_hash_map<const HloInstruction*, int64_t> height;
  auto ordered_hlos = comp->MakeInstructionPostOrder();
  int64_t max_height = 0;
  for (auto it = ordered_hlos.rbegin(); it != ordered_hlos.rend(); ++it) {
    auto hlo = *it;
    int64_t h = 0;
    for (auto user : hlo->users()) {
      h = std::max(h, height[user]) + 1;
    }
    max_height = std::max(max_height, h);
    height[hlo] = h;
  }

  auto lowest_user_height = [&](const HloInstruction* hlo) {
    int64_t lowest = height[hlo];
    for (auto user : hlo->users()) {
      lowest = std::min(lowest, height[user]);
    }
    return lowest;
  };

  absl::flat_hash_map<const HloInstruction*, std::vector<HloInstruction*>>
      operand_to_collective;
  for (HloInstruction* hlo : ordered_hlos) {
    HloInstruction* coll = MayConsiderCollective(hlo, for_replicas);
    if (!coll) {
      continue;
    }
    auto& earlier_colls =
        operand_to_collective[PassthroughDegenerateAddingReshapes(
            coll->operand(0))];
    bool found = false;
    int64_t coll_height = height[coll];
    for (HloInstruction* earlier_coll : earlier_colls) {
      if (!ShapeUtil::Equal(earlier_coll->shape(), coll->shape())) {
        continue;
      }
      HloInstruction* coll_operand = coll->mutable_operand(0);
      TF_RETURN_IF_ERROR(
          coll->ReplaceOperandWith(0, earlier_coll->mutable_operand(0)));
      if (!earlier_coll->IdenticalIgnoringChannelIdValues(*coll)) {
        TF_RETURN_IF_ERROR(coll->ReplaceOperandWith(0, coll_operand));
        continue;
      }
      found = true;
      if (ShouldConsiderSchedule(coll) &&
          lowest_user_height(earlier_coll) > coll_height + distance_threshold) {
        TF_RETURN_IF_ERROR(coll->ReplaceOperandWith(0, coll_operand));
        earlier_coll = coll;
        continue;
      }
      changed = true;
      VLOG(1) << "Replacing " << coll->ToString() << " with "
              << earlier_coll->ToString();
      TF_RETURN_IF_ERROR(coll->ReplaceAllUsesWith(earlier_coll));
      break;
    }
    if (!found) {
      earlier_colls.push_back(coll);
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ScheduleAwareCollectiveOpsCSE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto comp : module->computations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(
        auto comp_changed,
        RunOnComputation(comp, for_replicas_, distance_threshold_));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
