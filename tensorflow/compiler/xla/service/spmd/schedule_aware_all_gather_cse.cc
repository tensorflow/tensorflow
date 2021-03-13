/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/spmd/schedule_aware_all_gather_cse.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/stream_executor/lib/statusor.h"

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
             in_shape.rank();
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

HloCollectiveInstruction* MayConsiderAsAllGather(HloInstruction* hlo,
                                                 bool for_replicas) {
  auto coll = DynCast<HloCollectiveInstruction>(hlo);
  if (!coll) {
    return nullptr;
  }
  if (coll->constrain_layout()) {
    return nullptr;
  }
  if (for_replicas == coll->channel_id().has_value()) {
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

StatusOr<bool> RunOnComputation(HloComputation* comp, bool for_replicas,
                                int64 distance_threshold) {
  // We consider estimate the live ranges of all-gathers by comparing their
  // users' distance to the root, e.g., height.
  bool changed = false;
  absl::flat_hash_map<const HloInstruction*, int64> height;
  auto ordered_hlos = comp->MakeInstructionPostOrder();
  int64 max_height = 0;
  for (auto it = ordered_hlos.rbegin(); it != ordered_hlos.rend(); ++it) {
    auto hlo = *it;
    int64 h = 0;
    for (auto user : hlo->users()) {
      h = std::max(h, height[user]) + 1;
    }
    max_height = std::max(max_height, h);
    height[hlo] = h;
  }

  auto lowest_user_height = [&](const HloInstruction* hlo) {
    int64 lowest = height[hlo];
    for (auto user : hlo->users()) {
      lowest = std::min(lowest, height[user]);
    }
    return lowest;
  };

  absl::flat_hash_map<const HloInstruction*,
                      std::vector<HloCollectiveInstruction*>>
      operand_to_ag;
  for (auto hlo : ordered_hlos) {
    auto ag = MayConsiderAsAllGather(hlo, for_replicas);
    if (!ag) {
      continue;
    }
    auto& earlier_ags =
        operand_to_ag[PassthroughDegenerateAddingReshapes(ag->operand(0))];
    bool found = false;
    int64 ag_height = height[ag];
    for (auto& eag : earlier_ags) {
      if (!ShapeUtil::Equal(eag->shape(), ag->shape())) {
        continue;
      }
      HloInstruction* ag_operand = ag->mutable_operand(0);
      TF_RETURN_IF_ERROR(ag->ReplaceOperandWith(0, eag->mutable_operand(0)));
      if (!eag->IdenticalIgnoringChannelIdValues(*ag)) {
        TF_RETURN_IF_ERROR(ag->ReplaceOperandWith(0, ag_operand));
        continue;
      }
      found = true;
      if (lowest_user_height(eag) > ag_height + distance_threshold) {
        TF_RETURN_IF_ERROR(ag->ReplaceOperandWith(0, ag_operand));
        eag = ag;
        continue;
      }
      changed = true;
      VLOG(1) << "Replacing " << ag->ToString() << " with " << eag->ToString();
      TF_RETURN_IF_ERROR(ag->ReplaceAllUsesWith(eag));
      break;
    }
    if (!found) {
      earlier_ags.push_back(ag);
    }
  }
  return changed;
}

}  // namespace

StatusOr<bool> ScheduleAwareAllGatherCSE::Run(HloModule* module) {
  bool changed = false;
  for (auto comp : module->computations()) {
    TF_ASSIGN_OR_RETURN(
        auto comp_changed,
        RunOnComputation(comp, for_replicas_, distance_threshold_));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
