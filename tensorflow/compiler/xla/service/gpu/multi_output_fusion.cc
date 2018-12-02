/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"

#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <list>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

GpuMultiOutputFusion::GpuMultiOutputFusion() : MultiOutputFusion(INT64_MAX) {}

bool GpuMultiOutputFusion::ShapesCompatibleForFusion(HloInstruction* instr1,
                                                     HloInstruction* instr2) {
  return ShapesCompatibleForMultiOutputFusion(*instr1, *instr2);
}

bool GpuMultiOutputFusion::IsFusible(HloInstruction* instr) {
  // We can fuse reduces and loop fusions. Elementwise instructions can be fused
  // with any other instruction.
  // TODO(b/112957171): This should use the same isFusible logic as
  // instruction_fusion.
  return instr->IsFusible() &&
         (IsInputFusibleReduction(*instr) ||
          (instr->opcode() == HloOpcode::kFusion &&
           instr->fusion_kind() == HloInstruction::FusionKind::kLoop) ||
          instr->IsElementwise());
}

int64 GpuMultiOutputFusion::GetProfit(HloInstruction* instr1,
                                      HloInstruction* instr2) {
  absl::flat_hash_set<HloInstruction*> in_list;
  for (auto instr : instr1->operands()) {
    if (!IsProfitableOperand(instr)) {
      continue;
    }
    in_list.insert(instr);
  }
  int64 profit = 0;
  for (auto instr : instr2->operands()) {
    if (!IsProfitableOperand(instr) || in_list.count(instr) == 0) {
      continue;
    }
    profit += ShapeUtil::ByteSizeOf(instr->shape());
  }
  VLOG(2) << "Fusing instr1=" << instr1->name() << " instr2=" << instr2->name()
          << ", the profit is =" << profit;
  return profit;
}

bool GpuMultiOutputFusion::LegalToFuse(HloInstruction* instr1,
                                       HloInstruction* instr2) {
  if (!MultiOutputFusion::LegalToFuse(instr1, instr2)) {
    return false;
  }

  // If we're fusing fusions only do it if the fusion kind matches. Loop fusions
  // merge into bigger loop fusions and input (reduce) fusions become fusions
  // with multiple reduce outputs. We could fuse reduce and loop fusions
  // together too (the result being an input fusion) if we find cases where this
  // improves things. Also disable fusing standalone input-fusible reduces into
  // loop fusions.
  CHECK(instr1->opcode() == HloOpcode::kFusion);
  if ((instr2->opcode() == HloOpcode::kFusion &&
       instr1->fusion_kind() != instr2->fusion_kind()) ||
      (IsReductionToVector(*instr2) &&
       instr1->fusion_kind() == HloInstruction::FusionKind::kLoop)) {
    return false;
  }

  // The emitter only supports in-place DUS for fusions with a single DUS at the
  // root. Don't sibling fuse DUS for now.
  // TODO(b/119178699): Multi-output fusing DUS can improve performance if we
  // share the input and output buffers and add support to the emitter.
  if (instr1->fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice ||
      (instr2->opcode() == HloOpcode::kFusion &&
       instr2->fused_expression_root()->opcode() ==
           HloOpcode::kDynamicUpdateSlice)) {
    return false;
  }

  // Do this check last, as it may be expensive.
  return !GpuInstructionFusion::FusionWouldBeTooLarge(instr1, instr2);
}

bool GpuMultiOutputFusion::DoProducerConsumerMultiOutputFusion() {
  bool changed = false;
  RecomputeReachability();

  absl::flat_hash_set<HloInstruction*> to_fuse;
  // Keep a list of the instructions to fuse after making all the fusion
  // decisions. We first aggressively add instructions to potential_fusion_list,
  // then filter out instructions that will be no longer fusible because of
  // reachability change. This avoids recalculating reachability on a large set
  // of instructions.
  std::vector<std::pair<HloInstruction*, HloInstruction*>>
      potential_fusion_list;
  std::vector<std::pair<HloInstruction*, HloInstruction*>> fusion_list;
  std::vector<HloInstruction*> instrs_to_update_reachability;

  // For each reduce or reduce multi-output fusion, try to fuse it with loop
  // fusions operands.
  for (HloInstruction* consumer : computation()->MakeInstructionPostOrder()) {
    if (consumer->user_count() == 0) {
      VLOG(3) << consumer->name() << " has no users.";
      continue;
    }
    if (!IsInputFusibleReduction(*consumer)) {
      VLOG(3) << consumer->name() << " is not an input-fusible reduction.";
      continue;
    }
    VLOG(3) << consumer->name()
            << " is a fusion candidate. Looking for fuseable operands.";

    auto consumer_operands = consumer->operands();
    for (size_t i = 0; i < consumer_operands.size(); ++i) {
      HloInstruction* producer = consumer_operands[i];
      if (!producer->IsFusible()) {
        VLOG(3) << producer->name() << " is not fusible.";
        continue;
      }
      // Never multi-output fuse constants.  To the extent that we want to fuse
      // constants, that should be handled by the regular fusion pass.
      if (producer->opcode() == HloOpcode::kConstant) {
        VLOG(3) << producer->name() << " is a constant.";
        continue;
      }
      const bool is_loop_fusion =
          producer->opcode() == HloOpcode::kFusion &&
          producer->fusion_kind() == HloInstruction::FusionKind::kLoop;
      if (!producer->IsElementwise() && !is_loop_fusion) {
        VLOG(3) << producer->name() << " is not a loop fusion.";
        continue;
      }
      if (!ShapesCompatibleForMultiOutputFusion(*producer, *consumer)) {
        VLOG(3) << producer->name() << " has an incompatible shape.";
        continue;
      }
      if (!LayoutsAreReduceInputFusionFriendly(*producer, *consumer)) {
        VLOG(3) << producer->name() << " has inputs with mixed layouts.";
        continue;
      }
      // If we have already decided to fuse this producer, skip it.
      if (ContainsKey(to_fuse, producer)) {
        VLOG(3) << producer->name() << " will be fused with another consumer.";
        continue;
      }
      // Do not fuse a producer if the other operands of the fusion are
      // reachable from the producer, this would create a cycle.
      if (absl::c_any_of(consumer_operands, [&](HloInstruction* operand) {
            return producer != operand &&
                   reachability()->IsReachable(producer, operand);
          })) {
        VLOG(3) << producer->name() << " would introduce a cycle when fused.";
        break;
      }
      to_fuse.insert(producer);
      potential_fusion_list.emplace_back(producer, consumer);
      instrs_to_update_reachability.push_back(producer);
      instrs_to_update_reachability.push_back(consumer);
      break;
    }
  }

  // Filter out pairs that will be no longer fusible because of reachability
  // change.
  for (auto& fusion_pair : potential_fusion_list) {
    HloInstruction* producer = fusion_pair.first;
    HloInstruction* consumer = fusion_pair.second;
    if (!absl::c_any_of(consumer->operands(), [&](HloInstruction* operand) {
          return producer != operand &&
                 reachability()->IsReachable(producer, operand);
        })) {
      UpdateReachability(producer, consumer, instrs_to_update_reachability);
      fusion_list.push_back(fusion_pair);
    }
  }

  for (auto fusions_to_create : fusion_list) {
    HloInstruction* producer = fusions_to_create.first;
    HloInstruction* consumer = fusions_to_create.second;
    if (consumer->opcode() != HloOpcode::kFusion) {
      // Fusing with a reduce (fusion) always results in an input fusion.
      HloInstruction* input_fusion =
          computation()->AddInstruction(HloInstruction::CreateFusion(
              consumer->shape(), HloInstruction::FusionKind::kInput, consumer));
      VLOG(2) << "Fuse producer " << producer->name() << " and its consumer "
              << consumer->name() << " into " << input_fusion->name();
      TF_CHECK_OK(computation()->ReplaceInstruction(consumer, input_fusion));
      if (producer->opcode() == HloOpcode::kFusion) {
        input_fusion->MergeFusionInstructionIntoMultiOutput(producer);
      } else {
        input_fusion->FuseInstructionIntoMultiOutput(producer);
      }
    } else {
      VLOG(2) << "Fuse producer " << producer->name() << " into its consumer "
              << consumer->name();

      if (producer->opcode() == HloOpcode::kFusion) {
        consumer->MergeFusionInstructionIntoMultiOutput(producer);
      } else {
        consumer->FuseInstructionIntoMultiOutput(producer);
      }
    }
    changed = true;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
