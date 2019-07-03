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
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

GpuMultiOutputFusion::GpuMultiOutputFusion() {}

bool GpuMultiOutputFusion::ShapesCompatibleForFusion(HloInstruction* instr1,
                                                     HloInstruction* instr2) {
  return ShapesCompatibleForMultiOutputFusion(*instr1, *instr2);
}

bool GpuMultiOutputFusion::IsFusible(HloInstruction* instr) {
  return IsFusibleAsMultiOutputFusionRoot(*instr);
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
    if (!IsProfitableOperand(instr) || !in_list.contains(instr)) {
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
      (IsReductionFromOrToContiguousDimensions(*instr2) &&
       instr1->IsLoopFusion())) {
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
  return !FusionWouldBeTooLarge(*instr1, *instr2);
}

bool GpuMultiOutputFusion::DoProducerConsumerMultiOutputFusion() {
  bool changed = false;
  RecomputeReachability();
  std::vector<HloInstruction*> defs_before_uses =
      computation()->MakeInstructionPostOrder();

  while (!defs_before_uses.empty()) {
    // Traverse the HLO in uses-before-defs order by removing instruction from
    // the back of the vector.
    HloInstruction* producer = defs_before_uses.back();
    defs_before_uses.pop_back();
    for (HloInstruction* consumer : producer->users()) {
      VLOG(3) << "Looking at producer " << producer->name()
              << " and its consumer " << consumer->name();
      // TODO(b/136623068): Use IsFusibleAsMultiOutputFusionRoot(...) to lift
      // the restriction to input-fusible reductions.
      if (!IsInputFusibleReduction(*consumer)) {
        VLOG(3) << "Consumer " << consumer->name()
                << " is not an input-fusible reduction.";
        continue;
      }
      if (!IsProducerConsumerMultiOutputFusible(*producer, *consumer)) {
        VLOG(3) << producer->name() << " and " << consumer->name()
                << " are not fusible.";
        continue;
      }
      // Never multi-output fuse constants.  To the extent that we want to fuse
      // constants, that should be handled by the regular fusion pass.
      if (producer->opcode() == HloOpcode::kConstant) {
        VLOG(3) << producer->name() << " is a constant.";
        continue;
      }
      // Do not fuse a producer if the other operands of the fusion are
      // reachable from the producer, this would create a cycle.
      if (absl::c_any_of(
              consumer->operands(), [&](const HloInstruction* operand) {
                // If a get-tuple-elment instruction is not in the reachability
                // map, it has been created by fusion in this pass. Simply move
                // on to its operand, which is in the reachability map.
                if (!reachability()->IsPresent(operand) &&
                    operand->opcode() == HloOpcode::kGetTupleElement) {
                  operand = operand->operand(0);
                }
                CHECK(reachability()->IsPresent(operand) &&
                      reachability()->IsPresent(producer))
                    << "Reachability map is incomplete. This should never "
                       "happen.";
                return producer != operand &&
                       reachability()->IsReachable(producer, operand);
              })) {
        VLOG(3) << producer->name() << " would introduce a cycle when fused.";
        continue;
      }
      if (FusionWouldBeTooLarge(*producer, *consumer)) {
        VLOG(3) << producer->name() << " and " << consumer->name()
                << " would be too large of a fusion.";
        continue;
      }
      if (consumer->opcode() != HloOpcode::kFusion) {
        HloInstruction* input_fusion =
            computation()->AddInstruction(HloInstruction::CreateFusion(
                consumer->shape(), ChooseFusionKind(*producer, *consumer),
                consumer));
        VLOG(2) << "Fuse producer " << producer->name() << " and its consumer "
                << consumer->name() << " into " << input_fusion->name();
        reachability()->Replace(consumer, input_fusion);
        TF_CHECK_OK(computation()->ReplaceInstruction(consumer, input_fusion));
        if (producer->opcode() == HloOpcode::kFusion) {
          input_fusion->MergeFusionInstructionIntoMultiOutput(producer);
        } else {
          input_fusion->FuseInstructionIntoMultiOutput(producer);
          CHECK_EQ(0, producer->user_count());
          TF_CHECK_OK(computation()->RemoveInstruction(producer));
        }
      } else {
        VLOG(2) << "Fuse producer " << producer->name() << " into its consumer "
                << consumer->name();
        if (producer->opcode() == HloOpcode::kFusion) {
          consumer->MergeFusionInstructionIntoMultiOutput(producer);
        } else {
          consumer->FuseInstructionIntoMultiOutput(producer);
          CHECK_EQ(0, producer->user_count());
          TF_CHECK_OK(computation()->RemoveInstruction(producer));
        }
      }
      changed = true;
      break;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
