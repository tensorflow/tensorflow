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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

GpuMultiOutputFusion::GpuMultiOutputFusion() : MultiOutputFusion(INT64_MAX) {}

bool GpuMultiOutputFusion::ShapesCompatibleForFusion(HloInstruction* instr1,
                                                     HloInstruction* instr2) {
  auto get_element_instr =
      [&](const HloInstruction* instr) -> const HloInstruction* {
    const HloInstruction* element_instr = instr;
    if (instr->opcode() == HloOpcode::kFusion) {
      auto fused_expression_root = instr->fused_expression_root();
      if (instr->IsMultiOutputFusion()) {
        // If possible, we want to pick a reduce operand of the fusion root,
        // because it has the most constraints.
        for (const auto* inst : fused_expression_root->operands()) {
          if (inst->opcode() == HloOpcode::kReduce) {
            return inst;
          }
        }
        return fused_expression_root->operands()[0];
      } else {
        element_instr = fused_expression_root;
      }
    }
    return element_instr;
  };

  auto get_element_shape = [&](const HloInstruction* element_instr) {
    // Special handling of kReduce instructions -- the fusion
    // applies to the first operand.
    if (element_instr->opcode() == HloOpcode::kReduce) {
      return element_instr->operand(0)->shape();
    }
    return element_instr->shape();
  };

  // The shapes in all tuple operands should agree, unless it is a reduce.
  // In that case, the operand of the reduce needs to have the same shape
  // as the other tuple operands, but also we need to compare the output
  // shapes of the reduces.
  auto* element_instr_1 = get_element_instr(instr1);
  auto* element_instr_2 = get_element_instr(instr2);
  if (element_instr_1->opcode() == HloOpcode::kReduce &&
      element_instr_2->opcode() == HloOpcode::kReduce &&
      !ShapeUtil::Equal(element_instr_1->shape(), element_instr_2->shape())) {
    return false;
  }
  // The elementwise output shapes must be the same (including layout).
  return ShapeUtil::EqualIgnoringFpPrecision(
      get_element_shape(element_instr_1), get_element_shape(element_instr_2));
}

namespace {
bool IsInputFusibleReduction(HloInstruction* instr) {
  if (instr->IsMultiOutputFusion()) {
    for (const HloInstruction* operand :
         instr->fused_expression_root()->operands()) {
      if (operand->opcode() == HloOpcode::kReduce) {
        CHECK(instr->fusion_kind() == HloInstruction::FusionKind::kInput)
            << " Reduce multi-output fusion " << instr->ToString()
            << " must be an input fusion.";
        return true;
      }
    }
    return false;
  } else if (instr->opcode() == HloOpcode::kFusion) {
    // The loop emitter can handle to-vector reduce fusions. Such reduce
    // fusions have the fusion kind kLoop rather than kInput. We do not fuse
    // to-vector reduce fusions, because the resulting fusions may no longer be
    // supported by loop emitter.
    return IsReductionToVector(*instr->fused_expression_root());
  } else {
    return IsReductionToVector(*instr);
  }
}

// The code emitted for reduction suffers from poor data locality if the layouts
// of input parameters differ. In such situtations it is beneficial not to fuse.
// We consider input params with maximum rank only. Params with smaller ranks
// will be broadcasted and have not been observed to cause data locality issues.
// TODO(b/110927656): Improve reduce emitters to remove this limitation.
bool ReduceFriendlyInputLayouts(HloInstruction* instr) {
  int64 max_rank = 0;
  const Layout* max_rank_layout;
  for (HloInstruction* param : instr->fused_parameters()) {
    if (ShapeUtil::Rank(param->shape()) > max_rank) {
      max_rank = ShapeUtil::Rank(param->shape());
      max_rank_layout = &param->shape().layout();
    }
  }
  return c_all_of(instr->fused_parameters(), [&](HloInstruction* param) {
    return (ShapeUtil::Rank(param->shape()) < max_rank) ||
           (LayoutUtil::Equal(param->shape().layout(), *max_rank_layout));
  });
}

}  // namespace

bool GpuMultiOutputFusion::IsFusible(HloInstruction* instr) {
  // We can fuse reduces and loop fusions.
  return IsInputFusibleReduction(instr) ||
         (instr->opcode() == HloOpcode::kFusion &&
          instr->fusion_kind() == HloInstruction::FusionKind::kLoop);
}

int64 GpuMultiOutputFusion::GetProfit(HloInstruction* instr1,
                                      HloInstruction* instr2) {
  tensorflow::gtl::FlatSet<HloInstruction*> in_list;
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
  // improves things.
  CHECK(instr1->opcode() == HloOpcode::kFusion);
  if (instr2->opcode() == HloOpcode::kFusion) {
    return instr1->fusion_kind() == instr2->fusion_kind();
  }
  return instr1->fusion_kind() != HloInstruction::FusionKind::kLoop;
}

bool GpuMultiOutputFusion::DoProducerConsumerMultiOutputFusion() {
  bool changed = false;
  RecomputeReachability();

  tensorflow::gtl::FlatSet<HloInstruction*> to_fuse;
  // Keep a list of the instructions to fuse after making all the fusion
  // decisions. We first aggressively add instructions to potential_fusion_list,
  // then filter out instructions that will be no longer fusable because of
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
    if (!IsInputFusibleReduction(consumer)) {
      VLOG(3) << consumer->name() << " is not an input-fusable reduction.";
      continue;
    }
    VLOG(3) << consumer->name()
            << " is a fusion candidate. Looking for fuseable operands.";

    auto consumer_operands = consumer->operands();
    for (size_t i = 0; i < consumer_operands.size(); ++i) {
      HloInstruction* producer = consumer_operands[i];
      if (!producer->IsFusable()) {
        VLOG(3) << producer->name() << " is not fusable.";
        continue;
      }
      const bool is_loop_fusion =
          producer->opcode() == HloOpcode::kFusion &&
          producer->fusion_kind() == HloInstruction::FusionKind::kLoop;
      if (!is_loop_fusion) {
        VLOG(3) << producer->name() << " is not a loop fusion.";
        continue;
      }
      if (!ShapesCompatibleForFusion(producer, consumer)) {
        VLOG(3) << producer->name() << " has an incompatible shape.";
        continue;
      }
      if (!ReduceFriendlyInputLayouts(producer)) {
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
      if (c_any_of(consumer_operands, [&](HloInstruction* operand) {
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

  // Filter out pairs that will be no longer fusable because of reachability
  // change.
  for (auto& fusion_pair : potential_fusion_list) {
    HloInstruction* producer = fusion_pair.first;
    HloInstruction* consumer = fusion_pair.second;
    if (!c_any_of(consumer->operands(), [&](HloInstruction* operand) {
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
