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

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
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
  return ShapeUtil::Equal(get_element_shape(element_instr_1),
                          get_element_shape(element_instr_2));
}

bool GpuMultiOutputFusion::IsProfitableOperand(HloInstruction* instr) {
  // kConstant instruction will not have memory reads, so it won't be a profit
  // source. Skip them.
  if (instr->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    return false;
  }
  // We don't target to fuse producer/consumer instructions -- this should
  // be taken care of by the instruction_fusion pass. If instr has only
  // one user, it will not have sibling instructions. We won't consider it.
  if (instr->user_count() < 2) {
    return false;
  }
  return true;
}

namespace {
bool IsReduction(HloInstruction* instr) {
  if (instr->IsMultiOutputFusion()) {
    for (const HloInstruction* operand :
         instr->fused_expression_root()->operands()) {
      if (operand->opcode() == HloOpcode::kReduce) {
        return true;
      }
    }
    return false;
  } else if (instr->opcode() == HloOpcode::kFusion) {
    return instr->fused_expression_root()->opcode() == HloOpcode::kReduce;
  } else {
    return instr->opcode() == HloOpcode::kReduce;
  }
}
}  // namespace

bool GpuMultiOutputFusion::IsFusible(HloInstruction* instr) {
  return IsReduction(instr);
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

}  // namespace gpu
}  // namespace xla
