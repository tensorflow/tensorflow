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

#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {

namespace {
void AppendParams(const HloInstruction& instr,
                  std::vector<HloInstruction*>* params) {
  if (instr.opcode() == HloOpcode::kFusion) {
    params->insert(std::end(*params), std::begin(instr.fused_parameters()),
                   std::end(instr.fused_parameters()));
  } else {
    for (HloInstruction* operand : instr.operands()) {
      params->push_back(operand);
    }
  }
}
}  // namespace

bool LayoutsAreReduceInputFusionFriendly(const HloInstruction& producer,
                                         const HloInstruction& reduce) {
  std::vector<HloInstruction*> params;
  AppendParams(producer, &params);
  AppendParams(reduce, &params);
  int64 max_rank = -1;
  const Layout* max_rank_layout;
  for (HloInstruction* param : params) {
    if (ShapeUtil::IsArray(param->shape()) &&
        ShapeUtil::Rank(param->shape()) > max_rank) {
      max_rank = ShapeUtil::Rank(param->shape());
      max_rank_layout = &param->shape().layout();
    }
  }
  return absl::c_all_of(params, [&](HloInstruction* param) {
    return (!ShapeUtil::IsArray(param->shape())) ||
           (ShapeUtil::Rank(param->shape()) < max_rank) ||
           (LayoutUtil::Equal(param->shape().layout(), *max_rank_layout));
  });
}

bool IsInputFusibleReduction(const HloInstruction& instr) {
  if (instr.IsMultiOutputFusion()) {
    for (const HloInstruction* operand :
         instr.fused_expression_root()->operands()) {
      if (IsReductionToVector(*operand)) {
        CHECK(instr.fusion_kind() == HloInstruction::FusionKind::kInput)
            << " Multi-output fusion rooted at reduction-to-vector ops must be "
               "of kind kInput: "
            << instr.ToString();
        return true;
      }
    }
    return false;
  } else if (instr.opcode() == HloOpcode::kFusion) {
    if (IsReductionToVector(*instr.fused_expression_root())) {
      CHECK(instr.fusion_kind() == HloInstruction::FusionKind::kInput)
          << " Fusion rooted at reduction-to-vector op must be of kind kInput: "
          << instr.ToString();
      return true;
    }
    return false;
  }
  return IsReductionToVector(instr);
}

}  // namespace gpu
}  // namespace xla
