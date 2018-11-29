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

bool IsReduceInputFusion(const HloInstruction& instr) {
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
  } else if (instr.opcode() == HloOpcode::kFusion &&
             IsReductionToVector(*instr.fused_expression_root())) {
    CHECK(instr.fusion_kind() == HloInstruction::FusionKind::kInput)
        << " Fusion rooted at reduction-to-vector op must be of kind kInput: "
        << instr.ToString();
    return true;
  }
  return false;
}

bool IsInputFusibleReduction(const HloInstruction& instr) {
  return IsReduceInputFusion(instr) || IsReductionToVector(instr);
}

bool ShapesCompatibleForMultiOutputFusion(const HloInstruction& instr1,
                                          const HloInstruction& instr2) {
  // Returns the instructions that determines the emitter used for lowering,
  // sometimes referred to as "the real hero".
  auto get_real_hero =
      [&](const HloInstruction* instr) -> const HloInstruction* {
    if (instr->opcode() == HloOpcode::kFusion) {
      auto fused_expression_root = instr->fused_expression_root();
      if (instr->IsMultiOutputFusion()) {
        // If possible, we want to pick a reduction-to-vector operand of the
        // fusion root, because it has the most constraints.
        for (const auto* inst : fused_expression_root->operands()) {
          if (IsReductionToVector(*inst)) {
            return inst;
          }
        }
        return fused_expression_root->operands()[0];
      }
      return fused_expression_root;
    }
    return instr;
  };

  // Multi-output fusion kernels share a common parallel loop. The loop
  // dimenstions are determined by instruction shapes.
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
    // Special-case reduction-to-vector ops: The loop dimensions are determined
    // by the shape of the first operand.
    if (IsReductionToVector(*element_instr)) {
      return element_instr->operand(0)->shape();
    }
    return element_instr->shape();
  };

  // All shapes of the root tuple of multi-output fusions should agree, i.e. all
  // root ops should have equal output shapes. An exception are
  // reduction-to-vector ops. Here the input shapes of the reduction (first
  // operand shape) need to be considered.
  auto* instr_1 = get_real_hero(&instr1);
  auto* instr_2 = get_real_hero(&instr2);
  // TODO(tjoerg): Relax the shape constraint. The datatype does not matter.
  if (IsReductionToVector(*instr_1) && IsReductionToVector(*instr_2) &&
      !ShapeUtil::Equal(instr_1->shape(), instr_2->shape())) {
    return false;
  }
  // The elementwise output shapes must be the same (including layout).
  // TODO(tjoerg): Further relax the constraint. The datatype does not matter.
  return ShapeUtil::EqualIgnoringFpPrecision(get_loop_shape(instr_1),
                                             get_loop_shape(instr_2));
}

}  // namespace gpu
}  // namespace xla
