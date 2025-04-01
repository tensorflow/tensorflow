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

#include "xla/hlo/transforms/memory_space_propagation.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

absl::StatusOr<bool> MemorySpacePropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool modified = false;
  // Configure bitcasts to define values. Otherwise, if there is only a bitcast
  // between a fusion input and output and these two values are in different
  // memory spaces, we can get inconsistent memory spaces between the parameter
  // and fusion operand or root and fusion output.
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloDataflowAnalysis::Run(*module, /*ssa_form=*/false,
                                               /*bitcast_defines_value=*/true));
  dataflow_analysis_ = std::move(dataflow_analysis);

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        // Propagate the operand subshapes.
        for (int operand_idx = 0;
             operand_idx < instruction->fused_parameters().size();
             ++operand_idx) {
          ShapeUtil::ForEachLeafShape(
              instruction->operand(operand_idx)->shape(),
              [&](const Shape& sub_shape, const ShapeIndex& index) {
                modified |=
                    Propagate(index, instruction->fused_parameter(operand_idx),
                              sub_shape);
              });
        }

        // Propagate output subshapes.
        ShapeUtil::ForEachLeafShape(
            instruction->shape(),
            [&](const Shape& sub_shape, const ShapeIndex& index) {
              modified |= Propagate(index, instruction->fused_expression_root(),
                                    sub_shape);
            });
      }
    }
  }
  return modified;
}

bool MemorySpacePropagation::Propagate(ShapeIndexView index,
                                       const HloInstruction* callee_instruction,
                                       const Shape& src_shape) const {
  bool modified = false;
  const HloValue& value = dataflow_analysis_->GetUniqueValueAt(
      callee_instruction, ShapeIndex(index));

  for (const HloPosition& position : value.positions()) {
    HloInstruction* instruction = position.instruction;
    Shape* shape = ShapeUtil::GetMutableSubshape(instruction->mutable_shape(),
                                                 position.index);
    std::optional<SplitConfig> dest_split_config =
        LayoutUtil::GetSplitConfig(*shape);
    std::optional<SplitConfig> src_split_config =
        LayoutUtil::GetSplitConfig(src_shape);

    if (shape->layout().memory_space() == src_shape.layout().memory_space() &&
        dest_split_config == src_split_config) {
      continue;
    }
    shape->mutable_layout()->set_memory_space(
        src_shape.layout().memory_space());
    shape->mutable_layout()->clear_split_configs();
    if (src_split_config.has_value()) {
      shape->mutable_layout()->add_split_configs(*src_split_config);
    }
    modified = true;

    // For fusion outputs, propagate the memory space to the fusion root.
    if (instruction->opcode() == HloOpcode::kFusion) {
      Propagate(position.index, instruction->fused_expression_root(),
                src_shape);
    }

    const HloInstruction* parent_fusion =
        instruction->parent()->FusionInstruction();
    // For nested fusion roots, pop one level up and propagate the memory space
    // to the output of the calling fusion instruction.
    if (instruction == instruction->parent()->root_instruction() &&
        parent_fusion->parent()->IsFusionComputation()) {
      Propagate(position.index, parent_fusion, src_shape);
    }

    // For nested fusion parameters, pop one level up and propagate the memory
    // space to the operand of the calling fusion instruction.
    if (instruction->opcode() == HloOpcode::kParameter &&
        parent_fusion->parent()->IsFusionComputation()) {
      const HloInstruction* fusion_operand =
          parent_fusion->operand(instruction->parameter_number());
      Propagate(position.index, fusion_operand, src_shape);
    }
  }

  for (const HloUse& use : value.GetUses()) {
    // For fusion uses, propagate the memory space to the fusion parameter.
    if (use.instruction->opcode() == HloOpcode::kFusion) {
      modified |= Propagate(
          use.operand_index,
          use.instruction->fused_parameter(use.operand_number), src_shape);
    }
  }
  return modified;
}

}  // namespace xla
