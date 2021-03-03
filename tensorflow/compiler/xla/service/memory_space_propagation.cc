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

#include "tensorflow/compiler/xla/service/memory_space_propagation.h"

namespace xla {

StatusOr<bool> MemorySpacePropagation::Run(HloModule* module) {
  bool modified = false;
  // Configure bitcasts to define values. Otherwise, if there is only a bitcast
  // between a fusion input and output and these two values are in different
  // memory spaces, we can get inconsistent memory spaces between the parameter
  // and fusion operand or root and fusion output.
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloDataflowAnalysis::Run(*module, /*ssa_form=*/false,
                                               /*bitcast_defines_value=*/true));
  dataflow_analysis_ = std::move(dataflow_analysis);

  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        // Propagate the operand subshapes.
        for (int operand_idx = 0; operand_idx < instruction->operand_count();
             ++operand_idx) {
          for (const ShapeUtil::IndexedShape& indexed_shape :
               ShapeUtil::GetLeafShapes(
                   instruction->operand(operand_idx)->shape())) {
            int64 memory_space = indexed_shape.shape.layout().memory_space();
            modified |= Propagate(indexed_shape.index,
                                  instruction->fused_parameter(operand_idx),
                                  memory_space);
          }
        }

        // Propagate output subshapes.
        for (const ShapeUtil::IndexedShape& indexed_shape :
             ShapeUtil::GetLeafShapes(instruction->shape())) {
          int64 memory_space = indexed_shape.shape.layout().memory_space();
          modified |=
              Propagate(indexed_shape.index,
                        instruction->fused_expression_root(), memory_space);
        }
      }
    }
  }
  return modified;
}

bool MemorySpacePropagation::Propagate(ShapeIndexView index,
                                       const HloInstruction* callee_instruction,
                                       int64 memory_space) const {
  bool modified = false;
  const HloValue& value = dataflow_analysis_->GetUniqueValueAt(
      callee_instruction, index.ToShapeIndex());

  for (const HloPosition& position : value.positions()) {
    HloInstruction* instruction = position.instruction;
    Shape* shape = ShapeUtil::GetMutableSubshape(instruction->mutable_shape(),
                                                 position.index);
    if (shape->layout().memory_space() == memory_space) {
      continue;
    }
    shape->mutable_layout()->set_memory_space(memory_space);
    modified = true;

    // For fusion outputs, propagate the memory space to the fusion root.
    if (instruction->opcode() == HloOpcode::kFusion) {
      Propagate(position.index, instruction->fused_expression_root(),
                memory_space);
    }

    const HloInstruction* parent_fusion =
        instruction->parent()->FusionInstruction();
    // For nested fusion roots, pop one level up and propagate the memory space
    // to the output of the calling fusion instruction.
    if (instruction == instruction->parent()->root_instruction() &&
        parent_fusion->parent()->IsFusionComputation()) {
      Propagate(position.index, parent_fusion, memory_space);
    }

    // For nested fusion parameters, pop one level up and propagate the memory
    // space to the operand of the calling fusion instruction.
    if (instruction->opcode() == HloOpcode::kParameter &&
        parent_fusion->parent()->IsFusionComputation()) {
      const HloInstruction* fusion_operand =
          parent_fusion->operand(instruction->parameter_number());
      Propagate(position.index, fusion_operand, memory_space);
    }
  }

  for (const HloUse& use : value.uses()) {
    // For fusion uses, propagate the memory space to the fusion parameter.
    if (use.instruction->opcode() == HloOpcode::kFusion) {
      modified |= Propagate(
          use.operand_index,
          use.instruction->fused_parameter(use.operand_number), memory_space);
    }
  }
  return modified;
}

}  // namespace xla
