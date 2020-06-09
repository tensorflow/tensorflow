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
  TF_ASSIGN_OR_RETURN(auto dataflow_analysis,
                      HloDataflowAnalysis::Run(*module));
  dataflow_analysis_ = std::move(dataflow_analysis);

  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        // Propagate the operand subshapes.
        for (int operand_idx = 0; operand_idx < instruction->operand_count();
             ++operand_idx) {
          modified |=
              PropagateSubshapes(instruction->operand(operand_idx)->shape(),
                                 instruction->fused_parameter(operand_idx));
        }

        // Propagate output subshapes.
        modified |= PropagateSubshapes(instruction->shape(),
                                       instruction->fused_expression_root());
      }
    }
  }
  return modified;
}

bool MemorySpacePropagation::PropagateSubshapes(
    const Shape& caller_shape, const HloInstruction* callee_instruction) const {
  bool modified = false;
  for (const ShapeUtil::IndexedShape& indexed_shape :
       ShapeUtil::GetLeafShapes(caller_shape)) {
    int64 memory_space = indexed_shape.shape.layout().memory_space();
    const HloValue& value = dataflow_analysis_->GetUniqueValueAt(
        callee_instruction, indexed_shape.index);

    for (const HloPosition& position : value.positions()) {
      Shape* shape = ShapeUtil::GetMutableSubshape(
          position.instruction->mutable_shape(), position.index);
      if (shape->layout().memory_space() != memory_space) {
        shape->mutable_layout()->set_memory_space(memory_space);
        modified = true;
      }
    }
  }
  return modified;
}

}  // namespace xla
