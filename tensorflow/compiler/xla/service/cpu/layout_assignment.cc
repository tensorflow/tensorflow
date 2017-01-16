/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/layout_assignment.h"

#include <numeric>

#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace cpu {

Status CpuLayoutAssignment::AddBackendConstraints(
    LayoutConstraints* constraints) {
  auto row_major_shape = [](const Shape& old_shape) {
    Shape new_shape(old_shape);
    std::vector<int64> dimension_order(new_shape.dimensions_size());
    std::iota(dimension_order.rbegin(), dimension_order.rend(), 0);
    *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
    return new_shape;
  };
  const HloComputation* computation = constraints->computation();
  for (auto& instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kConvolution &&
        PotentiallyImplementedAsEigenConvolution(*instruction)) {
      const HloInstruction* convolution = instruction.get();
      const HloInstruction* lhs_instruction = convolution->operand(0);
      const HloInstruction* rhs_instruction = convolution->operand(1);

      // In order to implement `convolution` with Eigen convolution, the layouts
      // of the input, filter, and output need to be row-major.
      //
      // These constraints are not hard constraints. Ideally, we should decide
      // which layouts to choose according to some cost model.
      Shape output_shape(row_major_shape(convolution->shape()));
      Shape input_shape(row_major_shape(lhs_instruction->shape()));
      Shape filter_shape(row_major_shape(rhs_instruction->shape()));

      // Set layouts of the instructions' shapes.
      TF_RETURN_IF_ERROR(
          constraints->SetOperandLayout(input_shape, convolution, 0));
      TF_RETURN_IF_ERROR(
          constraints->SetOperandLayout(filter_shape, convolution, 1));
      TF_RETURN_IF_ERROR(
          constraints->SetInstructionLayout(output_shape, convolution));
    } else if (PotentiallyImplementedAsEigenDot(*instruction)) {
      const HloInstruction* dot = instruction.get();
      const HloInstruction* lhs_instruction = dot->operand(0);
      const HloInstruction* rhs_instruction = dot->operand(1);

      // In order to implement `dot` with Eigen dot, the layouts of the lhs,
      // rhs, and output need to be row-major.
      //
      // These constraints are not hard constraints. Ideally, we should decide
      // which layouts to choose according to some cost model.
      Shape output_shape(row_major_shape(dot->shape()));
      Shape lhs_shape(row_major_shape(lhs_instruction->shape()));
      Shape rhs_shape(row_major_shape(rhs_instruction->shape()));

      // Set layouts of the instructions' shapes.
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(lhs_shape, dot, 0));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(rhs_shape, dot, 1));
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(output_shape, dot));
    } else {
      for (int64 operand_no = 0; operand_no < instruction->operand_count();
           ++operand_no) {
        // Skip operands which already have a constraint.
        if (constraints->OperandLayout(instruction.get(), operand_no) !=
            nullptr) {
          continue;
        }
        // Skip over forwarded operands.
        if (constraints->OperandBufferForwarded(instruction.get(),
                                                operand_no)) {
          continue;
        }
        Shape operand_shape(
            row_major_shape(instruction->operand(operand_no)->shape()));
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
            operand_shape, instruction.get(), operand_no));
      }
      // Skip over the root instruction for the top-level computation.
      if (computation->parent()->entry_computation() == computation &&
          computation->root_instruction() == instruction.get()) {
        continue;
      }
      // Skip instructions which don't produce array shapes (tuples, opaque,
      // etc.).
      if (!ShapeUtil::IsArray(instruction->shape())) {
        continue;
      }
      tensorflow::gtl::ArraySlice<const LogicalBuffer*> buffers =
          constraints->points_to_analysis()
              .GetPointsToSet(instruction.get())
              .element({});
      // Only force the layout if the instruction hasn't been otherwise assigned
      // one or has ambiguous aliasing properties.
      if (buffers.size() == 1 &&
          buffers[0]->instruction() == instruction.get() &&
          constraints->BufferLayout(*buffers[0]) == nullptr) {
        Shape output_shape(row_major_shape(instruction->shape()));
        TF_RETURN_IF_ERROR(
            constraints->SetInstructionLayout(output_shape, instruction.get()));
      }
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace cpu
}  // namespace xla
