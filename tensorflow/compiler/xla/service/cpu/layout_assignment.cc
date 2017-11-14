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

#include "tensorflow/compiler/xla/map_util.h"
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
  auto col_major_shape = [](const Shape& old_shape) {
    Shape new_shape(old_shape);
    std::vector<int64> dimension_order(new_shape.dimensions_size());
    std::iota(dimension_order.begin(), dimension_order.end(), 0);
    *new_shape.mutable_layout() = LayoutUtil::MakeLayout(dimension_order);
    return new_shape;
  };

  // We want to change the layout of constant arrays to be column major when all
  // of their users are dot operations that can be made faster with the flipped
  // layout.  To avoid going quadriatic over the # of instructions, we cache
  // this property in should_make_rhs_col_major -- it maps a constant to true if
  // all of the users of said constant are dot operations that can be sped up.
  // This cache is populated lazily as we encounter dot operations traversing
  // the instruction stream.
  tensorflow::gtl::FlatMap<const HloInstruction*, bool>
      should_make_rhs_col_major_cache;
  auto should_make_rhs_col_major = [&](const HloInstruction& instruction) {
    if (ProfitableToImplementDotInUntiledLlvmIr(instruction) !=
        DotInLlvmIrProfitable::kWithColumnMajorRhs) {
      return false;
    }

    const auto* rhs = instruction.operand(1);
    if (rhs->opcode() != HloOpcode::kConstant) {
      return false;
    }

    auto it = should_make_rhs_col_major_cache.find(rhs);
    if (it != should_make_rhs_col_major_cache.end()) {
      return it->second;
    }

    bool result = std::all_of(
        rhs->users().begin(), rhs->users().end(), [&](HloInstruction* user) {
          return ProfitableToImplementDotInUntiledLlvmIr(*user) ==
                     DotInLlvmIrProfitable::kWithColumnMajorRhs &&
                 user->operand(0) != rhs;
        });

    InsertOrDie(&should_make_rhs_col_major_cache, rhs, result);
    return result;
  };

  const HloComputation* computation = constraints->computation();
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kConvolution &&
        PotentiallyImplementedAsEigenConvolution(*instruction)) {
      const HloInstruction* convolution = instruction;
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
    } else if (should_make_rhs_col_major(*instruction)) {
      auto* dot = instruction;
      const auto& rhs_shape = dot->operand(1)->shape();
      TF_RETURN_IF_ERROR(
          constraints->SetOperandLayout(col_major_shape(rhs_shape), dot, 1));
    } else if (PotentiallyImplementedAsEigenDot(*instruction)) {
      const HloInstruction* dot = instruction;
      // In order to implement `dot` with Eigen dot, the layouts of the lhs,
      // rhs, and output need to be row-major.
      //
      // These constraints are not hard constraints. Ideally, we should decide
      // which layouts to choose according to some cost model.
      Shape output_shape(row_major_shape(dot->shape()));

      const HloInstruction* lhs_instruction = dot->operand(0);
      Shape lhs_shape(row_major_shape(lhs_instruction->shape()));
      TF_RETURN_IF_ERROR(constraints->SetOperandLayout(lhs_shape, dot, 0));

      // dot is a kDot or a kTransposeDot fusion node.  In the latter case, if
      // it represents X @ X, it may have just one operand.
      if (dot->operand_count() > 1) {
        const HloInstruction* rhs_instruction = dot->operand(1);
        Shape rhs_shape(row_major_shape(rhs_instruction->shape()));
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(rhs_shape, dot, 1));
      }

      // Set layouts of the instructions' shapes.
      TF_RETURN_IF_ERROR(constraints->SetInstructionLayout(output_shape, dot));
    } else {
      for (int64 operand_no = 0; operand_no < instruction->operand_count();
           ++operand_no) {
        // Skip operands which already have a constraint.
        if (constraints->OperandLayout(instruction, operand_no) != nullptr) {
          continue;
        }
        // Skip over forwarded operands.
        if (constraints->OperandBufferForwarded(instruction, operand_no)) {
          continue;
        }
        Shape operand_shape(
            row_major_shape(instruction->operand(operand_no)->shape()));
        TF_RETURN_IF_ERROR(constraints->SetOperandLayout(
            operand_shape, instruction, operand_no));
      }
      // Skip over the root instruction for the top-level computation.
      if (computation->parent()->entry_computation() == computation &&
          computation->root_instruction() == instruction) {
        continue;
      }
      // Skip instructions which don't produce array shapes (tuples, opaque,
      // etc.).
      if (!ShapeUtil::IsArray(instruction->shape())) {
        continue;
      }
    }
  }
  return tensorflow::Status::OK();
}
}  // namespace cpu
}  // namespace xla
