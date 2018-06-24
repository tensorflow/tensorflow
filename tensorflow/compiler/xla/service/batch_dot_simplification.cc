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

#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"

namespace xla {
StatusOr<bool>
BatchDotSimplification::ElideDegenerateBatchDimensionFromBatchDot(
    HloInstruction* batch_dot) {
  const DotDimensionNumbers& dim_numbers = batch_dot->dot_dimension_numbers();
  HloInstruction *lhs = batch_dot->mutable_operand(0),
                 *rhs = batch_dot->mutable_operand(1);
  const Shape& lhs_shape = lhs->shape();

  std::vector<int64> degenerate_dims;
  for (int64 batch_dim : dim_numbers.lhs_batch_dimensions()) {
    if (lhs_shape.dimensions(batch_dim) == 1) {
      degenerate_dims.push_back(batch_dim);
    }
  }

  if (degenerate_dims.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                      ElideDegenerateDims(lhs, degenerate_dims));
  TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                      ElideDegenerateDims(rhs, degenerate_dims));

  DotDimensionNumbers new_dim_numbers = dim_numbers;
  new_dim_numbers.clear_lhs_batch_dimensions();
  new_dim_numbers.clear_rhs_batch_dimensions();

  for (int64 i = 0, e = dim_numbers.lhs_batch_dimensions_size() -
                        degenerate_dims.size();
       i < e; i++) {
    new_dim_numbers.add_lhs_batch_dimensions(i);
    new_dim_numbers.add_rhs_batch_dimensions(i);
  }

  new_dim_numbers.set_lhs_contracting_dimensions(
      0,
      new_dim_numbers.lhs_contracting_dimensions(0) - degenerate_dims.size());
  new_dim_numbers.set_rhs_contracting_dimensions(
      0,
      new_dim_numbers.rhs_contracting_dimensions(0) - degenerate_dims.size());

  TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                      MakeDotHlo(new_lhs, new_rhs, new_dim_numbers));

  TF_ASSIGN_OR_RETURN(HloInstruction * new_dot_reshaped,
                      MakeReshapeHlo(batch_dot->shape(), new_dot));

  VLOG(2) << "Replaced " << batch_dot->ToString() << " with "
          << new_dot->ToString();

  TF_RETURN_IF_ERROR(
      batch_dot->parent()->ReplaceInstruction(batch_dot, new_dot_reshaped));

  return true;
}

tensorflow::StringPiece BatchDotSimplification::name() const {
  return "batch-dot-simplification";
}

StatusOr<bool> BatchDotSimplification::Run(HloModule* module) {
  bool changed = false;
  std::vector<HloInstruction*> dot_instrs;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    c_copy_if(computation->instructions(), std::back_inserter(dot_instrs),
              [](HloInstruction* instr) {
                return instr->opcode() == HloOpcode::kDot;
              });
  }
  for (HloInstruction* dot_instr : dot_instrs) {
    TF_ASSIGN_OR_RETURN(bool elided_batch_dim_from_one,
                        ElideDegenerateBatchDimensionFromBatchDot(dot_instr));
    changed |= elided_batch_dim_from_one;
  }
  return changed;
}
}  // namespace xla
