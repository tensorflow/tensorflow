/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/permutation_sort_expander.h"

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {
bool IsSimpleLessThanComparator(HloComputation* compare) {
  HloInstruction* root = compare->root_instruction();
  if (HloPredicateIsNotOp<HloOpcode::kCompare>(root)) {
    return false;
  }
  if (root->comparison_direction() != ComparisonDirection::kLt) {
    return false;
  }
  return root->operand(0) == compare->parameter_instruction(0) &&
         root->operand(1) == compare->parameter_instruction(1);
}
}  // namespace

bool PermutationSortExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (HloPredicateIsNotOp<HloOpcode::kSort>(instruction)) {
    return false;
  }
  if (!IsSimpleLessThanComparator(instruction->to_apply())) {
    return false;
  }
  const HloInstruction* sort_key = instruction->operand(0);
  int64_t dimension_to_sort = instruction->dimensions(0);
  if (instruction->operand_count() == 2 &&
      HloPredicateIsOp<HloOpcode::kGetTupleElement>(sort_key)) {
    const HloInstruction* other_sort = sort_key->operand(0);
    // Check whether the 'values' parameter is the result of another sort with
    // the same sort dimension.
    if (HloPredicateIsOp<HloOpcode::kSort>(other_sort) &&
        other_sort->operand_count() >= 2 &&
        other_sort->dimensions(0) == dimension_to_sort &&
        HloPredicateIsOp<HloOpcode::kIota>(
            other_sort->operand(sort_key->tuple_index()))) {
      auto* iota = Cast<HloIotaInstruction>(
          other_sort->operand(sort_key->tuple_index()));
      // The sort operand needs to be an integral iota, and the iota dimension
      // needs to be the dimension that was sorted.
      return iota->iota_dimension() == dimension_to_sort &&
             ShapeUtil::ElementIsIntegral(iota->shape());
    }
  }
  return false;
}

absl::StatusOr<HloInstruction*> PermutationSortExpander::ExpandInstruction(
    HloInstruction* instruction) {
  // We use the following construction method for a Scatter that applies
  // the permutation from `keys` to the `values` parameter.
  // - Set updates of the scatter to be a reshaped `values` parameter of sort
  //   (adding `rank` many 1 dimensions at the end).
  // - Take the `keys` parameter of the second sort and reshape it to have
  //   another `1` dimension at the end. This is the permutation we want to
  //   apply in the `dimension_to_sort` dimension.
  // - We want to create scatter indices such that the most minor dimension
  //   represents the indices in order from most major to most minor where the
  //   scatter updates should be stored. This can be done with a concatenate of
  //   iotas with different iota dimensions, except the index that corresponds
  //   to the dimension to sort. For that we will use the reshaped `keys`
  //   parameter. E.g. with rank 3 and dimension_to_sort = 1, we would have
  //   concatenate of (iota with iota_dimension=0, keys, iota with
  //   iota_dimension = 2).
  int64_t dimension_to_sort = instruction->dimensions(0);
  HloInstruction* sort_key = instruction->mutable_operand(0);
  int rank = static_cast<int>(sort_key->shape().dimensions().size());
  Shape extended_shape = sort_key->shape();
  ShapeUtil::AppendMinorDimension(1, &extended_shape);
  HloInstruction* reshaped_permutation = instruction->AddInstruction(
      HloInstruction::CreateReshape(extended_shape, sort_key));
  std::vector<HloInstruction*> concat_operands;
  concat_operands.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (i == dimension_to_sort) {
      concat_operands.push_back(reshaped_permutation);
    } else {
      concat_operands.push_back(instruction->AddInstruction(
          HloInstruction::CreateIota(extended_shape, i)));
    }
  }
  Shape concat_shape = sort_key->shape();
  ShapeUtil::AppendMinorDimension(rank, &concat_shape);
  HloInstruction* scatter_indices =
      rank > 1 ? instruction->AddInstruction(HloInstruction::CreateConcatenate(
                     concat_shape, concat_operands, rank))
               : reshaped_permutation;

  // We don't care about the operand, it will be completely overridden by
  // the updates. Initialize it to zero.
  HloInstruction* values = instruction->mutable_operand(1);
  Shape update_shape = values->shape();
  HloInstruction* zero =
      instruction->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(update_shape.element_type())));
  HloInstruction* scatter_operand =
      instruction->AddInstruction(HloInstruction::CreateBroadcast(
          update_shape, zero, /*broadcast_dimensions=*/{}));

  // Construct the updates computation, which simply replaces the operand
  // values with the update values.
  HloComputation::Builder b("update_replace_computation");
  Shape scalar_shape = ShapeUtil::MakeShape(S32, {});
  b.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape, "scalar_lhs"));
  HloInstruction* scalar_rhs = b.AddInstruction(
      HloInstruction::CreateParameter(1, scalar_shape, "scalar_rhs"));
  HloComputation* update_replace_computation =
      instruction->parent()->parent()->AddEmbeddedComputation(
          b.Build(scalar_rhs));

  ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(rank);
  for (int64_t i = 0; i < rank; ++i) {
    dim_numbers.add_inserted_window_dims(i);
    dim_numbers.add_scatter_dims_to_operand_dims(i);
  }
  HloInstruction* scatter =
      instruction->AddInstruction(HloInstruction::CreateScatter(
          update_shape, scatter_operand, scatter_indices, values,
          update_replace_computation, dim_numbers,
          /*indices_are_sorted=*/false, /*unique_indices=*/true));
  return instruction->AddInstruction(HloInstruction::CreateTuple(
      {instruction->AddInstruction(
           HloInstruction::CreateIota(sort_key->shape(), dimension_to_sort)),
       scatter}));
}

}  // namespace xla
