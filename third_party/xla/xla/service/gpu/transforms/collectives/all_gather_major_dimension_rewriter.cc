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

#include "xla/service/gpu/transforms/collectives/all_gather_major_dimension_rewriter.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/permutation_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

absl::Status AllGatherMajorDimensionRewriter::Visitor::HandleAllGather(
    HloInstruction* instruction) {
  auto* all_gather = Cast<HloAllGatherInstruction>(instruction);
  const int64_t original_gather_dim = all_gather->all_gather_dimension();

  if (original_gather_dim == 0) {
    return absl::OkStatus();
  }

  HloComputation* computation = instruction->parent();
  const int64_t operand_count = all_gather->operand_count();

  const Shape& first_input_shape = all_gather->operand(0)->shape();
  const Shape& first_output_shape = operand_count == 1
                                        ? all_gather->shape()
                                        : all_gather->shape().tuple_shapes(0);
  const int64_t replica_count =
      first_output_shape.dimensions(original_gather_dim) /
      first_input_shape.dimensions(original_gather_dim);

  absl::InlinedVector<Shape, 1> new_all_gather_shapes;
  absl::InlinedVector<HloInstruction*, 1> operands;
  for (int64_t i = 0; i < operand_count; ++i) {
    HloInstruction* operand = all_gather->mutable_operand(i);
    const Shape& input_shape = operand->shape();
    if (!LayoutUtil::IsMonotonicWithDim0Major(input_shape.layout())) {
      // The pass is designed to run before layout assignment, layouts are
      // expected to be default.
      return absl::OkStatus();
    }
    absl::InlinedVector<int64_t, 4> new_input_dims(
        input_shape.dimensions().begin(), input_shape.dimensions().end());
    new_input_dims[0] *= replica_count;
    new_all_gather_shapes.push_back(
        ShapeUtil::MakeShape(input_shape.element_type(), new_input_dims));
    operands.push_back(operand);
  }

  const Shape new_all_gather_shape =
      operand_count == 1 ? new_all_gather_shapes[0]
                         : ShapeUtil::MakeTupleShape(new_all_gather_shapes);

  HloInstruction* new_all_gather = computation->AddInstruction(
      all_gather->CloneWithNewOperands(new_all_gather_shape, operands));
  Cast<HloAllGatherInstruction>(new_all_gather)->set_all_gather_dimension(0);

  absl::InlinedVector<HloInstruction*, 1> final_results;
  for (int64_t i = 0; i < operand_count; ++i) {
    HloInstruction* operand = all_gather->mutable_operand(i);
    const Shape& input_shape = operand->shape();
    const Shape& output_shape = operand_count == 1
                                    ? all_gather->shape()
                                    : all_gather->shape().tuple_shapes(i);

    HloInstruction* ag_result =
        operand_count == 1
            ? new_all_gather
            : computation->AddInstruction(
                  HloInstruction::CreateGetTupleElement(new_all_gather, i));

    absl::InlinedVector<int64_t, 4> first_reshape_dims;
    first_reshape_dims.push_back(replica_count);
    absl::c_copy(input_shape.dimensions(),
                 std::back_inserter(first_reshape_dims));
    TF_ASSIGN_OR_RETURN(HloInstruction * first_reshape,
                        MakeReshapeHlo(first_reshape_dims, ag_result));

    absl::InlinedVector<int64_t, 4> transpose_order;
    for (int64_t j = 0; j < input_shape.dimensions().size(); ++j) {
      if (j == original_gather_dim) {
        transpose_order.push_back(0);
      }
      transpose_order.push_back(j + 1);
    }
    HloInstruction* transpose =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            ShapeUtil::MakeShape(input_shape.element_type(),
                                 Permute(first_reshape_dims, transpose_order)),
            first_reshape, transpose_order));

    TF_ASSIGN_OR_RETURN(HloInstruction * final_reshape,
                        MakeReshapeHlo(output_shape, transpose));
    final_results.push_back(final_reshape);
  }

  HloInstruction* replacement =
      operand_count == 1 ? final_results[0]
                         : computation->AddInstruction(
                               HloInstruction::CreateTuple(final_results));

  return ReplaceInstruction(instruction, replacement);
}

absl::StatusOr<bool> AllGatherMajorDimensionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  Visitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace gpu
}  // namespace xla
