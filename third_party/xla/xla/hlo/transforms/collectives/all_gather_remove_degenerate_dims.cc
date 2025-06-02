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

#include "xla/hlo/transforms/collectives/all_gather_remove_degenerate_dims.h"

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Returns the dimensions major to the all_gather_dimension that are degenerate
// in all operands. Technically, we could drop dimensions even if their indices
// don't match, as long as the number of degenerate dimensions before the
// all_gather_dim matches, but we haven't yet seen a case where that is useful.
absl::InlinedVector<int64_t, 1> GetMajorDimsToDelete(
    const HloAllGatherInstruction* all_gather) {
  absl::InlinedVector<int64_t, 1> major_dims;
  int64_t all_gather_dim = all_gather->all_gather_dimension();
  for (int64_t dim = 0; dim < all_gather_dim; ++dim) {
    bool can_drop_dim;
    if (all_gather->shape().IsArray()) {
      can_drop_dim = all_gather->shape().dimensions(dim) == 1;
    } else {
      can_drop_dim = absl::c_all_of(
          all_gather->shape().tuple_shapes(),
          [&](const Shape& shape) { return shape.dimensions(dim) == 1; });
    }
    if (can_drop_dim) {
      major_dims.push_back(dim);
    }
  }
  return major_dims;
}

HloInstruction* ReshapeAllGatherOperand(
    HloInstruction* operand,
    const absl::InlinedVector<int64_t, 1>& major_dims_to_delete,
    int64_t all_gather_dim) {
  absl::InlinedVector<int64_t, 1> dims_to_delete = major_dims_to_delete;
  // Find the minor dimensions to drop in this shape. They can be different for
  // each operand.
  int64_t rank = operand->shape().dimensions().size();
  for (int64_t i = all_gather_dim + 1; i < rank; ++i) {
    if (operand->shape().dimensions(i) == 1) {
      dims_to_delete.push_back(i);
    }
  }

  // If this operand had no degenerate dimensions, just return it directly.
  if (dims_to_delete.empty()) {
    return operand;
  }

  Shape new_shape = operand->shape();
  new_shape.DeleteDimensions(dims_to_delete);
  return operand->parent()->AddInstruction(
      HloInstruction::CreateReshape(new_shape, operand));
}

int64_t GetShardCount(HloAllGatherInstruction* all_gather) {
  int64_t dim = all_gather->all_gather_dimension();
  int64_t input_size = all_gather->operand(0)->shape().dimensions(dim);
  int64_t result_size =
      (all_gather->shape().IsTuple() ? all_gather->shape().tuple_shapes(0)
                                     : all_gather->shape())
          .dimensions(dim);
  return result_size / input_size;
}

absl::StatusOr<HloInstruction*> CreateNewAllGather(
    HloAllGatherInstruction* all_gather) {
  int64_t all_gather_dim = all_gather->all_gather_dimension();
  // Find the dimensions before the all_gather_dim that we can delete in all
  // operands.
  absl::InlinedVector<int64_t, 1> major_dims_to_delete =
      GetMajorDimsToDelete(all_gather);
  absl::InlinedVector<HloInstruction*, 1> reshaped_operands;
  absl::InlinedVector<const Shape*, 1> reshaped_shapes;
  for (auto* operand : all_gather->mutable_operands()) {
    reshaped_operands.push_back(
        ReshapeAllGatherOperand(operand, major_dims_to_delete, all_gather_dim));
    reshaped_shapes.push_back(&reshaped_operands.back()->shape());
  }

  if (absl::c_equal(reshaped_operands, all_gather->operands())) {
    return all_gather;
  }

  int64_t new_all_gather_dim = all_gather_dim - major_dims_to_delete.size();
  int64_t shard_count = GetShardCount(all_gather);
  TF_ASSIGN_OR_RETURN(Shape new_all_gather_shape,
                      ShapeInference::InferAllGatherShape(
                          reshaped_shapes, new_all_gather_dim, shard_count));
  auto* new_all_gather = Cast<HloAllGatherInstruction>(
      all_gather->parent()->AddInstruction(all_gather->CloneWithNewOperands(
          new_all_gather_shape, reshaped_operands)));
  new_all_gather->set_all_gather_dimension(new_all_gather_dim);
  return new_all_gather;
}

// Reshapes the results of `new_all_gather` to conform to the shapes of
// `original_all_gather` and replaces the latter.
absl::Status ReshapeAndReplaceResults(HloInstruction* original_all_gather,
                                      HloInstruction* new_all_gather) {
  auto* computation = original_all_gather->parent();
  int operand_count = original_all_gather->operand_count();
  const Shape& original_shape = original_all_gather->shape();

  if (operand_count == 1) {
    return computation->ReplaceWithNewInstruction(
        original_all_gather,
        HloInstruction::CreateReshape(original_shape, new_all_gather));
  }

  absl::InlinedVector<HloInstruction*, 2> results;
  for (int i = 0; i < operand_count; ++i) {
    auto* gte = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_all_gather, i));
    auto* reshaped = computation->AddInstruction(
        HloInstruction::CreateReshape(original_shape.tuple_shapes(i), gte));
    results.push_back(reshaped);
  }
  return computation->ReplaceWithNewInstruction(
      original_all_gather, HloInstruction::CreateTuple(results));
}

}  // namespace

absl::StatusOr<bool> AllGatherRemoveDegenerateDims::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllGather) {
        continue;
      }
      // Skip all-gathers we can't change.
      auto* all_gather = Cast<HloAllGatherInstruction>(inst);
      if (all_gather->constrain_layout()) {
        continue;
      }

      TF_ASSIGN_OR_RETURN(HloInstruction * new_all_gather,
                          CreateNewAllGather(all_gather));
      if (new_all_gather != all_gather) {
        TF_RETURN_IF_ERROR(
            ReshapeAndReplaceResults(all_gather, new_all_gather));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
