/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/dynamic_dimension_simplifier.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"

namespace xla {
namespace {

// Concat(Concat(A, B), C) => Concat(A, B, C)
absl::StatusOr<bool> ConcatForwarding(HloInstruction* concat) {
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }
  bool changed = false;

  auto parent = concat->parent();
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : concat->operands()) {
    if (operand->opcode() != HloOpcode::kConcatenate ||
        operand->concatenate_dimension() != concat->concatenate_dimension()) {
      new_operands.push_back(operand);
    } else {
      changed = true;
      for (HloInstruction* operand_operand : operand->operands()) {
        new_operands.push_back(operand_operand);
      }
    }
  }
  if (changed) {
    auto new_concat = parent->AddInstruction(HloInstruction::CreateConcatenate(
        concat->shape(), new_operands, concat->concatenate_dimension()));
    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(concat, new_concat));
  }
  return changed;
}

// Slice(Concat(A1, A2, ..., An, ...), [n:n+1]) => An
absl::StatusOr<bool> SliceConcatForwarding(HloInstruction* slice) {
  if (slice->opcode() != HloOpcode::kSlice) {
    return false;
  }
  auto concat = slice->mutable_operand(0);
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }

  if (slice->shape().dimensions_size() != 1) {
    // Slice concat forwarding only work for size 1 tensor.
    return false;
  }

  int64_t concat_dim = concat->concatenate_dimension();

  std::vector<HloInstruction*> new_operands;
  int64_t size_so_far = 0;
  int64_t slice_size = slice->shape().dimensions(concat_dim);
  if (slice_size != slice->slice_limits(0) - slice->slice_starts(0)) {
    return false;
  }
  if (slice->slice_strides(0) != 1) {
    return false;
  }
  for (HloInstruction* operand : concat->operands()) {
    if (size_so_far == slice->slice_starts(0) &&
        operand->shape().dimensions(0) == slice_size) {
      // Found an operand that can be forwarded.
      TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(operand));
      return true;
    }
    size_so_far += operand->shape().dimensions(concat_dim);
  }

  return false;
}

// Reshape(Broadcast(A, []->[1]), [1]->[]) ==> A
absl::StatusOr<bool> ReshapeBroadcastForwarding(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto broadcast = reshape->mutable_operand(0);
  if (broadcast->opcode() != HloOpcode::kBroadcast) {
    return false;
  }

  if (reshape->shape().dimensions_size() != 0) {
    return false;
  }

  if (broadcast->shape().dimensions_size() != 1) {
    return false;
  }

  if (broadcast->mutable_operand(0)->shape().dimensions_size() != 0) {
    return false;
  }

  TF_RETURN_IF_ERROR(
      reshape->ReplaceAllUsesWith(broadcast->mutable_operand(0)));

  return true;
}

// Reshape(Reshape(A, []->[1]), [1]->[]) ==> A
absl::StatusOr<bool> ReshapeReshapeForwarding(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto reshape_2 = reshape->mutable_operand(0);
  if (reshape_2->opcode() != HloOpcode::kReshape) {
    return false;
  }

  if (!Shape::Equal()(reshape->shape(), reshape_2->operand(0)->shape())) {
    return false;
  }
  TF_RETURN_IF_ERROR(
      reshape->ReplaceAllUsesWith(reshape_2->mutable_operand(0)));

  return true;
}

// Convert(A, T->T) ==> A
absl::StatusOr<bool> IdentityConvertRemoving(HloInstruction* convert) {
  if (convert->opcode() != HloOpcode::kConvert) {
    return false;
  }
  auto operand = convert->mutable_operand(0);
  if (Shape::Equal()(convert->shape(), operand->shape())) {
    TF_RETURN_IF_ERROR(convert->ReplaceAllUsesWith(operand));
    return true;
  }
  return false;
}

// Reshape(A, S->S) ==> A
absl::StatusOr<bool> IdentityReshapeRemoving(HloInstruction* reshape) {
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  auto operand = reshape->mutable_operand(0);
  if (Shape::Equal()(reshape->shape(), operand->shape())) {
    TF_RETURN_IF_ERROR(reshape->ReplaceAllUsesWith(operand));
    return true;
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> DynamicDimensionSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "DynamicDimensionSimplifier::Run(), before:\n" + module->ToString());
  bool changed = false;

  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, ConcatForwarding(inst));
      changed |= local_changed;
    }
  }

  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, SliceConcatForwarding(inst));
      changed |= local_changed;
    }
  }

  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, ReshapeBroadcastForwarding(inst));
      changed |= local_changed;
    }
  }
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, ReshapeReshapeForwarding(inst));
      changed |= local_changed;
    }
  }
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, IdentityConvertRemoving(inst));
      changed |= local_changed;
    }
  }
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool local_changed, IdentityReshapeRemoving(inst));
      changed |= local_changed;
    }
  }
  XLA_VLOG_LINES(
      2, "DynamicDimensionSimplifier::Run(), after:\n" + module->ToString());
  return changed;
}
}  // namespace xla
