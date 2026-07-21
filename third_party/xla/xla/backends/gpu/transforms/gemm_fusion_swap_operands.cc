/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/gemm_fusion_swap_operands.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Swaps operands of a dot instruction, while keeping the same physical output
// layout. Logically, dot output shape has lhs non-contracting dimensions
// followed by rhs non-contracting dimensions that have to be swapped, but
// that's countered by the layout.
absl::StatusOr<HloDotInstruction*> MakeDotWithSwappedOperands(
    HloInstruction* dot) {
  HloComputation* computation = dot->parent();

  ASSIGN_OR_RETURN(DotOperandDims lhs_dims,
                   DotOperandDims::FromDotOperand(dot, 0));
  ASSIGN_OR_RETURN(DotOperandDims rhs_dims,
                   DotOperandDims::FromDotOperand(dot, 1));

  const size_t num_batch_dims = lhs_dims.Rank(DotOperandDims::kBatch);
  const size_t num_lhs_noncontracting_dims =
      lhs_dims.Rank(DotOperandDims::kNonContracting);
  const size_t num_rhs_noncontracting_dims =
      rhs_dims.Rank(DotOperandDims::kNonContracting);

  std::vector<int64_t> out_shape_permutation;
  out_shape_permutation.reserve(dot->shape().dimensions().size());
  auto fill_permutation = [&](int64_t count, int64_t start) {
    while (count--) {
      out_shape_permutation.push_back(start++);
    }
  };
  // The output shape of a dot is batch dimensions, then lhs non-contracting,
  // then rhs non-contracting. Batch dimensions stay where they were. and
  // contracting dimensions of lhs and rhs swapped.
  fill_permutation(num_batch_dims, 0);
  fill_permutation(num_rhs_noncontracting_dims,
                   num_batch_dims + num_lhs_noncontracting_dims);
  fill_permutation(num_lhs_noncontracting_dims, num_batch_dims);
  const Shape new_dot_shape =
      ShapeUtil::ReorderLogicalDimensions(dot->shape(), out_shape_permutation);

  ASSIGN_OR_RETURN(
      DotDimensionNumbers new_dot_dims,
      DotOperandDims::CreateDotDimensionNumbers(rhs_dims, lhs_dims));

  return DynCast<HloDotInstruction>(computation->AddInstruction(
      HloInstruction::CreateDot(new_dot_shape, dot->mutable_operand(1),
                                dot->mutable_operand(0), new_dot_dims,
                                dot->precision_config()),
      &dot->metadata()));
}

// Swaps operands of a dot instruction in a fusion. This is done by swapping the
// operands of the dot instruction, which keeps the same physical output layout,
// and then bitcasting the result to the original logical shape.
absl::Status SwapDotOperandsInFusion(HloComputation* computation) {
  HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  ASSIGN_OR_RETURN(HloDotInstruction * new_dot,
                   MakeDotWithSwappedOperands(dot));
  HloInstruction* new_bitcast = computation->AddInstruction(
      HloInstruction::CreateBitcast(dot->shape(), new_dot), &dot->metadata());
  RETURN_IF_ERROR(dot->ReplaceAllUsesWith(new_bitcast));
  RETURN_IF_ERROR(computation->RemoveInstruction(dot));
  return absl::OkStatus();
}

bool HasCodeGeneratingInstructions(const HloInstruction* instruction) {
  while (!instruction->operands().empty()) {
    // Skip instructions that are likely to just affect indexing or layout
    // rather than result in actual computation.
    switch (instruction->opcode()) {
      case HloOpcode::kBitcast:
      case HloOpcode::kBroadcast:
      case HloOpcode::kConstant:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kParameter:
      case HloOpcode::kReshape:
      case HloOpcode::kTranspose:
        break;
      default:
        // Any other instruction is considered code generating.
        return true;
    }
    instruction = instruction->operand(0);
  }
  return false;
}

// There are two reasons to swap operands:
// 1. If one side performs computation and the other doesn't, we want the
// "computing" side to be the lhs. wgmma supports lhs in registers, and
// computation would happen in registers too, so putting it to lhs avoids an
// extra roundtrip to a shared memory.
// 2. wgmma instruction only supports 64 for the M (lhs non-contracting)
// dimension, so if it's smaller, move it to the rhs that supports smaller
// powers of two.
absl::StatusOr<bool> ShouldSwapOperands(const HloInstruction* instr) {
  const HloDotInstruction* dot = DynCast<HloDotInstruction>(instr);
  if (dot == nullptr) {
    return false;
  }
  const bool lhs_has_code = HasCodeGeneratingInstructions(dot->operand(0));
  const bool rhs_has_code = HasCodeGeneratingInstructions(dot->operand(1));

  ASSIGN_OR_RETURN(DotOperandDims lhs_dims,
                   DotOperandDims::FromDotOperand(dot, /*operand_number=*/0));
  ASSIGN_OR_RETURN(DotOperandDims rhs_dims,
                   DotOperandDims::FromDotOperand(dot, /*operand_number=*/1));
  const int64_t lhs_size = lhs_dims.TotalSize(DotOperandDims::kNonContracting);
  const int64_t rhs_size = rhs_dims.TotalSize(DotOperandDims::kNonContracting);

  if (lhs_size < 64 && rhs_size >= 64) {
    return true;
  }
  if (!lhs_has_code && rhs_has_code && rhs_size >= 64) {
    return true;
  }
  return false;
}

absl::StatusOr<bool> MaybeSwapOperands(HloComputation* computation) {
  HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  if (dot == nullptr) {
    return false;
  }
  ASSIGN_OR_RETURN(const bool should_swap_operands, ShouldSwapOperands(dot));
  if (!should_swap_operands) {
    return false;
  }
  RETURN_IF_ERROR(SwapDotOperandsInFusion(computation));
  return true;
}

}  // namespace

absl::StatusOr<bool> GemmFusionSwapOperands::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (!IsTritonFusedComputation(*computation)) {
      continue;
    }
    ASSIGN_OR_RETURN(const bool changed, MaybeSwapOperands(computation));
    any_changed |= changed;
  }
  return any_changed;
}

}  // namespace gpu
}  // namespace xla
