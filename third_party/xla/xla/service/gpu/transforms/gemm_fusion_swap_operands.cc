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

#include "xla/service/gpu/transforms/gemm_fusion_swap_operands.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Swaps operands of a dot instruction, while keeping the same physical output
// layout. Logically, dot output shape has lhs non-contracting dimensions
// followed by rhs non-contracting dimensions that have to be swapped, but
// that's countered by the layout.
HloDotInstruction* MakeDotWithSwappedOperands(HloInstruction* dot) {
  HloComputation* computation = dot->parent();

  const DotDimensionNumbers& dot_dims = dot->dot_dimension_numbers();
  const size_t num_batch_dims = dot_dims.lhs_batch_dimensions_size();
  const size_t num_lhs_noncontracting_dims =
      dot->operand(0)->shape().dimensions_size() - num_batch_dims -
      dot_dims.lhs_contracting_dimensions_size();
  const size_t num_rhs_noncontracting_dims =
      dot->operand(1)->shape().dimensions_size() - num_batch_dims -
      dot_dims.rhs_contracting_dimensions_size();

  std::vector<int64_t> out_shape_permutation;
  out_shape_permutation.reserve(dot->shape().dimensions_size());
  auto fill_permutation = [&](int64_t count, int64_t start) {
    while (count--) out_shape_permutation.push_back(start++);
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

  DotDimensionNumbers new_dot_dims = dot_dims;
  std::swap(*new_dot_dims.mutable_lhs_batch_dimensions(),
            *new_dot_dims.mutable_rhs_batch_dimensions());
  std::swap(*new_dot_dims.mutable_lhs_contracting_dimensions(),
            *new_dot_dims.mutable_rhs_contracting_dimensions());

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
  HloDotInstruction* new_dot = MakeDotWithSwappedOperands(dot);
  HloInstruction* new_bitcast = computation->AddInstruction(
      HloInstruction::CreateBitcast(dot->shape(), new_dot), &dot->metadata());
  TF_RETURN_IF_ERROR(dot->ReplaceAllUsesWith(new_bitcast));
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(dot));
  return absl::OkStatus();
}

bool HasCodeGeneratingInstructions(const HloInstruction* instruction) {
  while (!instruction->operands().empty()) {
    // Skip instruction that are likely to just affect the address computation
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

absl::StatusOr<int64_t> GetNonContractingDimsNumElements(
    const HloInstruction* dot, size_t operand_index) {
  const Shape& shape = dot->operand(operand_index)->shape();
  const DotDimensionNumbers& dot_dims = dot->dot_dimension_numbers();
  const absl::Span<const int64_t> batch_dim_indices =
      operand_index == 0 ? dot_dims.lhs_batch_dimensions()
                         : dot_dims.rhs_batch_dimensions();
  const absl::Span<const int64_t> contracting_dim_indices =
      operand_index == 0 ? dot_dims.lhs_contracting_dimensions()
                         : dot_dims.rhs_contracting_dimensions();
  const DimensionVector noncontracting_dim_indices = GetNonContractingDims(
      shape.dimensions_size(), batch_dim_indices, contracting_dim_indices);
  return absl::c_accumulate(
      noncontracting_dim_indices, int64_t{1},
      [&](int64_t acc, int64_t dim) { return acc * shape.dimensions(dim); });
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
  if (dot == nullptr) return false;
  // Sparsity is generally not symmetric, so we cannot swap operands.
  if (dot->sparse_operands()) return false;
  const bool lhs_has_code = HasCodeGeneratingInstructions(dot->operand(0));
  const bool rhs_has_code = HasCodeGeneratingInstructions(dot->operand(1));
  TF_ASSIGN_OR_RETURN(const int64_t lhs_size, GetNonContractingDimsNumElements(
                                                  dot, /*operand_index=*/0));
  TF_ASSIGN_OR_RETURN(const int64_t rhs_size, GetNonContractingDimsNumElements(
                                                  dot, /*operand_index=*/1));
  if (lhs_size < 64 && rhs_size >= 64) return true;
  if (!lhs_has_code && rhs_has_code && rhs_size >= 64) return true;
  return false;
}

// Triton emitter is not fully symmetric, so it's not possible to emit all
// fusions with swapped dot operands. This function checks if the emitter could
// handle such a fusion.
absl::StatusOr<bool> EmitterCanHandleSwappedOperands(
    const HloInstruction* dot) {
  auto tmp_module = HloModule("tmp", dot->parent()->parent()->config());
  HloCloneContext clone_context(&tmp_module);
  HloComputation* cloned_computation = tmp_module.AddEntryComputation(
      dot->parent()->CloneInContext(clone_context));
  TF_RETURN_IF_ERROR(SwapDotOperandsInFusion(cloned_computation));
  return TritonFusionAnalysis::Execute(*cloned_computation).ok();
}

absl::StatusOr<bool> MaybeSwapOperands(HloComputation* computation) {
  HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  if (dot == nullptr) return false;
  TF_ASSIGN_OR_RETURN(const bool should_swap_operands, ShouldSwapOperands(dot));
  if (!should_swap_operands) return false;
  TF_ASSIGN_OR_RETURN(const bool can_handle_swapped_operands,
                      EmitterCanHandleSwappedOperands(dot));
  if (!can_handle_swapped_operands) return false;
  TF_RETURN_IF_ERROR(SwapDotOperandsInFusion(computation));
  return true;
}

}  // namespace

absl::StatusOr<bool> GemmFusionSwapOperands::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (!IsTritonFusedComputation(*computation)) continue;
    TF_ASSIGN_OR_RETURN(const bool changed, MaybeSwapOperands(computation));
    any_changed |= changed;
  }
  return any_changed;
}

}  // namespace gpu
}  // namespace xla
