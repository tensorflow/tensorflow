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

#include "tensorflow/compiler/xla/service/transpose_folding.h"

#include <vector>

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

bool IsOperandFoldableToDot(const HloInstruction& hlo) {
  return hlo.IsRank2Transpose() &&
         hlo.user_count() == 1;  // The dot is its only user.
}

bool CanFoldOperandsIntoDot(
    const HloInstruction& dot,
    const TransposeFolding::IsTransposableGemmFn& is_transposable_gemm) {
  if (HloOpcode::kDot != dot.opcode()) {
    return false;
  }

  if (!is_transposable_gemm(dot)) {
    return false;
  }

  const HloInstruction* lhs = dot.operand(0);
  const HloInstruction* rhs = dot.operand(1);
  bool lhs_foldable = IsOperandFoldableToDot(*lhs);
  bool rhs_foldable = IsOperandFoldableToDot(*rhs);
  if (!lhs_foldable && !rhs_foldable) {
    return false;
  }
  return true;
}

// Folds the operands of `dot` that are foldable transposes. `computation` is
// the parent HLO computation of `dot`. `module` is the parent HloModule of
// `computation`.
//
// Returns whether the module is changed.
bool FoldTransposeIntoDot(HloInstruction* dot, HloComputation* computation) {
  std::vector<HloInstruction*> instructions_to_fuse(1, dot);
  for (HloInstruction* operand : dot->operands()) {
    if (IsOperandFoldableToDot(*operand)) {
      instructions_to_fuse.push_back(operand);
    }
  }

  // Early-exit if no operands are foldable.
  if (instructions_to_fuse.size() == 1) {
    return false;
  }

  computation->CreateFusionInstruction(
      instructions_to_fuse, HloInstruction::FusionKind::kTransposeDot);
  return true;
}

}  // namespace

TransposeFolding::TransposeFolding(IsTransposableGemmFn is_transposable_gemm)
    : is_transposable_gemm_(std::move(is_transposable_gemm)) {}

StatusOr<bool> TransposeFolding::Run(HloModule* module) {
  // Modifying the graph while traversing is dangerous, so we find all folding
  // opportunities before actually folding them.
  HloComputation* entry_computation = module->entry_computation();

  std::vector<HloInstruction*> foldable_dots;
  auto visit_fn = [this, &foldable_dots](HloInstruction* instruction) {
    if (CanFoldOperandsIntoDot(*instruction, is_transposable_gemm_)) {
      foldable_dots.emplace_back(instruction);
    }
    return tensorflow::Status::OK();
  };
  TF_RETURN_IF_ERROR(entry_computation->root_instruction()->Accept(visit_fn));

  bool changed = false;
  for (HloInstruction* dot : foldable_dots) {
    changed |= FoldTransposeIntoDot(dot, entry_computation);
  }
  return changed;
}

}  // namespace xla
