/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/pdl_launch_annotation.h"

#include <optional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/frontend_attributes.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"

namespace xla {
namespace gpu {

namespace {

bool OriginatesFrom(const HloInstruction& operand,
                    const HloInstruction& producer) {
  const HloInstruction* current = &operand;
  while (current != &producer) {
    switch (current->opcode()) {
      case HloOpcode::kBitcast:
        current = current->operand(0);
        break;
      case HloOpcode::kGetTupleElement: {
        const HloInstruction* src = current->operand(0);
        if (src->opcode() == HloOpcode::kTuple) {
          current = src->operand(current->tuple_index());
        } else {
          current = src;
        }
        break;
      }
      default:
        return false;
    }
  }
  return true;
}

std::optional<HloInstruction*> GetSuccessorFusion(
    const std::vector<HloInstruction*>& instructions, int from_index) {
  for (int i = from_index + 1; i < instructions.size(); ++i) {
    HloInstruction* next = instructions[i];
    if (HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                         HloOpcode::kTuple, HloOpcode::kGetTupleElement,
                         HloOpcode::kBitcast>(next)) {
      continue;
    }
    if (!HloPredicateIsOp<HloOpcode::kFusion>(next)) {
      return std::nullopt;
    }
    return next;
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<bool> PdlLaunchAnnotationPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  TF_RET_CHECK(module->has_schedule());

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RET_CHECK(module->schedule().is_computation_scheduled(comp));
    const auto& instructions = module->schedule().sequence(comp).instructions();

    for (auto [i, hlo] : llvm::enumerate(instructions)) {
      if (DoesPdlLaunch(*hlo) || !IsTritonGemm(*hlo)) {
        continue;
      }

      std::optional<HloInstruction*> maybe_successor =
          GetSuccessorFusion(instructions, i);
      if (!maybe_successor.has_value()) {
        continue;
      }
      HloInstruction& successor = **maybe_successor;

      hlo->set_frontend_attribute(kXlaPdlLaunch, "true");
      changed = true;

      const absl::flat_hash_set<int> existing_no_invariant_operands =
          NonInvariantOperands(successor);
      std::vector<int> new_no_invariant_operands;
      for (int op_idx = 0; op_idx < successor.operand_count(); ++op_idx) {
        if (existing_no_invariant_operands.contains(op_idx) ||
            OriginatesFrom(*successor.operand(op_idx), *hlo)) {
          new_no_invariant_operands.push_back(op_idx);
        }
      }
      if (new_no_invariant_operands.size() >
          existing_no_invariant_operands.size()) {
        successor.set_frontend_attribute(
            kXlaNoInvariantOperands,
            absl::StrJoin(new_no_invariant_operands, ","));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
