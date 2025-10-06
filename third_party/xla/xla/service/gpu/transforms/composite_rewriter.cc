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

#include "xla/service/gpu/transforms/composite_rewriter.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> CompositeRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kCall) {
      continue;
    }
    auto call = Cast<HloCallInstruction>(instruction);
    if (!call->is_composite()) {
      continue;
    }
    if (!call->has_frontend_attributes()) {
      VLOG(3) << "No frontend attributes";
      continue;
    }
    auto attrs = call->frontend_attributes().map();
    auto key = "composite.name";
    if (!attrs.contains(key) || attrs.at(key) != "xla.scaled_dot") {
      VLOG(3) << key << " is not xla.scaled_dot: " << attrs.at(key);
      continue;
    }
    DotDimensionNumbers dot_dimension_numbers;
    dot_dimension_numbers.add_lhs_contracting_dimensions(2);
    dot_dimension_numbers.add_rhs_contracting_dimensions(2);
    dot_dimension_numbers.add_lhs_batch_dimensions(0);
    dot_dimension_numbers.add_rhs_batch_dimensions(0);

    auto* scaled_dot =
        computation->AddInstruction(HloInstruction::CreateScaledDot(
            call->shape(), call->mutable_operand(0), call->mutable_operand(1),
            call->mutable_operand(2), call->mutable_operand(3),
            dot_dimension_numbers, PrecisionConfig{}));
    TF_RETURN_IF_ERROR(call->ReplaceAllUsesWith(scaled_dot));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(call));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> CompositeRewriter::Run(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
