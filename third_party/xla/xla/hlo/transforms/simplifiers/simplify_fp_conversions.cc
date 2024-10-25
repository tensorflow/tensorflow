/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/simplify_fp_conversions.h"

#include <cstddef>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Simplifies floating-point conversions `A -> B -> C -> D` as `A -> D`.
absl::StatusOr<bool> RunOnComputation(HloComputation& computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation.MakeInstructionPostOrder()) {
    HloInstruction* input = instruction;
    size_t convert_chain_length = 0;

    while (input->opcode() == HloOpcode::kConvert &&
           primitive_util::IsFloatingPointType(input->shape().element_type())) {
      input = input->mutable_operand(0);
      ++convert_chain_length;
    }

    if (convert_chain_length < 2) {
      continue;
    }

    if (instruction->shape().element_type() == input->shape().element_type()) {
      TF_RETURN_IF_ERROR(
          instruction->parent()->ReplaceInstruction(instruction, input));
    } else {
      TF_RETURN_IF_ERROR(instruction->parent()->ReplaceWithNewInstruction(
          instruction,
          HloInstruction::CreateConvert(instruction->shape(), input)));
    }
    changed = true;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> SimplifyFPConversions::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, absl::StrFormat("SimplifyFPConversions::Run() with before:\n%s",
                         module->ToString()));
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(*computation));
    changed |= comp_changed;
  }
  XLA_VLOG_LINES(2,
                 absl::StrFormat("SimplifyFPConversions::Run() with after:\n%s",
                                 module->ToString()));
  return changed;
}

}  // namespace xla
