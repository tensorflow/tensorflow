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

#include "xla/hlo/transforms/hlo_module_stitcher.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

absl::StatusOr<bool> HloModuleStitcher::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder(execution_threads);

  for (HloComputation* comp : computations) {
    std::vector<HloInstruction*> instructions =
        comp->MakeInstructionPostOrder();
    for (HloInstruction* inst : instructions) {
      if (inst->opcode() == HloOpcode::kCustomCall &&
          inst->custom_call_target() == kMultiModuleCustomCallTarget) {
        std::string sub_module_name = inst->raw_backend_config_string();
        auto it = optimized_modules_.find(sub_module_name);
        if (it == optimized_modules_.end()) {
          return absl::NotFoundError(
              absl::StrCat("Sub-module ", sub_module_name, " not found"));
        }

        const HloModule* sub_module = it->second;
        HloComputation* sub_entry = sub_module->entry_computation();

        if (inst->operand_count() != sub_entry->num_parameters()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Operand count mismatch: custom call has ", inst->operand_count(),
              " operands but sub-module expects ",
              sub_entry->num_parameters()));
        }

        HloCloneContext context(module);
        HloComputation* cloned_sub_entry =
            module->DeepCloneComputation(sub_entry, &context);

        std::vector<HloInstruction*> operands;
        operands.reserve(inst->operand_count());
        for (int i = 0; i < inst->operand_count(); ++i) {
          HloInstruction* operand = inst->mutable_operand(i);
          const Shape& expected_shape =
              cloned_sub_entry->parameter_instruction(i)->shape();
          if (!ShapeUtil::Equal(operand->shape(), expected_shape)) {
            if (!ShapeUtil::Compatible(operand->shape(), expected_shape)) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Incompatible operand shape at index ", i, ": expected ",
                  ShapeUtil::HumanString(expected_shape), ", got ",
                  ShapeUtil::HumanString(operand->shape())));
            }
            operand = comp->AddInstruction(HloInstruction::CreateUnary(
                expected_shape, HloOpcode::kCopy, operand));
          }
          operands.push_back(operand);
        }

        const Shape& result_shape =
            cloned_sub_entry->root_instruction()->shape();
        HloInstruction* call = comp->AddInstruction(HloInstruction::CreateCall(
            result_shape, operands, cloned_sub_entry));
        call->set_frontend_attributes(inst->frontend_attributes());

        HloInstruction* replacement = call;
        if (!ShapeUtil::Equal(result_shape, inst->shape())) {
          if (!ShapeUtil::Compatible(result_shape, inst->shape())) {
            return absl::InvalidArgumentError(
                absl::StrCat("Incompatible result shape: expected ",
                             ShapeUtil::HumanString(inst->shape()), ", got ",
                             ShapeUtil::HumanString(result_shape)));
          }
          replacement = comp->AddInstruction(HloInstruction::CreateUnary(
              inst->shape(), HloOpcode::kCopy, call));
        }

        RETURN_IF_ERROR(inst->ReplaceAllUsesWith(replacement));
        RETURN_IF_ERROR(comp->RemoveInstruction(inst));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
