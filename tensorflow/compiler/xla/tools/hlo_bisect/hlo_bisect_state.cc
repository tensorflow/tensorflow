/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/hlo_bisect/hlo_bisect_state.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace bisect {
namespace {

// Returns the modified post order of the instructions for the computation. In
// particular, the returned vector of instructions has all the parameter
// instructions in the front and the rest are in post order.
std::vector<HloInstruction*> GetModifiedInstructionPostOrder(
    HloComputation* computation) {
  std::vector<HloInstruction*> instructions(
      computation->parameter_instructions().begin(),
      computation->parameter_instructions().end());
  absl::c_copy_if(computation->MakeInstructionPostOrder(),
                  std::back_inserter(instructions),
                  [&](const HloInstruction* instr) {
                    return instr->opcode() != HloOpcode::kParameter;
                  });
  return instructions;
}

// Changes the module by replacing the original root instruction of the entry
// computation with a new root instruction that is a tuple containing the values
// in `outputs`.
Status MorphModuleWithOutputs(HloModule* module,
                              absl::Span<HloInstruction* const> outputs) {
  HloComputation* entry_computation = module->entry_computation();
  HloInstruction* new_root = outputs.size() == 1
                                 ? outputs[0]
                                 : entry_computation->AddInstruction(
                                       HloInstruction::CreateTuple(outputs));

  entry_computation->set_root_instruction(new_root, true);
  *module->mutable_entry_computation_layout() =
      module->compute_computation_layout();

  HloDCE dce;
  StatusOr<bool> dce_result = dce.Run(module);
  return dce_result.status();
}

// Changes the module by keeping only the provided instructions of the entry
// computation (should be sorted in the modified instruction post order),
// inserting a new root instruction to keep all values live.
Status MorphModuleWithInstructions(
    HloModule* module, absl::Span<HloInstruction* const> instructions) {
  ConstHloInstructionSet in_range_instructions(instructions.begin(),
                                               instructions.end());
  auto keep_result = [&](const HloInstruction* instruction) {
    return instruction->opcode() != HloOpcode::kParameter &&
           !absl::c_any_of(instruction->users(),
                           [&](const HloInstruction* user) {
                             return in_range_instructions.count(user) != 0;
                           });
  };

  // If an instruction doesn't have a user within the range, add the result of
  // the instruction to the outputs to keep the value live.
  std::vector<HloInstruction*> outputs;
  absl::c_copy_if(instructions, std::back_inserter(outputs), keep_result);
  return MorphModuleWithOutputs(module, outputs);
}

Status MorphModuleWithInstructions(HloModule* module, size_t num_instructions) {
  std::vector<HloInstruction*> ordered_instructions =
      GetModifiedInstructionPostOrder(module->entry_computation());
  HloInstruction* const* instructions_begin = &ordered_instructions.front();
  return MorphModuleWithInstructions(
      module, absl::MakeSpan(instructions_begin, num_instructions));
}

// Changes the module by replacing some instructions in the entry computation
// with literals.
Status MorphModuleWithLiterals(
    HloModule* module, absl::flat_hash_map<std::string, Literal> literal_map) {
  HloComputation* entry_computation = module->entry_computation();

  // Iterate over instructions, as lookup by instruction name is linear.
  absl::flat_hash_map<HloInstruction*, Literal> replace_map;
  for (HloInstruction* instruction : entry_computation->instructions()) {
    auto it = literal_map.find(instruction->name());
    if (it != literal_map.end()) {
      replace_map.emplace(instruction, std::move(it->second));
    }
  }
  for (auto& [instruction, literal] : replace_map) {
    if (!instruction->IsDead()) {
      HloInstruction* new_instruction = entry_computation->AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      Status replace_status =
          entry_computation->ReplaceInstruction(instruction, new_instruction);
      TF_RETURN_IF_ERROR(replace_status);
    }
  }

  xla::HloDCE dce;
  StatusOr<bool> dce_status = dce.Run(module);
  return dce_status.status();
}

// We shouldn't replace a constant in a module with another constant for
// bisecting. We should also avoid straightforwardly replacing compound values
// produced by kTuple, as they may have constant values inside.

bool InstructionNotReplaceableWithConstant(HloInstruction* instruction) {
  return instruction->shape().is_dynamic() ||
         instruction->opcode() == HloOpcode::kConstant ||
         instruction->opcode() == HloOpcode::kTuple ||
         instruction->opcode() == HloOpcode::kParameter;
}

}  // namespace

StatusOr<bool> HloBisectState::ShouldProcess() {
  // Running the unmodified module should trigger the bug checker.
  return RunModule(*module_);
}

StatusOr<bool> HloBisectState::TrimEntryComputation() {
  bool changed_in_loop = false;
  bool changed = false;
  for (int iter = 0; changed || iter < 2; iter++) {
    if (iter % 2 == 0) {
      VLOG(2) << "Trimming by outputs, iteration " << iter;
      TF_ASSIGN_OR_RETURN(changed, TrimByOutputs());
    } else {
      VLOG(2) << "Trimming by instructions, iteration " << iter;
      TF_ASSIGN_OR_RETURN(changed, TrimByInstructions());
    }
    changed_in_loop |= changed;
  }
  VLOG(2) << "Trimming by replacing instructions with literals";
  TF_ASSIGN_OR_RETURN(changed, TrimByUsingConstants());
  VLOG(2) << "Final module: " << module_->ToString();
  return changed || changed_in_loop;
}

std::unique_ptr<xla::HloModule>&& HloBisectState::GetResult() {
  return std::move(module_);
}

StatusOr<bool> HloBisectState::RunModule(const HloModule& module) {
  VLOG(3) << "Modified module: " << module.ToString();

  // Run the modified module with the bug checker.
  StatusOr<bool> bug_result = bug_checker_->Run(module);
  TF_RETURN_IF_ERROR(bug_result.status());
  VLOG(3) << "Bug checker result: " << bug_result.value();

  // Update foldable instructions data.
  if (!bug_result.value()) {
    for (HloInstruction* instr : module.entry_computation()->instructions()) {
      foldable_instructions_.emplace(instr->name());
    }
    for (auto& [key, value] : bug_checker_->GetResults()) {
      foldable_instructions_values_[key] = std::move(value);
    }
  }
  return bug_result;
}

StatusOr<bool> HloBisectState::TrimByOutputs() {
  // Only available if the root instruction is a tuple.
  HloInstruction* root_instruction =
      module_->entry_computation()->root_instruction();
  if (root_instruction->opcode() != HloOpcode::kTuple ||
      root_instruction->operand_count() < 2) {
    return false;
  }

  // Run the modified module and return the error state.
  auto run_modified = [&](int64_t start, int64_t end) -> StatusOr<bool> {
    std::unique_ptr<HloModule> new_module = module_->Clone(/*suffix=*/"");
    HloInstruction* const* new_operands =
        new_module->entry_computation()->root_instruction()->operands().begin();
    TF_RETURN_IF_ERROR(MorphModuleWithOutputs(
        new_module.get(),
        absl::MakeSpan(new_operands + start, end - start + 1)));
    return RunModule(*new_module);
  };

  // Binary search for the operands range that exhibits a bug.
  int64_t bisect_low = 0;
  int64_t bisect_high = root_instruction->operand_count() - 1;
  while (bisect_low < bisect_high) {
    int64_t cur = bisect_low + (bisect_high - bisect_low) / 2;
    VLOG(2) << "Number of outputs: " << (cur - bisect_low + 1) << " ["
            << bisect_low << ".." << cur << "]";
    TF_ASSIGN_OR_RETURN(bool has_bug, run_modified(bisect_low, cur));
    if (has_bug) {
      bisect_high = cur;
    } else {
      TF_ASSIGN_OR_RETURN(has_bug, run_modified(cur + 1, bisect_high));
      if (has_bug) {
        bisect_low = cur + 1;
      } else {
        break;
      }
    }
  }

  // Update the current module and verify that the bug is present, if changed.
  bool changed =
      (bisect_high - bisect_low) < (root_instruction->operand_count() - 1);
  if (changed) {
    TF_RETURN_IF_ERROR(MorphModuleWithOutputs(
        module_.get(),
        absl::MakeSpan(root_instruction->operands().begin() + bisect_low,
                       bisect_high - bisect_low + 1)));
    TF_RETURN_IF_ERROR(ExpectModuleIsBuggy());
  }
  return changed;
}

StatusOr<bool> HloBisectState::TrimByInstructions() {
  HloComputation* computation = module_->entry_computation();

  // If the root instruction is a tuple, exclude it from the bisect range.
  int64_t upper_bound = computation->instruction_count() -
                        computation->root_instruction()->shape().IsTuple();

  // Binary search for the instructions range that exhibits a bug.
  int64_t bisect_low = computation->num_parameters() - 1;
  int64_t bisect_high = upper_bound;
  while (bisect_low + 1 < bisect_high) {
    int64_t cur = bisect_low + (bisect_high - bisect_low) / 2;
    VLOG(2) << "Number of instructions: " << cur << " (of "
            << computation->instruction_count() << ")";
    std::unique_ptr<HloModule> new_module = module_->Clone(/*suffix=*/"");
    TF_RETURN_IF_ERROR(MorphModuleWithInstructions(new_module.get(), cur));
    TF_ASSIGN_OR_RETURN(bool has_bug, RunModule(*new_module));
    if (has_bug) {
      bisect_high = cur;
    } else {
      bisect_low = cur;
    }
  }

  // Sanity check for the bug checker.
  if (bisect_high == computation->num_parameters()) {
    return InternalError(
        "The checker fails on an empty computation! Something is not right. "
        "Can't bisect.");
  }

  // Update the current module and verify that the bug is present, if changed.
  bool changed = bisect_high < upper_bound;
  if (changed) {
    TF_RETURN_IF_ERROR(MorphModuleWithInstructions(module_.get(), bisect_high));
    TF_RETURN_IF_ERROR(ExpectModuleIsBuggy());
  }
  return changed;
}

StatusOr<bool> HloBisectState::TrimByUsingConstants() {
  // Use random literals for the instructions which do not trigger the bug
  // checker and also didn't get a definitive value from it.
  absl::flat_hash_map<std::string, Literal> literal_map;
  int64_t random_literals_count = 0;
  for (HloInstruction* instr : module_->entry_computation()->instructions()) {
    if (InstructionNotReplaceableWithConstant(instr)) {
      continue;
    }
    if (foldable_instructions_values_.contains(instr->name())) {
      auto it = foldable_instructions_values_.extract(instr->name());
      literal_map.insert(std::move(it));
    } else if (foldable_instructions_.contains(instr->name())) {
      StatusOr<Literal> literal_status = MakeFakeLiteral(instr->shape());
      TF_RETURN_IF_ERROR(literal_status.status());
      literal_map[instr->name()] = std::move(literal_status).value();
      ++random_literals_count;
    }
  }
  VLOG(2) << "Number of literals: " << literal_map.size()
          << " (random: " << random_literals_count << ")";

  // Replace instructions with constants and run the bug checker.
  // It is possible that the random literals will make the bug disappear, in
  // which case the module will not get reduced.
  std::unique_ptr<HloModule> new_module = module_->Clone(/*suffix=*/"");
  TF_RETURN_IF_ERROR(
      MorphModuleWithLiterals(new_module.get(), std::move(literal_map)));
  TF_ASSIGN_OR_RETURN(bool has_bug, RunModule(*new_module));
  if (has_bug) {
    std::swap(module_, new_module);
  }
  return has_bug;
}

Status HloBisectState::ExpectModuleIsBuggy() {
  // Verify that the current module has a bug.
  TF_ASSIGN_OR_RETURN(bool has_bug, RunModule(*module_));
  if (has_bug) {
    return OkStatus();
  }

  // Check for the bug checker stability.
  const int retry_count = 5;
  int bug_count = 0;
  for (int i = 0; i < retry_count; i++) {
    TF_ASSIGN_OR_RETURN(has_bug, bug_checker_->Run(*module_));
    if (has_bug) {
      bug_count++;
    }
  }
  if (bug_count != 0) {
    return InternalErrorStrCat("The checker is non deterministic! (only ",
                               bug_count, " failures seen in ",
                               (retry_count + 1), " runs)");
  }
  return InternalError("We \"lost\" the bug while bisecting!");
}

}  // namespace bisect
}  // namespace xla
