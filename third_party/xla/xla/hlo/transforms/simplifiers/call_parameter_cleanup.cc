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

#include "xla/hlo/transforms/simplifiers/call_parameter_cleanup.h"

#include <memory>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {

// Construct a mapping from parameter numbers in the old computation to
// parameter numbers in the new computation. This is basically a compaction of
// the parameters after skipping the ones we'll remove.
// Also figures out if we need to adjust the parameters (for dead and pure
// pass-through parameters) and the root (for any kind of pass-through).
absl::flat_hash_map<int, int> BuildParameterMap(HloComputation* computation,
                                                bool& adjust_params,
                                                bool& adjust_root) {
  adjust_params = false;
  adjust_root = false;

  absl::flat_hash_map<int, int> old_to_new_parameter_number;
  int curr_old = 0, curr_new = 0;
  for (HloInstruction* parameter : computation->parameter_instructions()) {
    bool dead = false;
    if (parameter->users().empty()) {
      // Case 1: Dead parameter, we want to remove it.
      dead = true;
    } else {
      bool found_root_use = false;
      for (HloInstruction* user : parameter->users()) {
        if (user == computation->root_instruction()) {
          found_root_use = true;
          break;
        }
      }
      if (found_root_use) {
        // Case 2: Pass-through parameter, we want to remove it from the root
        // tuple and forward the users to the call operand.
        adjust_root = true;
        if (parameter->users().size() == 1) {
          // Case 2b: Pure pass-through parameter, we want to remove it from
          // both the root tuple *and* the parameter list.
          dead = true;
        }
      }
    }

    if (dead) {
      adjust_params = true;
    } else {
      old_to_new_parameter_number[curr_old] = curr_new;
      ++curr_new;
    }
    ++curr_old;
  }

  return old_to_new_parameter_number;
}

// Similarly, construct a mapping from output numbers (i.e. tuple indices) in
// the original computation to output numbers in the new computation, by
// skipping the ones we'll remove.
// Also collects the outputs we want to keep into `new_outputs`.
absl::flat_hash_map<int, int> BuildOutputMap(
    HloComputation* computation, std::vector<HloInstruction*>& new_outputs) {
  absl::flat_hash_map<int, int> old_to_new_output_number;
  int curr_old = 0, curr_new = 0;
  for (HloInstruction* output : computation->root_instruction()->operands()) {
    if (output->opcode() == HloOpcode::kParameter) {
      ++curr_old;
      continue;
    }
    old_to_new_output_number[curr_old] = curr_new;
    ++curr_old;
    ++curr_new;
    new_outputs.push_back(output);
  }
  return old_to_new_output_number;
}

absl::Status ReplaceCallSite(
    HloInstruction* old_call, HloComputation* new_computation,
    const absl::flat_hash_map<int, int>& old_to_new_parameter_number,
    const absl::flat_hash_map<int, int>& old_to_new_output_number,
    bool adjust_root) {
  // Create a new call instruction with the new computation and new parameters.
  std::vector<HloInstruction*> new_call_operands;
  new_call_operands.reserve(old_call->operands().size());

  for (int i = 0; i < old_call->operands().size(); ++i) {
    if (old_to_new_parameter_number.find(i) !=
        old_to_new_parameter_number.end()) {
      new_call_operands.push_back(old_call->mutable_operand(i));
    }
  }

  HloComputation* enclosing_computation = old_call->parent();
  HloInstruction* new_call =
      enclosing_computation->AddInstruction(old_call->CloneWithNewOperands(
          new_computation->root_instruction()->shape(), new_call_operands));
  new_call->set_to_apply(new_computation);

  // If we didn't remove any pass-through parameters, we're done with this
  // callsite. Note that we can't unconditionally replace here, because the
  // output will create a mismatch.
  if (!adjust_root) {
    return old_call->ReplaceAllUsesWith(new_call);
  }

  // The old call produced a tuple. To ensure the shapes match up, create a new
  // tuple instruction with the right shape, and populate it based on the call's
  // operands (for pass-through parameters) and the new call's outputs (for
  // everything else). This creates some cruft, but the tuple simplifier will
  // clean it up later.
  HloInstruction* old_root = old_call->to_apply()->root_instruction();
  std::vector<HloInstruction*> tuple_inputs;
  for (int i = 0; i < old_root->operands().size(); ++i) {
    auto iter = old_to_new_output_number.find(i);
    if (iter != old_to_new_output_number.end()) {
      HloInstruction* gte = enclosing_computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_call, iter->second));
      tuple_inputs.push_back(gte);
    } else {
      tuple_inputs.push_back(
          old_call->mutable_operand(old_root->operand(i)->parameter_number()));
    }
  }

  HloInstruction* new_tuple = enclosing_computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_inputs));
  return old_call->ReplaceAllUsesWith(new_tuple);
}

absl::StatusOr<bool> RemoveDeadParameters(HloComputation* computation) {
  bool adjust_params, adjust_root;
  absl::flat_hash_map<int, int> old_to_new_parameter_number =
      BuildParameterMap(computation, adjust_params, adjust_root);

  // If we don't need to adjust anything, we're done.
  if (!adjust_params && !adjust_root) {
    return false;
  }

  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      replacements;
  // If we're removing parameters, we need to (a) replace the ones being removed
  // with null, and (b) adjust the parameter numbers on the remaining ones so
  // that we don't have "holes".
  if (adjust_params) {
    for (HloInstruction* parameter : computation->parameter_instructions()) {
      auto iter =
          old_to_new_parameter_number.find(parameter->parameter_number());
      if (iter == old_to_new_parameter_number.end()) {
        replacements.insert({parameter, nullptr});
      } else {
        replacements.insert({parameter, HloInstruction::CreateParameter(
                                            iter->second, parameter->shape(),
                                            parameter->name())});
      }
    }
  }

  HloComputation* new_computation;
  absl::flat_hash_map<int, int> old_to_new_output_number;
  if (adjust_root) {
    replacements.insert({computation->root_instruction(), nullptr});
    std::vector<HloInstruction*> new_outputs;
    old_to_new_output_number = BuildOutputMap(computation, new_outputs);
    new_computation = computation->parent()->AddEmbeddedComputation(
        computation->CloneWithReplacements(
            &replacements, /*extra_parameters=*/{},
            /*context=*/nullptr, /*suffix=*/"undead",
            /*new_root=*/new_outputs));
  } else {
    // Don't fill old_to_new_output_number here, we won't need it.
    new_computation = computation->parent()->AddEmbeddedComputation(
        computation->CloneWithReplacements(
            &replacements, /*extra_parameters=*/{},
            /*context=*/nullptr, /*suffix=*/"undead"));
  }

  // The new call computation is ready, now make all the call sites use it.
  for (HloInstruction* old_call : computation->caller_instructions()) {
    TF_RETURN_IF_ERROR(ReplaceCallSite(old_call, new_computation,
                                       old_to_new_parameter_number,
                                       old_to_new_output_number, adjust_root));
  }

  return true;
}

bool ShouldProcessComputation(HloComputation* computation) {
  // Only process computations with tuple roots. In theory we could also remove
  // completely dead parameters from a computation with a non-tuple root, but
  // since pass-through is only a thing for tuples, and it complicates the code,
  // we don't bother for now.
  if (computation->root_instruction()->opcode() != HloOpcode::kTuple) {
    return false;
  }
  for (HloInstruction* instruction : computation->caller_instructions()) {
    if (instruction->opcode() != HloOpcode::kCall) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> CallParameterCleanup::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloComputation*> computations_to_process;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (ShouldProcessComputation(computation)) {
      computations_to_process.push_back(computation);
    }
  }

  bool changed = false;
  for (HloComputation* computation : computations_to_process) {
    TF_ASSIGN_OR_RETURN(bool removed, RemoveDeadParameters(computation));
    changed |= removed;
  }
  return changed;
}

}  // namespace xla
