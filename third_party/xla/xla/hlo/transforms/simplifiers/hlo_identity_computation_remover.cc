/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_identity_computation_remover.h"

#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/util.h"

namespace xla {

namespace {
// Returns true if the given instruction is an "identity" value. An identity
// value is a value that comes from a parameter without any change.
bool IsIdentityValue(const HloInstruction* instruction,
                     absl::flat_hash_map<const HloInstruction*, bool>* map) {
  if (auto it = map->find(instruction); it != map->end()) {
    return it->second;
  }

  (*map)[instruction] = false;

  bool result = false;
  if (!instruction->HasSideEffect()) {
    switch (instruction->opcode()) {
      case HloOpcode::kParameter:
        result = true;
        break;
      case HloOpcode::kTuple: {
        bool all_operands_are_identity = true;
        for (const HloInstruction* operand : instruction->operands()) {
          if (!IsIdentityValue(operand, map)) {
            all_operands_are_identity = false;
            break;
          }
        }
        result = all_operands_are_identity;
        break;
      }
      default:
        result = false;
        break;
    }
  }

  (*map)[instruction] = result;
  return result;
}

void FindIdentityComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::flat_hash_set<HloComputation*>& identity_computations) {
  for (HloComputation* computation : module->computations()) {
    if (!execution_threads.empty() &&
        !execution_threads.contains(computation->execution_thread())) {
      continue;
    }
    if (HloIdentityComputationRemover::IsIdentityComputation(computation)) {
      identity_computations.insert(computation);
    }
  }
}

// Creates a replacement for the given identity instruction, to be used in
// the caller.
HloInstruction* GetReplacement(HloInstruction* caller,
                               const HloInstruction* root) {
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> replacements;
  std::function<HloInstruction*(const HloInstruction*)> get_replacement_for =
      [&](const HloInstruction* instruction) -> HloInstruction* {
    if (auto it = replacements.find(instruction); it != replacements.end()) {
      return it->second;
    }

    HloInstruction* replacement = nullptr;
    switch (instruction->opcode()) {
      case HloOpcode::kParameter:
        replacement = caller->mutable_operand(instruction->parameter_number());
        break;
      case HloOpcode::kTuple: {
        std::vector<HloInstruction*> new_operands;
        new_operands.reserve(instruction->operand_count());
        for (const HloInstruction* operand : instruction->operands()) {
          new_operands.push_back(get_replacement_for(operand));
        }
        replacement = caller->parent()->AddInstruction(
            HloInstruction::CreateTuple(new_operands));
        break;
      }
      default:
        LOG(FATAL) << "Unexpected opcode in identity computation: "
                   << instruction->opcode();
    }
    replacements[instruction] = replacement;
    return replacement;
  };

  return get_replacement_for(root);
}

// Bypasses all calls to `computation`, which must be an identity computation.
// Returns true if any callers were changed.
bool BypassCallers(
    HloComputation* computation, CallGraph* call_graph,
    absl::flat_hash_set<HloComputation*>& updated_caller_computations) {
  bool changed = false;
  const HloInstruction* root = computation->root_instruction();
  for (HloInstruction* caller :
       call_graph->GetComputationCallers(computation)) {
    HloComputation* caller_computation = caller->parent();
    HloInstruction* replacement = GetReplacement(caller, root);
    if (caller->opcode() == HloOpcode::kCall) {
      TF_CHECK_OK(caller->parent()
                      ->ReplaceInstruction(caller, replacement,
                                           /*preserve_sharding=*/false,
                                           /*relay_control_dependency=*/true)
                      .status());
      changed = true;
    } else if (caller->opcode() == HloOpcode::kAsyncStart) {
      for (HloInstruction* user : caller->users()) {
        if (user->opcode() == HloOpcode::kAsyncDone) {
          TF_CHECK_OK(user->parent()
                          ->ReplaceInstruction(
                              user, replacement, /*preserve_sharding=*/false,
                              /*relay_control_dependency=*/true)
                          .status());
          changed = true;
        }
      }
    }
    if (changed) {
      updated_caller_computations.insert(caller_computation);
    }
  }
  return changed;
}

void RunDceAndFindNewIdentityComputations(
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::flat_hash_set<HloComputation*>& identity_computations,
    CallGraph* call_graph, std::vector<HloComputation*>& worklist,
    absl::flat_hash_set<HloComputation*>& updated_caller_computations) {
  HloDCE dce;
  for (HloComputation* caller_computation : updated_caller_computations) {
    VLOG(2) << "HloIdentityComputationRemover updated computation "
            << caller_computation->ToString();
    dce.RunOnComputation(caller_computation, false, call_graph).value();

    if (HloIdentityComputationRemover::IsIdentityComputation(
            caller_computation) &&
        (execution_threads.empty() ||
         execution_threads.contains(caller_computation->execution_thread())) &&
        identity_computations.insert(caller_computation).second) {
      worklist.push_back(caller_computation);
    }
  }
  updated_caller_computations.clear();
}

bool BypassIdentityComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::flat_hash_set<HloComputation*>& identity_computations) {
  bool changed = false;
  std::vector<HloComputation*> worklist;
  worklist.assign(identity_computations.begin(), identity_computations.end());

  if (worklist.empty()) {
    return false;
  }

  absl::flat_hash_set<HloComputation*> updated_caller_computations;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  while (!worklist.empty()) {
    HloComputation* computation = worklist.back();
    worklist.pop_back();

    if (BypassCallers(computation, call_graph.get(),
                      updated_caller_computations)) {
      changed = true;
    }

    if (worklist.empty() && !updated_caller_computations.empty()) {
      RunDceAndFindNewIdentityComputations(
          execution_threads, identity_computations, call_graph.get(), worklist,
          updated_caller_computations);
    }
  }
  return changed;
}
}  // namespace

// An identity computation is a computation whose root is an identity value.
bool HloIdentityComputationRemover::IsIdentityComputation(
    HloComputation* computation) {
  if (computation->IsEntryComputation()) {
    return false;
  }
  absl::flat_hash_map<const HloInstruction*, bool> map;
  return IsIdentityValue(computation->root_instruction(), &map);
}

absl::Status HloIdentityComputationRemover::CleanUp(HloModule* module) {
  HloDCE dce;
  return dce.Run(module).status();
}

// HloIdentityComputationRemover is a pass that removes identity computations
// from the HLO module. An identity computation is a computation that simply
// returns its parameter or a tuple of its parameters.
absl::StatusOr<bool> HloIdentityComputationRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Before identity_computation_remover; threads: "
          << absl::StrJoin(execution_threads, ",");
  XLA_VLOG_LINES(2, module->ToString());

  // Step 1: Find identity computations.
  absl::flat_hash_set<HloComputation*> identity_computations;
  FindIdentityComputations(module, execution_threads, identity_computations);

  // Step 2: Bypass identity computation calls.
  bool changed = BypassIdentityComputations(module, execution_threads,
                                            identity_computations);

  // Step 3: Cleanup.
  if (changed) {
    if (run_cleanup_) {
      TF_RETURN_IF_ERROR(CleanUp(module));
    }
    VLOG(2) << "After identity_computation_remover:";
    XLA_VLOG_LINES(2, module->ToString());
  }

  return changed;
}

}  // namespace xla
