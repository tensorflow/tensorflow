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

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/call_graph.h"
#include "xla/tsl/platform/status.h"
#include "xla/util.h"

namespace xla {

// Returns true if the computation is an identity computation. An identity
// computation has a single parameter operation as its root.
bool HloIdentityComputationRemover::IsIdentityComputation(
    HloComputation* computation) {
  if (computation->IsEntryComputation()) {
    return false;
  }
  const HloInstruction* root = computation->root_instruction();
  if (root->opcode() != HloOpcode::kParameter) {
    return false;
  }
  if (computation->instruction_count() != 1) {
    return false;
  }
  return true;
}

namespace {

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

void ReplaceCallInstruction(HloInstruction* caller, HloInstruction* operand,
                            bool& changed) {
  CHECK_EQ(caller->opcode(), HloOpcode::kCall);
  CHECK_EQ(caller->operand_count(), 1);
  TF_CHECK_OK(caller->parent()->ReplaceInstruction(caller, operand));
  changed = true;
}

void ReplaceAsyncStartAndDone(HloInstruction* caller, HloInstruction* operand,
                              bool& changed) {
  CHECK_EQ(caller->opcode(), HloOpcode::kAsyncStart);
  CHECK_EQ(caller->operand_count(), 1);
  for (HloInstruction* user : caller->users()) {
    if (user->opcode() == HloOpcode::kAsyncDone) {
      TF_CHECK_OK(user->parent()->ReplaceInstruction(user, operand));
      changed = true;
    }
  }
}

void BypassCallers(
    HloComputation* computation, CallGraph* call_graph, bool& changed,
    absl::flat_hash_set<HloComputation*>& updated_caller_computations) {
  for (HloInstruction* caller :
       call_graph->GetComputationCallers(computation)) {
    HloComputation* caller_computation = caller->parent();
    if (caller->opcode() == HloOpcode::kCall) {
      ReplaceCallInstruction(caller, caller->mutable_operand(0), changed);
    } else if (caller->opcode() == HloOpcode::kAsyncStart) {
      ReplaceAsyncStartAndDone(caller, caller->mutable_operand(0), changed);
    } else {
      continue;
    }
    updated_caller_computations.insert(caller_computation);
  }
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

void BypassIdentityComputations(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::flat_hash_set<HloComputation*>& identity_computations,
    bool& changed) {
  std::vector<HloComputation*> worklist;
  worklist.assign(identity_computations.begin(), identity_computations.end());

  if (worklist.empty()) {
    return;
  }

  absl::flat_hash_set<HloComputation*> updated_caller_computations;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  while (!worklist.empty()) {
    HloComputation* computation = worklist.back();
    worklist.pop_back();

    BypassCallers(computation, call_graph.get(), changed,
                  updated_caller_computations);

    if (worklist.empty() && !updated_caller_computations.empty()) {
      RunDceAndFindNewIdentityComputations(
          execution_threads, identity_computations, call_graph.get(), worklist,
          updated_caller_computations);
    }
  }
}

}  // namespace

void HloIdentityComputationRemover::CleanUp(HloModule* module) {
  if (should_run_dce_) {
    HloDCE dce;
    dce.Run(module).value();
  }
}

absl::StatusOr<bool> HloIdentityComputationRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Before identity_computation_remover; threads: "
          << absl::StrJoin(execution_threads, ",");
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;

  // Step 1: Find identity computations.
  absl::flat_hash_set<HloComputation*> identity_computations;
  FindIdentityComputations(module, execution_threads, identity_computations);

  // Step 2: Bypass identity computation calls.
  BypassIdentityComputations(module, execution_threads, identity_computations,
                             changed);

  // Step 3: Cleanup.
  if (changed) {
    CleanUp(module);
    VLOG(2) << "After identity_computation_remover:";
    XLA_VLOG_LINES(2, module->ToString());
  }

  return changed;
}

}  // namespace xla
