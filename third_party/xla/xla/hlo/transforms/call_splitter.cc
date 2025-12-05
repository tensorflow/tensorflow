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

#include "xla/hlo/transforms/call_splitter.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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

// Returns all instructions in the body that match the boundary predicate.
std::vector<HloInstruction*> GetBoundaryInstructions(
    HloComputation* body, HloPredicate boundary_predicate) {
  std::vector<HloInstruction*> boundary_instructions;
  for (HloInstruction* instruction : body->instructions()) {
    if (boundary_predicate(instruction)) {
      boundary_instructions.push_back(instruction);
    }
  }
  return boundary_instructions;
}

// Returns all instructions that must go into the second call, because they
// depend on the boundary instructions.
absl::flat_hash_set<HloInstruction*> GetSecondCallInstructions(
    HloComputation* body,
    const std::vector<HloInstruction*>& boundary_instructions) {
  absl::flat_hash_set<HloInstruction*> second_call_instructions(
      boundary_instructions.begin(), boundary_instructions.end());
  std::vector<HloInstruction*> worklist(boundary_instructions.begin(),
                                        boundary_instructions.end());
  while (!worklist.empty()) {
    HloInstruction* curr = worklist.back();
    worklist.pop_back();
    auto process = [&](HloInstruction* user) {
      if (second_call_instructions.contains(user)) {
        return;
      }
      second_call_instructions.insert(user);
      worklist.push_back(user);
    };
    for (HloInstruction* user : curr->users()) {
      process(user);
    }
    for (HloInstruction* successor : curr->control_successors()) {
      process(successor);
    }
  }
  return second_call_instructions;
}

// Create new call ops, connect them together, and splice them
// where the original call was.
absl::Status SplitCallSite(HloInstruction* call,
                           HloComputation* first_call_computation,
                           HloComputation* second_call_computation) {
  HloComputation* enclosing_computation = call->parent();
  HloInstruction* first_call =
      enclosing_computation->AddInstruction(call->CloneWithNewOperands(
          first_call_computation->root_instruction()->shape(),
          call->operands()));
  first_call->set_to_apply(first_call_computation);
  std::vector<HloInstruction*> first_call_output_gtes;
  int num_outputs =
      first_call_computation->root_instruction()->shape().tuple_shapes().size();
  first_call_output_gtes.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    first_call_output_gtes.push_back(enclosing_computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(first_call, i)));
  }
  HloInstruction* second_call =
      enclosing_computation->AddInstruction(call->CloneWithNewOperands(
          second_call_computation->root_instruction()->shape(),
          first_call_output_gtes));
  second_call->set_to_apply(second_call_computation);
  return call->ReplaceAllUsesWith(second_call);
}
}  // namespace

std::pair<HloComputation*, HloComputation*> CallSplitter::SplitCallBody(
    HloComputation* body, HloPredicate boundary_predicate) {
  // We need to do several things here:
  // 1. Figure out which instructions go into the first call and which into the
  // second. In particular:
  //    a) The boundary instructions go into the second call.
  //    b) Anything that consumes the results of the boundary instructions goes
  //    into the second call.
  //    c) Anything that feeds the instructions from (a) and (b) goes into the
  //    first call.
  //    d) The remaining instructions go into the first call.
  // 2. Figure out the outputs of the first call and the inputs to the second
  // call, and how to connect them.
  // 3. Materialized the two new computations and the calls, and put them in the
  // enclosing computation.

  // TODO(mkuper): This splits "down". We also want a version that splits "up",
  // i.e. the boundary ends up in the first call, and the "irrelevant"
  // instructions end up in the second one.
  HloModule* module = body->parent();

  std::vector<HloInstruction*> boundary_instructions =
      GetBoundaryInstructions(body, boundary_predicate);
  if (boundary_instructions.empty()) {
    return std::make_pair(nullptr, nullptr);
  }

  absl::flat_hash_set<HloInstruction*> second_call_instructions =
      GetSecondCallInstructions(body, boundary_instructions);

  absl::flat_hash_set<HloInstruction*> first_call_instructions;
  for (HloInstruction* instruction : body->instructions()) {
    if (!second_call_instructions.contains(instruction)) {
      first_call_instructions.insert(instruction);
    }
  }
  if (first_call_instructions.empty() || second_call_instructions.empty()) {
    return std::make_pair(nullptr, nullptr);
  }

  if (VLOG_IS_ON(1)) {
    VLOG(1) << "First call instructions: ";
    for (HloInstruction* instruction : first_call_instructions) {
      VLOG(1) << instruction->ToString();
    }
    VLOG(1) << "Second call instructions: ";
    for (HloInstruction* instruction : second_call_instructions) {
      VLOG(1) << instruction->ToString();
    }
  }

  // The outputs of the first call are instructions that will be in the first
  // call that are directly used by instructions that will be in the second
  // call. It's convenient to have both a set and a vector representation. We
  // could use a single ordered associative container, but this is simpler.
  absl::flat_hash_set<HloInstruction*> first_call_outputs;
  for (HloInstruction* instruction : second_call_instructions) {
    for (HloInstruction* control_pred : instruction->control_predecessors()) {
      // Don't break the function if it would create a control edge that needs
      // to be threaded between the two new functions.
      if (first_call_instructions.contains(control_pred)) {
        return std::make_pair(nullptr, nullptr);
      }
    }
    for (HloInstruction* data_pred : instruction->operands()) {
      if (first_call_instructions.contains(data_pred)) {
        first_call_outputs.insert(data_pred);
      }
    }
  }

  // Make sure the order of outputs is deterministic.
  std::vector<HloInstruction*> first_call_outputs_vec(
      first_call_outputs.begin(), first_call_outputs.end());
  std::sort(first_call_outputs_vec.begin(), first_call_outputs_vec.end(),
            [](HloInstruction* a, HloInstruction* b) {
              return a->unique_id() < b->unique_id();
            });
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "First call outputs: ";
    for (HloInstruction* instruction : first_call_outputs_vec) {
      VLOG(1) << instruction->ToString();
    }
  }

  // Construct the first call body. We delete everything that goes into the
  // second call from the call body, and construct a new output tuple based on
  // the inputs the second call needs.
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      first_call_replacements;
  for (HloInstruction* instruction : second_call_instructions) {
    first_call_replacements.insert({instruction, nullptr});
  }
  HloComputation* first_call_computation =
      module->AddEmbeddedComputation(body->CloneWithReplacements(
          &first_call_replacements, /*extra_parameters=*/{},
          /*context=*/nullptr, /*suffix=*/"first", first_call_outputs_vec));

  // Now construct the second call body. In the call body, the instructions
  // that were assigned to the first call that are directly used are replaced by
  // parameters, and the rest are deleted.
  absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
      second_call_replacements;
  for (int i = 0; i < first_call_outputs_vec.size(); ++i) {
    second_call_replacements.insert(
        {first_call_outputs_vec[i],
         HloInstruction::CreateParameter(i, first_call_outputs_vec[i]->shape(),
                                         absl::StrCat("first_output_", i))});
  }
  for (HloInstruction* instruction : first_call_instructions) {
    if (first_call_outputs.contains(instruction)) {
      continue;
    }
    second_call_replacements.insert({instruction, nullptr});
  }
  HloComputation* second_call_computation =
      module->AddEmbeddedComputation(body->CloneWithReplacements(
          &second_call_replacements, /*extra_parameters=*/{},
          /*context=*/nullptr, /*suffix=*/"second", /*new_root=*/nullptr));

  return std::make_pair(first_call_computation, second_call_computation);
}

absl::StatusOr<bool> CallSplitter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Find all the call instructions that match the predicate. We don't process
  // them immediately since we're going to change their enclosing computation.
  // process all calls in a computation together. Note that we want to process
  // them in the same order as we encounter them, because for nested calls, we
  // want to process the deeper call first.

  // TODO(mkuper): Support unflattened graphs properly - if a function has
  // several callsites, we should only split it once, and then reuse the
  // resulting computations.
  std::vector<HloInstruction*> calls_to_process;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kCall) {
        continue;
      }
      // TODO(mkuper): Support calls with control dependencies, if that appears
      // useful.
      if (instruction->HasControlDependencies()) {
        continue;
      }
      if (!execution_threads.empty() &&
          !execution_threads.contains(
              instruction->to_apply()->execution_thread())) {
        continue;
      }
      // TODO(mkuper): We could support removing dead parameters from non-tuple
      // shaped calls. We could also potentially support pass-through for
      // tuple-shaped calls where the root *instruction* is not kTuple by doing
      // more complex analysis.
      if (instruction->to_apply()->root_instruction()->opcode() !=
          HloOpcode::kTuple) {
        continue;
      }
      VLOG(1) << "Found matching call: " << instruction->ToString();
      calls_to_process.push_back(instruction);
    }
  }

  bool changed = false;
  split_call_bodies_.clear();
  for (HloInstruction* call : calls_to_process) {
    // We may have already split this callee when wer processed another
    // callsite, in which case we can reuse the results.
    auto get_split = [&](HloComputation* body) {
      auto it = split_call_bodies_.find(body);
      if (it != split_call_bodies_.end()) {
        return it->second;
      }
      auto split = SplitCallBody(body, boundary_predicate_);
      split_call_bodies_[body] = split;
      return split;
    };

    auto split_result = get_split(call->to_apply());
    if (split_result.first != nullptr) {
      changed |= true;
      TF_RETURN_IF_ERROR(
          SplitCallSite(call, split_result.first, split_result.second));
    }
  }

  return changed;
}

}  // namespace xla
