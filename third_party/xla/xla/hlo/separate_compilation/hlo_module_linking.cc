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

#include "xla/hlo/separate_compilation/hlo_module_linking.h"

#include <memory>
#include <stack>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/separate_compilation/hlo_linking_manifest.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/compilation_environments.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::separate_compilation {

namespace {
struct TraversalState {
  // The principal computation to link
  const HloComputation* absl_nonnull principal;
  // The original stub this principal replaces
  const HloComputation* absl_nonnull stub;
  // True if we started linking `computation`; callees pushed.
  bool entered = false;
};

// Helper class to encapsulate the state and logic of linking HLO computations.
class HloLinker {
 public:
  HloLinker(HloModule* module, const HloLinkingManifest& linking_manifest,
            const HloComputation* root_computation)
      : module_(module),
        linking_manifest_(linking_manifest),
        root_computation_(root_computation),
        clone_context_(module) {}

  absl::StatusOr<HloComputation*> Link() {
    // Slightly abusing `TraversalState` here to avoid adding more boilerplate.
    // Stub field should hold a stub, but here stub == root_computation.
    stack_.push({root_computation_, root_computation_, false});

    HloComputation* result = nullptr;

    while (!stack_.empty()) {
      TraversalState& current = stack_.top();

      if (SkipIfFinished(current)) {
        VLOG(6) << "Already linked: " << current.stub->name();
        stack_.pop();

      } else if (!current.entered) {
        VLOG(6) << "First visit to link: " << current.principal->name();
        TF_RETURN_IF_ERROR(HandleFirstVisit(current));

      } else {
        VLOG(6) << "Second visit to link: " << current.principal->name();
        HloComputation* linked_computation = HandleSecondVisit(current);
        if (current.principal == root_computation_) {
          result = linked_computation;
        }
        stack_.pop();
      }
    }

    CHECK(result != nullptr)
        << "Failed to produce a linked version of the root computation: "
        << root_computation_->name();
    return result;
  }

 private:
  // If the principal computation corresponding to state.stub has already been
  // linked, map state.stub to the existing linked computation and pops state
  // from stack_. Returns true if we popped, false otherwise.
  bool SkipIfFinished(const TraversalState& state) {
    if (auto it = finished_principals_.find(state.principal);
        it != finished_principals_.end()) {
      clone_context_.MapComputation(state.stub, it->second);
      return true;
    }
    return false;
  }

  // First time visiting `state.principal`: check for cycles, mark as entered,
  // and push dependencies to stack_.
  absl::Status HandleFirstVisit(TraversalState& state) {
    if (!being_linked_.insert(state.principal).second) {
      // Computation is currently being linked, which indicates we entered
      // but have not finished all children and exited, i.e. we came back
      // to it through its descendants forming a cycle.
      return absl::InternalError(absl::StrCat(
          "Discovered a cycle during linking, involving computation: ",
          state.principal->name()));
    }
    state.entered = true;

    PushDependencies(state);
    return absl::OkStatus();
  }

  // Pushes dependencies of `state.principal` onto stack_ if they are stubs
  // that need to be linked.
  void PushDependencies(const TraversalState& state) {
    std::vector<HloInstruction*> post_order_instrs =
        state.principal->MakeInstructionPostOrder();
    VLOG(6) << "Processing callees:";
    for (HloInstruction* caller : post_order_instrs) {
      if (caller->opcode() != HloOpcode::kCall) {
        continue;
      }
      HloComputation* callee = caller->to_apply();

      // If callee is a stub it will be in the linking manifest.
      // Non-stub callees within the same source module are handled
      // automatically by CloneInContext.
      if (auto it = linking_manifest_.stub_links.find(callee);
          it != linking_manifest_.stub_links.end()) {
        // Only push if its principal hasn't been mapped yet.
        if (clone_context_.FindComputation(callee) == nullptr) {
          const HloComputation* principal = it->second;
          stack_.push({principal, callee, false});
        }
      }
    }
  }

  // Second time visiting `state.principal`: all dependencies are linked.
  // Clone `state.principal` into module_, update context, and pop from stack_.
  HloComputation* HandleSecondVisit(TraversalState& state) {
    // We are missing const overload for DeepCloneComputations with non-null
    // context.
    HloComputation* mutable_principal = const_cast<HloComputation*>(  // NOLINT
        state.principal);  // This method is effectively const.

    HloComputation* linked_computation =
        module_->DeepCloneComputation(mutable_principal, &clone_context_);
    // TODO: b/429370488 - Add original names to `HloLinkingManifest`.
    linked_computation->SetAndSanitizeName(
        absl::StrCat("linked.", linked_count_++));
    VLOG(6) << "Processed: " << linked_computation->ToString() << " unique_id("
            << linked_computation->unique_id() << ")";

    clone_context_.MapComputation(state.stub, linked_computation);
    finished_principals_.insert({state.principal, linked_computation});

    being_linked_.erase(state.principal);
    return linked_computation;
  }

  HloModule* module_;
  const HloLinkingManifest& linking_manifest_;
  const HloComputation* root_computation_;

  // Cycle detection.
  absl::flat_hash_set<const HloComputation*> being_linked_;
  absl::flat_hash_map<const HloComputation*, HloComputation*>
      finished_principals_;
  std::stack<TraversalState, std::vector<TraversalState>> stack_;

  HloCloneContext clone_context_;
  int linked_count_ = 0;
};
}  // namespace

absl::StatusOr<std::unique_ptr<HloModule>> LinkComputation(
    const HloLinkingManifest& linking_manifest,
    const HloComputation* absl_nonnull root_computation) {
  VLOG(6) << "Root computation: " << root_computation->name();
  auto linked_module = std::make_unique<HloModule>(
      "linked_module", linking_manifest.module_config,
      std::make_unique<CompilationEnvironments>(
          *linking_manifest.compilation_environment));

  HloLinker linker(linked_module.get(), linking_manifest, root_computation);
  TF_ASSIGN_OR_RETURN(HloComputation * linked_clone_ptr, linker.Link());

  linked_module->ReplaceEntryComputation(linked_clone_ptr);
  linked_module->mutable_config().SetComputationLayoutIfExists(
      linked_clone_ptr->ComputeProgramShape());
  xla::HloDCE dce_pass;
  TF_RETURN_IF_ERROR(dce_pass.Run(linked_module.get()).status());

  if (VLOG_IS_ON(6)) {
    for (const HloComputation* comp : linked_module->computations()) {
      LOG(INFO) << comp->name() << " [" << comp->unique_id() << "]";
      for (const HloInstruction* instr : comp->instructions()) {
        LOG(INFO) << "  " << instr->ToString() << " [" << instr->unique_id()
                  << "]";
      }
    }
    LOG(INFO) << linked_module->ToString(HloPrintOptions().set_print_ids(true));
  }
  return linked_module;
}

}  // namespace xla::separate_compilation
