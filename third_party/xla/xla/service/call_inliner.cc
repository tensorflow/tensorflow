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

#include "xla/service/call_inliner.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_domain_isolator.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Traverses the callee computation, inlining cloned nodes into the caller
// computation and connecting them to producers/consumers appropriately.
// When the traversal has completed, the provided call instruction is entirely
// replaced in the caller's graph.
class SubcomputationInsertionVisitor : public DfsHloVisitorWithDefault {
 public:
  // call is the call operation -- it will be replaced with the body of the
  // called computation.
  explicit SubcomputationInsertionVisitor(HloInstruction* call)
      : call_(call), outer_(call->parent()) {
    CHECK_EQ(HloOpcode::kCall, call_->opcode());
  }

  // Resolves the operands to the HLO instruction in the inlined (caller) graph,
  // and clones the HLO instruction into that graph with the new operands.
  absl::Status DefaultAction(HloInstruction* hlo) override {
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : hlo->operands()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand, Resolve(operand));
      new_operands.push_back(new_operand);
    }
    VLOG(1) << "Cloning HLO and adding to caller: " << hlo->ToString();
    auto new_hlo = hlo->CloneWithNewOperands(hlo->shape(), new_operands);
    HloInstruction* new_hlo_pointer =
        outer_->AddInstruction(std::move(new_hlo));
    TF_RETURN_IF_ERROR(NoteMapping(hlo, new_hlo_pointer));

    new_hlo_pointer->CopyOriginalValue(hlo, /*clone=*/true);
    if (std::shared_ptr<OriginalValue> original_value =
            new_hlo_pointer->original_value()) {
      for (auto& leaf : original_value->leaves()) {
        std::optional<OriginalArray>& original_array = leaf.second;
        if (original_array.has_value()) {
          std::string call_instruction_name;
          if (std::shared_ptr<OriginalValue> call_original_value =
                  call_->original_value()) {
            call_instruction_name =
                call_original_value->leaf_begin()->second->instruction_name;
          }
          absl::StrAppend(&original_array->instruction_name, "/",
                          call_instruction_name);
        }
      }
    }

    // Account for control edges.
    for (HloInstruction* control_predecessor : hlo->control_predecessors()) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_control_predecessor,
                          Resolve(control_predecessor));
      TF_RETURN_IF_ERROR(
          new_control_predecessor->AddControlDependencyTo(new_hlo_pointer));
    }

    // The newly inlined instructions should honor the control predecessors of
    // the previous call instruction.
    for (HloInstruction* control_predecessor : call_->control_predecessors()) {
      TF_RETURN_IF_ERROR(control_predecessor->AddControlDependencyTo(
          /*instruction=*/new_hlo_pointer));
    }

    return absl::OkStatus();
  }

  // Does not create new nodes for the parameter; rather, notes the mapping from
  // the subcomputation parameter node to the call operands in the caller
  // computation.
  absl::Status HandleParameter(HloInstruction* parameter) override {
    TF_RETURN_IF_ERROR(NoteMapping(
        parameter, call_->mutable_operand(parameter->parameter_number())));
    return absl::OkStatus();
  }

  // Wires the consumers of the call to instead point at the newly created root,
  // replacing the call operation in the caller computation.
  absl::Status FinishVisit(HloInstruction* root) override {
    TF_ASSIGN_OR_RETURN(HloInstruction * new_root, Resolve(root));
    VLOG(1) << "Replacing all uses of " << call_->ToString()
            << " with new root " << new_root->ToString();
    auto original_value = new_root->original_value();
    // We must relay the control dependencies from this call instruction to the
    // successors too after inlining. The will now depend on the newly inlined
    // root.
    auto result =
        outer_
            ->ReplaceInstruction(
                /*old_instruction=*/call_, /*new_instruction=*/new_root,
                /*preserve_sharding=*/false, /*relay_control_dependency=*/true,
                /*remove_unused_operands=*/true)
            .status();
    // Restores the original value of the new root, which gets overwritten when
    // it's used to replace the call instruction.
    new_root->set_original_value(original_value);
    return result;
  }

  CallInliner::InlinedInstructionMap ConsumeInstructionMap() {
    return std::move(subcomputation_hlo_to_new_hlo_);
  }

 private:
  // Resolves the callee subcomputation_hlo to the new (inline) HLO in the
  // caller computation, or returns a NotFound error if that subcomputation HLO
  // has not been mapped.
  absl::StatusOr<HloInstruction*> Resolve(HloInstruction* subcomputation_hlo) {
    auto it = subcomputation_hlo_to_new_hlo_.find(subcomputation_hlo);
    if (it == subcomputation_hlo_to_new_hlo_.end()) {
      return NotFound(
          "Could not find mapping from subcomputation HLO %s to a cloned HLO.",
          subcomputation_hlo->ToString());
    }
    return it->second;
  }

  // Notes that the given subcomputation_hlo in the callee has been mapped to
  // the (inline) new_hlo in the caller computation.
  //
  // Returns an error status if the subcomputation_hlo is mapped more than
  // once.
  absl::Status NoteMapping(HloInstruction* subcomputation_hlo,
                           HloInstruction* new_hlo) {
    auto result = subcomputation_hlo_to_new_hlo_.insert(
        std::make_pair(subcomputation_hlo, new_hlo));
    TF_RET_CHECK(result.second)
        << "A mapping for the subcomputation HLO is already present.";
    return absl::OkStatus();
  }

  HloInstruction* call_;
  HloComputation* outer_;
  CallInliner::InlinedInstructionMap subcomputation_hlo_to_new_hlo_;
};

// Specific inlining rules when needing to round-trip from MLIR->HLO->MLIR when
// using Shardy (github.com/openxla/shardy).
//
// - shmap_body: We don't want to inline the bodies of JAX shard maps in order
//   to import them into an `sdy.ManualComputationOp`. This is for the MHLO
//   round-trip pipeline
// - kManualComputationBodyFuncName: Same as shmap_body except for the SDY
//   round-trip pipeline.
bool InlineUnderShardy(HloInstruction* instruction) {
  return !(instruction->GetModule()->config().use_shardy_partitioner() &&
           (absl::StrContains(instruction->to_apply()->name(), "shmap_body") ||
            absl::StrContains(instruction->to_apply()->name(),
                              sdy::kManualComputationBodyFuncName.str())));
}

bool InlineComposites(
    HloInstruction* instruction,
    const absl::flat_hash_set<std::string>& composites_to_preserve) {
  return !instruction->is_composite() ||
         !composites_to_preserve.contains(
             instruction->frontend_attributes().map().at("composite.name"));
}

// Introduces a specific attribute so that the frontend has the direct
// control over inlining specific calls.
bool InlineInstruction(HloInstruction* instruction) {
  auto it = instruction->frontend_attributes().map().find("inlineable");
  if (it != instruction->frontend_attributes().map().end()) {
    return it->second == "true";
  }
  return true;
}

}  // namespace

/* static */ absl::StatusOr<CallInliner::InlinedInstructionMap>
CallInliner::Inline(HloInstruction* call) {
  TF_RET_CHECK(call->opcode() == HloOpcode::kCall)
      << "Instruction was not a call op: " << call->opcode();
  if (call->is_composite()) {
    // Remove composite FE attrs before inlining, else they will appear on the
    // inlined instructions.
    FrontendAttributes frontend_attributes = call->frontend_attributes();
    frontend_attributes.mutable_map()->erase("composite.name");
    frontend_attributes.mutable_map()->erase("composite.attributes");
    frontend_attributes.mutable_map()->erase("composite.version");
    call->set_frontend_attributes(frontend_attributes);
  }

  const auto& callees = call->called_computations();
  TF_RET_CHECK(callees.size() == 1);
  HloComputation* callee = callees[0];

  // Propagate the frontend attributes related to fusion from the call to the
  // inlined instructions.
  if (call->has_frontend_attributes()) {
    const FrontendAttributes& call_attributes = call->frontend_attributes();
    std::string has_fuse =
        call_attributes.map().contains("MUST_FUSE")      ? "MUST_FUSE"
        : call_attributes.map().contains("MAXIMAL_FUSE") ? "MAXIMAL_FUSE"
                                                         : "";
    if (!has_fuse.empty()) {
      for (auto instruction : callee->instructions()) {
        // Do so for only fusible instructions.
        if (instruction->IsFusible()) {
          FrontendAttributes frontend_attributes =
              instruction->frontend_attributes();
          frontend_attributes.mutable_map()->insert(
              {has_fuse, call_attributes.map().at(has_fuse)});
          instruction->set_frontend_attributes(frontend_attributes);
        }
      }
    }
  }

  // We visit the callee, cloning its body into its caller.
  SubcomputationInsertionVisitor visitor(call);
  TF_RETURN_IF_ERROR(callee->Accept(&visitor));
  return visitor.ConsumeInstructionMap();
}

bool CallInliner::IsInlineableCallOp(HloInstruction* instruction) const {
  bool prerequisite = instruction->opcode() == HloOpcode::kCall &&
                      !instruction->has_backend_config() &&
                      !instruction->parent()->IsAsyncComputation();
  if (!prerequisite) {
    return false;
  }
  if (!InlineInstruction(instruction)) {
    // Always prioritize user's explicit requests after fulfilling the
    // prerequisites.
    return false;
  }
  return InlineUnderShardy(instruction) &&
         InlineComposites(instruction, composites_to_preserve_);
}

absl::StatusOr<bool> CallInliner::InlineAndLegalize(
    const CallGraph& call_graph, HloComputation* computation,
    absl::Span<HloInstruction* const> instruction_sequence) const {
  HloModule* module = computation->parent();
  bool did_node_mutate = false;
  std::vector<HloInstruction*> inlined_instructions;
  for (HloInstruction* instruction : instruction_sequence) {
    // Don't inline async called computation since currently it's only
    // used for parallel device computation.
    // TODO(b/229887502): update the inliner to ignore only parallel
    // device type async call instead of all.
    if (IsInlineableCallOp(instruction) &&
        (!single_call_site_ || call_graph.GetNode(instruction->to_apply())
                                       .caller_callsites()
                                       .size() == 1)) {
      // The caller instruction will get removed after inlining. Record the
      // callee computation beforehand, so we can find its schedule.
      HloComputation* callee = instruction->to_apply();
      TF_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap inline_map,
                          Inline(instruction));
      if (module->has_schedule()) {
        for (HloInstruction* inlined_instruction :
             module->schedule().sequence(callee).instructions()) {
          // Parameters were already added to sequence as operands to the
          // call.
          if (inlined_instruction->opcode() != HloOpcode::kParameter) {
            inlined_instructions.push_back(inline_map[inlined_instruction]);
          }
        }
      }
      if (update_domain_) {
        HloDomainIsolator isolator([]() { return ShardingDomainCreator{}; });
        for (const auto& [call_inst, inlined_inst] : inline_map) {
          TF_RETURN_IF_ERROR(isolator.UpdateDomains(inlined_inst).status());
        }
      }
      did_node_mutate = true;
    } else if (module->has_schedule()) {
      inlined_instructions.push_back(instruction);
    }
  }
  if (did_node_mutate && module->has_schedule()) {
    module->schedule().GetOrCreateSequence(computation) =
        HloInstructionSequence(inlined_instructions);
  }
  if (did_node_mutate && uniquify_channel_ids_) {
    int unique_channel_id = 1;
    for (HloInstruction* instruction : computation->instructions()) {
      if (dynamic_cast<HloChannelInstruction*>(instruction)) {
        instruction->set_channel_id(unique_channel_id++);
      }
    }
  }
  return did_node_mutate;
}

absl::StatusOr<bool> CallInliner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  // Because call graph nodes are visited in post-order (callees before callers)
  // we'll always inline kCalls into their callers in the appropriate order.
  TF_ASSIGN_OR_RETURN(
      bool did_mutate,
      call_graph->VisitNodesWithReturn([&](const CallGraphNode& node)
                                           -> absl::StatusOr<bool> {
        if (!HloInstruction::IsThreadIncluded(
                node.computation()->execution_thread(), execution_threads)) {
          return false;
        };
        if (module->has_schedule()) {
          HloInstructionSequence& sequence =
              module->schedule().GetOrCreateSequence(node.computation());
          return InlineAndLegalize(*call_graph, node.computation(),
                                   sequence.instructions());
        }

        return InlineAndLegalize(
            *call_graph, node.computation(),
            node.computation()->MakeInstructionPostOrder());
      }));
  if (did_mutate) {
    // Run DCE to remove called computations which are now becoming unused.
    // This can result then in problems if within the called computation, there
    // were send/recv instructions, which the module group verifier will flag as
    // error finding the same channel ID used for multiple send/recv
    // instructions.
    TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());
    if (module->has_schedule()) {
      TF_RETURN_IF_ERROR(module->schedule().Update(execution_threads));
    }
  }
  return did_mutate;
}

}  // namespace xla
