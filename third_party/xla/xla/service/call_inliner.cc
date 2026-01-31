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

#include "absl/algorithm/container.h"
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
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/utils/hlo_query.h"
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

// Recursively prepends the given prefix to the op name of the given HLO
// instruction as well as all the instructions in its called computations.
void RecursivelyUpdateOpName(HloInstruction* hlo, absl::string_view prefix) {
  if (prefix.empty()) {
    return;
  }

  // We only want to descend into "control flow" computations, since annotating
  // embedded computations is wasted effort.
  //
  // TODO(b/429017389): We don't want to descend into calls, since this will
  // produce incorrect metadata for computations with multiple callsites.
  // However we're still seeing some missing prefix metadata that we'll need to
  // figure out that recursing into calls does appear to help with.
  if (GetInstructionCallContext(hlo->opcode()) == CallContext::kControlFlow &&
      hlo->opcode() != HloOpcode::kCall) {
    for (HloComputation* computation : hlo->called_computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        RecursivelyUpdateOpName(instruction, prefix);
      }
    }
  }

  // We found that some users are sticking many megabytes of strings into
  // op_name. Don't form op names that would be too big.
  OpMetadata metadata = hlo->metadata();
  if (prefix.size() + metadata.op_name().size() < CallInliner::kMaxOpNameSize) {
    if (metadata.op_name().empty()) {
      metadata.set_op_name(prefix);
    } else {
      metadata.set_op_name(absl::StrCat(prefix, "/", metadata.op_name()));
    }
    hlo->set_metadata(metadata);
  }
}

// Traverses the callee computation, inlining cloned nodes into the caller
// computation and connecting them to producers/consumers appropriately.
// When the traversal has completed, the provided call instruction is entirely
// replaced in the caller's graph.
class SubcomputationInsertionVisitor : public DfsHloVisitorWithDefault {
 public:
  // call is the call operation -- it will be replaced with the body of the
  // called computation.
  explicit SubcomputationInsertionVisitor(HloInstruction* call,
                                          absl::string_view call_op_name)
      : call_(call), outer_(call->parent()), call_op_name_(call_op_name) {
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
    RecursivelyUpdateOpName(new_hlo.get(), call_op_name_);
    HloInstruction* new_hlo_pointer =
        outer_->AddInstruction(std::move(new_hlo));
    TF_RETURN_IF_ERROR(NoteMapping(hlo, new_hlo_pointer));

    PropagateOriginalValue(new_hlo_pointer, hlo);

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

  // Does not create new nodes for the parameter; rather, notes the mapping
  // from the subcomputation parameter node to the call operands in the caller
  // computation.
  absl::Status HandleParameter(HloInstruction* parameter) override {
    TF_RETURN_IF_ERROR(NoteMapping(
        parameter, call_->mutable_operand(parameter->parameter_number())));
    return absl::OkStatus();
  }

  // Wires the consumers of the call to instead point at the newly created
  // root, replacing the call operation in the caller computation.
  absl::Status FinishVisit(HloInstruction* root) override {
    TF_ASSIGN_OR_RETURN(HloInstruction * new_root, Resolve(root));
    VLOG(1) << "Replacing all uses of " << call_->ToString()
            << " with new root " << new_root->ToString();
    auto original_value = new_root->original_value();
    // We must relay the control dependencies from this call instruction to
    // the successors too after inlining. The will now depend on the newly
    // inlined root.
    auto result =
        outer_
            ->ReplaceInstruction(
                /*old_instruction=*/call_, /*new_instruction=*/new_root,
                /*preserve_sharding=*/false,
                /*relay_control_dependency=*/true,
                /*remove_unused_operands=*/true)
            .status();
    // Restores the original value of the new root, which gets overwritten
    // when it's used to replace the call instruction.
    new_root->set_original_value(original_value);
    return result;
  }

  CallInliner::InlinedInstructionMap ConsumeInstructionMap() {
    return std::move(subcomputation_hlo_to_new_hlo_);
  }

 private:
  // Resolves the callee subcomputation_hlo to the new (inline) HLO in the
  // caller computation, or returns a NotFound error if that subcomputation
  // HLO has not been mapped.
  absl::StatusOr<HloInstruction*> Resolve(HloInstruction* subcomputation_hlo) {
    auto it = subcomputation_hlo_to_new_hlo_.find(subcomputation_hlo);
    if (it == subcomputation_hlo_to_new_hlo_.end()) {
      return NotFound(
          "Could not find mapping from subcomputation HLO %s to a cloned "
          "HLO.",
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

  // Propagates original value information from the call and the original HLO
  // to the newly cloned HLO.
  void PropagateOriginalValue(HloInstruction* new_hlo_pointer,
                              HloInstruction* hlo) {
    std::shared_ptr<OriginalValue> call_original_value =
        call_->original_value();
    if (!call_original_value) {
      new_hlo_pointer->set_original_value(nullptr);
      return;
    }
    std::optional<std::string> call_instructions =
        call_original_value->GetOriginalCallLikeInstructions();
    if (!call_instructions.has_value()) {
      // If the call instruction is lost, we must drop the original values
      // on the inlined instructions because the call hierarchy is lost.
      new_hlo_pointer->set_original_value(nullptr);
      return;
    }
    new_hlo_pointer->CopyOriginalValue(hlo, /*clone=*/true,
                                       /*issue_warning=*/true);
    if (call_instructions->empty()) {
      // Empty call instructions means the call is synthetic and hence the
      // inlined instruction do not need to be prefixed with the call
      // instructions. Hence we can just return here to have the copied original
      // value to be used.
      return;
    }
    std::shared_ptr<OriginalValue> original_value =
        new_hlo_pointer->original_value();
    if (!original_value) {
      return;
    }
    for (auto& pair : original_value->mutable_original_arrays()) {
      std::optional<OriginalArray>& original_array = pair.second;
      if (original_array.has_value()) {
        original_array->instruction_name = absl::StrCat(
            *call_instructions, "/", original_array->instruction_name);
      }
    }
  }

  HloInstruction* call_;
  HloComputation* outer_;
  CallInliner::InlinedInstructionMap subcomputation_hlo_to_new_hlo_;
  absl::string_view call_op_name_;
};

bool InlineComposites(
    HloInstruction* instruction,
    const absl::flat_hash_set<std::string>& composites_to_preserve) {
  return !instruction->is_composite() ||
         !composites_to_preserve.contains(
             instruction->frontend_attributes().map().at("composite.name"));
}

// Introduces a specific attribute so that the frontend has the direct
// control over inlining specific calls.
bool FrontendAttributesAllowInlining(HloInstruction* instruction) {
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
    for (auto maybe_attribute :
         {call_attributes.map().contains("MUST_FUSE")
              ? std::make_optional("MUST_FUSE")
          : call_attributes.map().contains("MAXIMAL_FUSE")
              ? std::make_optional("MAXIMAL_FUSE")
              : std::nullopt,
          call_attributes.map().contains("mosaic_fusion_group")
              ? std::make_optional("mosaic_fusion_group")
              : std::nullopt}) {
      if (!maybe_attribute.has_value()) {
        continue;
      }
      const auto attribute = *maybe_attribute;
      for (auto instruction : callee->instructions()) {
        // Do so for only fusible instructions.
        if (instruction->IsFusible()) {
          FrontendAttributes frontend_attributes =
              instruction->frontend_attributes();
          frontend_attributes.mutable_map()->insert(
              {attribute, call_attributes.map().at(attribute)});
          instruction->set_frontend_attributes(frontend_attributes);
        }
      }
    }
  }

  // We visit the callee, cloning its body into its caller.
  SubcomputationInsertionVisitor visitor(call, call->metadata().op_name());
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
  if (instruction->GetModule()->config().use_shardy_partitioner() &&
      (absl::StrContains(instruction->to_apply()->name(), "shmap_body") ||
       absl::StrContains(instruction->to_apply()->name(),
                         sdy::kManualComputationFuncName.str()))) {
    // TODO(b/436603025). Remove this special handling by marking the
    // instruction as uninlineable with the frontend attribute.
    //
    // Specific inlining rules when needing to round-trip from MLIR->HLO->MLIR
    // when using Shardy (github.com/openxla/shardy).
    //
    // - shmap_body: We do not want to inline the bodies of JAX shard maps to
    //   import them into an `sdy.ManualComputationOp`. This is for the MHLO
    //   round-trip pipeline
    // - kManualComputationFuncName: Same as shmap_body except for the SDY
    //   round-trip pipeline.
    return false;
  }
  return InlineComposites(instruction, composites_to_preserve_);
}

bool CallInliner::ShouldInline(const CallGraph& call_graph,
                               HloInstruction* instruction) const {
  // Check this is an inlineable call op (but not frontend attributes)
  if (!IsInlineableCallOp(instruction)) {
    return false;
  }

  // Check the override policy, if any.
  InlineOverridePolicy policy = InlineOverridePolicy::kAllowInline;
  if (override_policy_.has_value()) {
    policy = (*override_policy_)(call_graph, instruction);
  }

  // If the policy is to never inline, we're done.
  if (policy == InlineOverridePolicy::kProhibitInline) {
    return false;
  }

  // If the policy is to ignore frontend attributes, do so.
  if (policy != InlineOverridePolicy::kAllowIgnoreFrontendAttributes) {
    if (!FrontendAttributesAllowInlining(instruction)) {
      return false;
    }
  }

  // If we're only inlining calls with a single call site, check that.
  if (single_call_site_) {
    return call_graph.GetNode(instruction->to_apply())
               .caller_callsites()
               .size() == 1;
  }

  return true;
}

absl::StatusOr<bool> CallInliner::InlineAndLegalize(
    const CallGraph& call_graph, HloComputation* computation,
    absl::Span<HloInstruction* const> instruction_sequence,
    std::optional<InlinedInstructionMap*> inline_map) {
  HloModule* module = computation->parent();
  bool did_node_mutate = false;
  std::vector<HloInstruction*> inlined_instructions;
  for (HloInstruction* instruction : instruction_sequence) {
    // Don't inline async called computation since currently it's only
    // used for parallel device computation.
    // TODO(b/229887502): update the inliner to ignore only parallel
    // device type async call instead of all.
    if (ShouldInline(call_graph, instruction)) {
      // The caller instruction will get removed after inlining. Record the
      // callee computation beforehand, so we can find its schedule.
      HloComputation* callee = instruction->to_apply();
      TF_ASSIGN_OR_RETURN(
          CallInliner::InlinedInstructionMap inline_map_cur_call,
          Inline(instruction));
      if (module->has_schedule()) {
        for (HloInstruction* inlined_instruction :
             module->schedule().sequence(callee).instructions()) {
          // Parameters were already added to sequence as operands to the
          // call.
          if (inlined_instruction->opcode() != HloOpcode::kParameter) {
            inlined_instructions.push_back(
                inline_map_cur_call[inlined_instruction]);
          }
        }
      }
      if (update_domain_) {
        HloDomainIsolator isolator([]() { return ShardingDomainCreator{}; });
        for (const auto& [call_inst, inlined_inst] : inline_map_cur_call) {
          TF_RETURN_IF_ERROR(isolator.UpdateDomains(inlined_inst).status());
        }
      }
      if (inline_map.has_value()) {
        inline_map.value()->insert(inline_map_cur_call.begin(),
                                   inline_map_cur_call.end());
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
    for (HloInstruction* instruction : computation->instructions()) {
      if (!HloChannelInstruction::ClassOf(instruction)) {
        continue;
      }
      // Channel IDs for host transfers are part of the ABI, and can never be
      // uniquified.
      HloSendRecvInstruction* send_recv =
          DynCast<HloSendRecvInstruction>(instruction);
      if (send_recv && send_recv->is_host_transfer()) {
        continue;
      }
      instruction->set_channel_id(next_unique_channel_id_++);
    }
  }
  return did_node_mutate;
}

absl::StatusOr<bool> CallInliner::RunWithInlineMap(
    HloModule* module, std::optional<InlinedInstructionMap*> inline_map,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  if (uniquify_channel_ids_) {
    next_unique_channel_id_ = hlo_query::NextChannelId(*module);
  }

  // Because call graph nodes are visited in post-order (callees before callers)
  // we'll always inline kCalls into their callers in the appropriate order.
  TF_ASSIGN_OR_RETURN(
      bool did_mutate,
      call_graph->VisitNodesWithReturn(
          [&](const CallGraphNode& node) -> absl::StatusOr<bool> {
            if (!HloInstruction::IsThreadIncluded(
                    node.computation()->execution_thread(),
                    execution_threads)) {
              return false;
            };
            if (module->has_schedule()) {
              HloInstructionSequence& sequence =
                  module->schedule().GetOrCreateSequence(node.computation());
              return InlineAndLegalize(*call_graph, node.computation(),
                                       sequence.instructions(), inline_map);
            }

            return InlineAndLegalize(
                *call_graph, node.computation(),
                node.computation()->MakeInstructionPostOrder(), inline_map);
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

absl::StatusOr<bool> CallInliner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return RunWithInlineMap(module, std::nullopt, execution_threads);
}

bool IsInlineableComputation(HloComputation* computation) {
  auto is_inlineable_call_op = [](HloInstruction* instruction) {
    bool prerequisite = instruction->opcode() == HloOpcode::kCall &&
                        !instruction->has_backend_config() &&
                        !instruction->parent()->IsAsyncComputation();
    if (!prerequisite || (!FrontendAttributesAllowInlining(instruction))) {
      return false;
    }
    return true;
  };
  return absl::c_any_of(computation->instructions(), is_inlineable_call_op);
}

const HloInstruction* InlinedModule::get_inlined_inst(
    const HloInstruction* inst) {
  auto it = clone_context->cloned_instructions().find(inst);
  if (it != clone_context->cloned_instructions().end()) {
    auto it2 = clone_inlined_map.find(it->second);
    if (it2 != clone_inlined_map.end()) {
      return it2->second;
    }
    return it->second;
  }
  return nullptr;
}

absl::StatusOr<InlinedModule> GetInlinedModule(const HloModule* module) {
  auto [cloned_module, clone_context] =
      module->CloneWithContext("inline", module->config());
  CallInliner::InlinedInstructionMap clone_inlined_map;
  CallInliner inliner;
  TF_RETURN_IF_ERROR(
      inliner.RunWithInlineMap(cloned_module.get(), &clone_inlined_map, {})
          .status());
  return InlinedModule{std::move(cloned_module), std::move(clone_context),
                       std::move(clone_inlined_map)};
}

}  // namespace xla
