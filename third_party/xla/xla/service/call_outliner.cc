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

#include "xla/service/call_outliner.h"

#include <cstdint>
#include <string>
#include <utility>
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
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Extracts the original computation name from the frontend attributes.
std::string GetMarkedComputationName(const HloInstruction* instruction) {
  if (instruction->has_frontend_attributes()) {
    auto it = instruction->frontend_attributes().map().find(
        kCallMarkedComputationAttribute);
    if (it != instruction->frontend_attributes().map().end()) {
      return it->second;
    }
  }
  return "";
}

// Safely replaces all uses of a marker with its operands, reconstructing
// the tuple if the marker has a tuple shape and has been flattened or
// is otherwise shape-incompatible with its single operand.
absl::Status ReplaceMarkerWithOperands(HloComputation* computation,
                                       HloInstruction* marker) {
  if (marker->shape().IsTuple()) {
    if (marker->operand_count() == 1 &&
        marker->shape() == marker->operand(0)->shape()) {
      return marker->ReplaceAllUsesWith(marker->mutable_operand(0));
    }
    // Reconstruct the tuple.
    std::vector<HloInstruction*> tuple_operands;
    tuple_operands.reserve(marker->operand_count());
    for (int i = 0; i < marker->operand_count(); ++i) {
      tuple_operands.push_back(marker->mutable_operand(i));
    }
    HloInstruction* new_tuple = computation->AddInstruction(
        HloInstruction::CreateTuple(tuple_operands));
    return marker->ReplaceAllUsesWith(new_tuple);
  }

  TF_RET_CHECK(marker->operand_count() == 1)
      << "Non-tuple marker must have exactly one operand, but has "
      << marker->operand_count();
  return marker->ReplaceAllUsesWith(marker->mutable_operand(0));
}

// Restores metadata, sharding, frontend attributes, and control dependencies
// from the before/after markers back to the newly created outlined call
// instruction.
absl::Status RestoreMetadataAndDependencies(HloInstruction* call_instruction,
                                            HloInstruction* innermost_before,
                                            HloInstruction* innermost_after) {
  // Restore instruction sharding from the 'after' marker.
  if (innermost_after->has_sharding()) {
    call_instruction->set_sharding(innermost_after->sharding());
  }

  // Restore HLO metadata from the 'after' marker.
  call_instruction->set_metadata(innermost_after->metadata());

  // Restore backend config from the 'after' marker.
  if (innermost_after->has_backend_config()) {
    call_instruction->set_raw_backend_config_string(
        innermost_after->raw_backend_config_string());
  }

  // Restore original instruction name if present.
  std::string instruction_name;
  if (innermost_after->has_frontend_attributes()) {
    auto map = innermost_after->frontend_attributes().map();
    auto it = map.find(kCallMarkedInstructionNameAttribute.data());
    if (it != map.end()) {
      instruction_name = it->second;
    }
  }

  // Restore original frontend attributes from the 'after' marker (excluding the
  // internal marker attributes).
  FrontendAttributes attributes = innermost_after->frontend_attributes();
  attributes.mutable_map()->erase(kCallMarkedComputationAttribute.data());
  attributes.mutable_map()->erase(kCallMarkedInstructionNameAttribute.data());
  if (!attributes.map().empty()) {
    call_instruction->set_frontend_attributes(attributes);
  }

  if (!instruction_name.empty()) {
    call_instruction->SetAndSanitizeName(instruction_name);
  }

  // Restore control predecessors from the 'before' marker.
  std::vector<HloInstruction*> before_predecessors =
      innermost_before->control_predecessors();
  for (HloInstruction* pred : before_predecessors) {
    RETURN_IF_ERROR(pred->AddControlDependencyTo(call_instruction));
    RETURN_IF_ERROR(pred->RemoveControlDependencyTo(innermost_before));
  }

  // Restore control successors from the 'after' marker.
  std::vector<HloInstruction*> after_successors =
      innermost_after->control_successors();
  for (HloInstruction* succ : after_successors) {
    RETURN_IF_ERROR(call_instruction->AddControlDependencyTo(succ));
    RETURN_IF_ERROR(innermost_after->RemoveControlDependencyTo(succ));
  }

  return absl::OkStatus();
}

}  // namespace

void CallOutliner::InitializeParameters(
    const OutlineBlock& block, HloComputation::Builder& builder,
    std::vector<HloInstruction*>& new_parameters,
    std::vector<HloInstruction*>& old_operands) {
  new_parameters.reserve(block.before->operand_count());
  old_operands.reserve(block.before->operand_count());
  for (int i = 0; i < block.before->operand_count(); ++i) {
    HloInstruction* param =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i, block.before->operand(i)->shape(), absl::StrCat("p", i)));
    new_parameters.push_back(param);
    old_operands.push_back(block.before->mutable_operand(i));
  }
}

HloInstruction* CallOutliner::GetOrCreateMappedOperand(
    HloInstruction* operand, const OutlineBlock& block,
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& new_parameters,
    std::vector<HloInstruction*>& old_operands) {
  auto it = original_to_outlined_map_.find(operand);
  if (it != original_to_outlined_map_.end()) {
    return it->second;
  }
  if (operand->opcode() == HloOpcode::kGetTupleElement &&
      operand->operand(0) == block.before) {
    return new_parameters[operand->tuple_index()];
  }
  if (operand == block.before && !operand->shape().IsTuple()) {
    return new_parameters[0];
  }
  if (operand->opcode() == HloOpcode::kConstant) {
    HloInstruction* cloned = builder.AddInstruction(operand->Clone());
    original_to_outlined_map_[operand] = cloned;
    return cloned;
  }
  // Capture externally defined values implicitly referenced in the block.
  int add_index = old_operands.size();
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          add_index, operand->shape(), absl::StrCat("extra_", add_index)));
  new_parameters.push_back(param);
  old_operands.push_back(operand);
  original_to_outlined_map_[operand] = param;
  return param;
}

void CallOutliner::ProcessInstruction(
    HloInstruction* instruction, const OutlineBlock& block,
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& new_parameters,
    std::vector<HloInstruction*>& old_operands) {
  if (instruction->opcode() == HloOpcode::kGetTupleElement &&
      instruction->operand(0) == block.before) {
    original_to_outlined_map_[instruction] =
        new_parameters[instruction->tuple_index()];
    return;
  }

  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(instruction->operand_count());
  for (int i = 0; i < instruction->operand_count(); ++i) {
    HloInstruction* operand = instruction->mutable_operand(i);
    new_operands.push_back(GetOrCreateMappedOperand(
        operand, block, builder, new_parameters, old_operands));
  }

  HloInstruction* cloned = builder.AddInstruction(
      instruction->CloneWithNewOperands(instruction->shape(), new_operands));
  original_to_outlined_map_[instruction] = cloned;
}

absl::StatusOr<HloComputation*> CallOutliner::BuildOutlinedComputation(
    HloModule* module, const OutlineBlock& block,
    std::vector<HloInstruction*>& old_operands) {
  std::string computation_name = GetMarkedComputationName(block.after);
  if (computation_name.empty()) {
    computation_name = kDefaultOutlinedComputationName;
  }
  HloComputation::Builder builder(computation_name);

  std::vector<HloInstruction*> new_parameters;
  InitializeParameters(block, builder, new_parameters, old_operands);

  original_to_outlined_map_.clear();
  for (HloInstruction* instruction : block.body) {
    ProcessInstruction(instruction, block, builder, new_parameters,
                       old_operands);
  }

  // If the outlined block has multiple outputs, the '_after' marker will have
  // a Tuple shape and multiple operands. We must reconstruct a Tuple inside
  // the outlined computation so that its output shape matches the Call
  // instruction's shape. Otherwise, we just return the mapped single operand.
  if (block.after->operand_count() > 1) {
    std::vector<HloInstruction*> mapped_outputs;
    mapped_outputs.reserve(block.after->operand_count());
    for (int i = 0; i < block.after->operand_count(); ++i) {
      mapped_outputs.push_back(
          GetOrCreateMappedOperand(block.after->mutable_operand(i), block,
                                   builder, new_parameters, old_operands));
    }
    HloInstruction* root_tuple =
        builder.AddInstruction(HloInstruction::CreateTuple(mapped_outputs));
    return module->AddEmbeddedComputation(builder.Build(root_tuple));
  }
  return module->AddEmbeddedComputation(builder.Build(
      GetOrCreateMappedOperand(block.after->mutable_operand(0), block, builder,
                               new_parameters, old_operands)));
}

absl::StatusOr<HloInstruction*> CallOutliner::OutlineAndReplaceBlock(
    HloModule* module, HloComputation* computation, const OutlineBlock& block) {
  HloInstruction* innermost_before = block.before;
  HloInstruction* innermost_after = block.after;

  std::vector<HloInstruction*> call_operands;
  ASSIGN_OR_RETURN(HloComputation * outlined_computation,
                   BuildOutlinedComputation(module, block, call_operands));
  std::string original_computation_name = GetMarkedComputationName(block.after);
  if (!original_computation_name.empty()) {
    outlined_computation->SetAndSanitizeName(original_computation_name);
  }
  outlined_computation->SetExecutionThread(computation->execution_thread());
  VLOG(2) << "CallOutliner created outlined computation "
          << outlined_computation->name() << " in module " << module->name();

  HloInstruction* call_instruction =
      computation->AddInstruction(HloInstruction::CreateCall(
          innermost_after->shape(), call_operands, outlined_computation));

  RETURN_IF_ERROR(RestoreMetadataAndDependencies(
      call_instruction, innermost_before, innermost_after));

  // Replace _after marker uses with the new call result.
  RETURN_IF_ERROR(innermost_after->ReplaceAllUsesWith(call_instruction));

  // Replace _before marker uses.
  if (innermost_before->shape().IsTuple()) {
    // Mapping get-tuple-element directly to original operands.
    std::vector<HloInstruction*> before_uses = innermost_before->users();
    for (HloInstruction* use : before_uses) {
      if (use->opcode() == HloOpcode::kGetTupleElement) {
        int index = use->tuple_index();
        RETURN_IF_ERROR(
            use->ReplaceAllUsesWith(innermost_before->mutable_operand(index)));
        RETURN_IF_ERROR(computation->RemoveInstruction(use));
      }
    }
  }

  if (!innermost_before->IsDead()) {
    RETURN_IF_ERROR(ReplaceMarkerWithOperands(computation, innermost_before));
  }

  // Verify that markers are dead before removal.
  TF_RET_CHECK(innermost_after->IsDead()) << "innermost_after still has users";
  TF_RET_CHECK(innermost_before->IsDead())
      << "innermost_before still has users";

  // Cleanup markers.
  if (innermost_after->parent()) {
    RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(innermost_after));
  }
  if (innermost_before->parent()) {
    RETURN_IF_ERROR(
        computation->RemoveInstructionAndUnusedOperands(innermost_before));
  }

  return call_instruction;
}

bool CallOutliner::IsBeforeMarker(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCustomCall &&
         instruction->custom_call_target() == kCallMarkerBeforeTarget;
}

bool CallOutliner::IsAfterMarker(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kCustomCall &&
         instruction->custom_call_target() == kCallMarkerAfterTarget;
}

void CallOutliner::HandleBeforeMarker(HloInstruction* instruction) {
  std::string name = GetMarkedComputationName(instruction);
  name_to_stack_indices_[name].push_back(stack_.size());
  stack_.push_back(OutlineBlock{instruction, nullptr, {}});
}

absl::StatusOr<bool> CallOutliner::HandleAfterMarker(
    HloModule* module, HloComputation* computation,
    HloInstruction* instruction) {
  std::string target_name = GetMarkedComputationName(instruction);
  auto it = name_to_stack_indices_.find(target_name);
  if (it == name_to_stack_indices_.end() || it->second.empty()) {
    return absl::InternalError(
        absl::StrCat("Found _after marker without matching _before marker for ",
                     target_name));
  }
  std::vector<int>& stack_index_vector = it->second;
  int target_index = stack_index_vector.back();

  if (stack_.size() - 1 > target_index) {
    std::string abandoned_name =
        GetMarkedComputationName(stack_[target_index + 1].before);
    return absl::InternalError(absl::StrCat(
        "Found _after marker for ", target_name,
        " but nested _before marker for ", abandoned_name, " was not closed."));
  }

  OutlineBlock block = std::move(stack_.back());
  stack_.pop_back();

  stack_index_vector.pop_back();
  if (stack_index_vector.empty()) {
    name_to_stack_indices_.erase(it);
  }

  block.after = instruction;

  // Outline the block.
  ASSIGN_OR_RETURN(HloInstruction * call_instruction,
                   OutlineAndReplaceBlock(module, computation, block));

  // The new call instruction is added to the parent block's body if any.
  if (!stack_.empty()) {
    stack_.back().body.push_back(call_instruction);
  }
  return true;
}

void CallOutliner::HandleOtherInstruction(HloInstruction* instruction) {
  // Only accumulate instructions if:
  // 1. The stack is not empty (we are inside an outline block). Instructions
  //    outside of markers should not be outlined.
  // 2. The instruction is not a parameter. Parent parameters cannot be cloned
  //    directly;
  if (!stack_.empty() && instruction->opcode() != HloOpcode::kParameter) {
    stack_.back().body.push_back(instruction);
  }
}

absl::StatusOr<bool> CallOutliner::OutlineComputation(
    HloModule* module, HloComputation* computation,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // We only outline calls in computations that are included in
  // `execution_threads`. Note that if we ever decide to
  // temporarily inline a call on a different execution thread to outline it
  // later, storing the execution thread on the marker won't work.
  if (!HloInstruction::IsThreadIncluded(computation->execution_thread(),
                                        execution_threads)) {
    return false;
  }

  const int64_t before_count = computation->instruction_count();

  stack_.clear();
  name_to_stack_indices_.clear();
  // We must process instructions in post-order (dependency-first). An arbitrary
  // order or preorder traversal would not work because they can visit a user
  // instruction before call is outlined. In our mapping logic, if operand of an
  // instruction inside the block is not yet mapped, it is assumed to be an
  // externally defined value and is captured as a parameter to the outlined
  // computation. Post-order ensures that all internal dependencies are mapped
  // before their users are processed.
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();

  bool mutated = false;
  for (HloInstruction* instruction : instructions) {
    if (IsBeforeMarker(instruction)) {
      HandleBeforeMarker(instruction);
    } else if (IsAfterMarker(instruction)) {
      ASSIGN_OR_RETURN(bool outlined,
                       HandleAfterMarker(module, computation, instruction));
      mutated |= outlined;
    } else {
      HandleOtherInstruction(instruction);
    }
  }

  if (!stack_.empty()) {
    std::string abandoned_name =
        GetMarkedComputationName(stack_.front().before);
    return absl::InternalError(
        absl::StrCat("Found _before marker without matching _after marker for ",
                     abandoned_name));
  }

  if (mutated) {
    const int64_t after_count = computation->instruction_count();
    VLOG(2) << "Outlining stats for computation " << computation->name()
            << " in module " << module->name()
            << ": instructions before = " << before_count
            << ", after = " << after_count
            << ", diff = " << (after_count - before_count);
  }

  return mutated;
}

absl::StatusOr<bool> CallOutliner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  const int64_t module_before_count = module->instruction_count();
  const int64_t module_before_comp_count = module->computation_count();
  bool mutated = false;

  // Copy computations to a vector to avoid iterator invalidation when we add
  // outlined computations to the module.
  std::vector<HloComputation*> computations(module->computations().begin(),
                                            module->computations().end());

  // Order in which we process computations does not matter because outlining
  // is local to each computation and does not affect the structure of other
  // existing computations (iff there are no cycles in the call graph).
  // We iterate over multiple computations because inlined module can
  // contain sub-computations for control flow (such as while loop bodies and
  // conditional branches) or due to partial inlining, any of which may contain
  // call markers.
  for (HloComputation* computation : computations) {
    ASSIGN_OR_RETURN(
        bool computation_mutated,
        OutlineComputation(module, computation, execution_threads));
    mutated |= computation_mutated;
  }

  if (mutated) {
    int64_t module_after_count = module->instruction_count();
    int64_t module_after_comp_count = module->computation_count();
    VLOG(2) << "Outlining stats for module " << module->name()
            << ": instructions before = " << module_before_count
            << ", after = " << module_after_count
            << ", diff = " << (module_after_count - module_before_count)
            << "; computations before = " << module_before_comp_count
            << ", after = " << module_after_comp_count << ", diff = "
            << (module_after_comp_count - module_before_comp_count);
  }

  return mutated;
}

}  // namespace xla
