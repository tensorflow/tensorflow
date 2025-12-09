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

#include "xla/service/call_outliner.h"

#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_marker.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace {

// Extracts the original computation name from the frontend attributes.
std::string GetMarkedComputationName(const HloInstruction* inst) {
  if (inst->has_frontend_attributes()) {
    auto it =
        inst->frontend_attributes().map().find(kCallMarkedComputationAttribute);
    if (it != inst->frontend_attributes().map().end()) {
      return it->second;
    }
  }
  return "";
}

// Safely replaces all uses of a marker with its operands, reconstructing
// the tuple if the marker has a tuple shape and has been flattened or
// is otherwise shape-incompatible with its single operand.
absl::Status ReplaceMarkerWithOperands(HloComputation* comp,
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
    HloInstruction* new_tuple =
        comp->AddInstruction(HloInstruction::CreateTuple(tuple_operands));
    return marker->ReplaceAllUsesWith(new_tuple);
  }

  TF_RET_CHECK(marker->operand_count() == 1)
      << "Non-tuple marker must have exactly one operand, but has "
      << marker->operand_count();
  return marker->ReplaceAllUsesWith(marker->mutable_operand(0));
}

}  // namespace

void CallOutliner::InitializeParameters(
    const OutlineBlock& block, HloComputation::Builder& builder,
    std::vector<HloInstruction*>& new_params,
    std::vector<HloInstruction*>& old_operands) {
  for (int i = 0; i < block.before->operand_count(); ++i) {
    HloInstruction* param =
        builder.AddInstruction(HloInstruction::CreateParameter(
            i, block.before->operand(i)->shape(), absl::StrCat("p", i)));
    new_params.push_back(param);
    old_operands.push_back(block.before->mutable_operand(i));
  }
}

HloInstruction* CallOutliner::GetOrCreateMappedOperand(
    HloInstruction* op, const OutlineBlock& block,
    HloComputation::Builder& builder, std::vector<HloInstruction*>& new_params,
    std::vector<HloInstruction*>& old_operands) {
  if (original_to_outlined_map_.contains(op)) {
    return original_to_outlined_map_[op];
  }
  if (op->opcode() == HloOpcode::kGetTupleElement &&
      op->operand(0) == block.before) {
    return new_params[op->tuple_index()];
  }
  // Capture externally defined values implicitly referenced in the block.
  int add_index = old_operands.size();
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          add_index, op->shape(), absl::StrCat("extra_", add_index)));
  new_params.push_back(param);
  old_operands.push_back(op);
  original_to_outlined_map_[op] = param;
  return param;
}

void CallOutliner::ProcessInstruction(
    HloInstruction* inst, const OutlineBlock& block,
    HloComputation::Builder& builder, std::vector<HloInstruction*>& new_params,
    std::vector<HloInstruction*>& old_operands) {
  if (inst->opcode() == HloOpcode::kGetTupleElement &&
      inst->operand(0) == block.before) {
    original_to_outlined_map_[inst] = new_params[inst->tuple_index()];
    return;
  }

  std::vector<HloInstruction*> new_operands;
  for (int i = 0; i < inst->operand_count(); ++i) {
    HloInstruction* op = inst->mutable_operand(i);
    new_operands.push_back(
        GetOrCreateMappedOperand(op, block, builder, new_params, old_operands));
  }

  HloInstruction* cloned = builder.AddInstruction(
      inst->CloneWithNewOperands(inst->shape(), new_operands));
  original_to_outlined_map_[inst] = cloned;
}

absl::StatusOr<HloComputation*> CallOutliner::BuildOutlinedComputation(
    HloModule* module, const OutlineBlock& block,
    std::vector<HloInstruction*>& old_operands) {
  std::string comp_name = GetMarkedComputationName(block.after);
  if (comp_name.empty()) {
    comp_name = "outlined_computation";
  }
  HloComputation::Builder builder(comp_name);

  std::vector<HloInstruction*> new_params;
  InitializeParameters(block, builder, new_params, old_operands);

  original_to_outlined_map_.clear();
  for (HloInstruction* inst : block.body) {
    ProcessInstruction(inst, block, builder, new_params, old_operands);
  }

  return module->AddEmbeddedComputation(builder.Build(
      original_to_outlined_map_[block.after->mutable_operand(0)]));
}

absl::StatusOr<HloInstruction*> CallOutliner::OutlineAndReplaceBlock(
    HloModule* module, HloComputation* comp, const OutlineBlock& block) {
  HloInstruction* innermost_before = block.before;
  HloInstruction* innermost_after = block.after;

  std::vector<HloInstruction*> call_operands;
  ASSIGN_OR_RETURN(HloComputation * outlined_comp,
                   BuildOutlinedComputation(module, block, call_operands));
  LOG(INFO) << "CallOutliner created outlined computation "
            << outlined_comp->name() << " in module " << module->name();

  HloInstruction* call_inst = comp->AddInstruction(HloInstruction::CreateCall(
      innermost_after->shape(), call_operands, outlined_comp));

  // Replace _after marker uses with the new call result.
  RETURN_IF_ERROR(innermost_after->ReplaceAllUsesWith(call_inst));

  // Replace _before marker uses.
  if (innermost_before->shape().IsTuple()) {
    // Mapping get-tuple-element directly to original operands.
    std::vector<HloInstruction*> before_uses = innermost_before->users();
    for (HloInstruction* use : before_uses) {
      if (use->opcode() == HloOpcode::kGetTupleElement) {
        int idx = use->tuple_index();
        RETURN_IF_ERROR(
            use->ReplaceAllUsesWith(innermost_before->mutable_operand(idx)));
        RETURN_IF_ERROR(comp->RemoveInstruction(use));
      }
    }
  }

  if (!innermost_before->IsDead()) {
    RETURN_IF_ERROR(ReplaceMarkerWithOperands(comp, innermost_before));
  }

  if (!innermost_before->IsDead()) {
    std::cerr << "innermost_before still has users:\n";
    for (HloInstruction* user : innermost_before->users()) {
      std::cerr << "  user: " << user->ToString() << "\n";
    }
    std::cerr << std::flush;
  }

  // Verify that markers are dead before removal.
  TF_RET_CHECK(innermost_after->IsDead()) << "innermost_after still has users";
  TF_RET_CHECK(innermost_before->IsDead())
      << "innermost_before still has users";

  // Cleanup markers.
  if (innermost_after->parent()) {
    RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(innermost_after));
  }
  if (innermost_before->parent()) {
    RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(innermost_before));
  }

  return call_inst;
}

void CallOutliner::PopAndMergeAbandonedBlocks(int target_idx) {
  while (stack_.size() - 1 > target_idx) {
    OutlineBlock abandoned = std::move(stack_.back());
    stack_.pop_back();

    std::string abandoned_name = GetMarkedComputationName(abandoned.before);
    name_to_stack_indices_[abandoned_name].pop_back();
    if (name_to_stack_indices_[abandoned_name].empty()) {
      name_to_stack_indices_.erase(abandoned_name);
    }

    // Merge abandoned block into the block below it.
    stack_.back().body.push_back(abandoned.before);
    stack_.back().body.insert(stack_.back().body.end(), abandoned.body.begin(),
                              abandoned.body.end());
  }
}

bool CallOutliner::IsBeforeMarker(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kCustomCall &&
         inst->custom_call_target() == kCallMarkerBeforeTarget;
}

bool CallOutliner::IsAfterMarker(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kCustomCall &&
         inst->custom_call_target() == kCallMarkerAfterTarget;
}

void CallOutliner::HandleBeforeMarker(HloInstruction* inst) {
  std::string name = GetMarkedComputationName(inst);
  name_to_stack_indices_[name].push_back(stack_.size());
  stack_.push_back(OutlineBlock{inst, nullptr, {}});
}

absl::StatusOr<bool> CallOutliner::HandleAfterMarker(HloModule* module,
                                                     HloComputation* comp,
                                                     HloInstruction* inst) {
  std::string target_name = GetMarkedComputationName(inst);
  auto it = name_to_stack_indices_.find(target_name);
  if (it == name_to_stack_indices_.end() || it->second.empty()) {
    LOG(WARNING) << "Found _after marker without matching _before marker for "
                 << target_name << ". Removing _after marker.";
    RETURN_IF_ERROR(ReplaceMarkerWithOperands(comp, inst));
    RETURN_IF_ERROR(comp->RemoveInstruction(inst));
    return true;
  }
  int target_idx = it->second.back();

  PopAndMergeAbandonedBlocks(target_idx);

  OutlineBlock block = std::move(stack_.back());
  stack_.pop_back();

  it->second.pop_back();
  if (it->second.empty()) {
    name_to_stack_indices_.erase(it);
  }

  block.after = inst;

  // Outline the block.
  ASSIGN_OR_RETURN(HloInstruction * call_inst,
                   OutlineAndReplaceBlock(module, comp, block));

  // The new call instruction is added to the parent block's body if any.
  if (!stack_.empty()) {
    stack_.back().body.push_back(call_inst);
  }
  return true;
}

void CallOutliner::HandleOtherInstruction(HloInstruction* inst) {
  if (!stack_.empty() && inst->opcode() != HloOpcode::kParameter) {
    stack_.back().body.push_back(inst);
  }
}

absl::StatusOr<bool> CallOutliner::OutlineComputation(
    HloModule* module, HloComputation* comp,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!HloInstruction::IsThreadIncluded(comp->execution_thread(),
                                        execution_threads)) {
    return false;
  }

  const int64_t before_count = comp->instruction_count();

  stack_.clear();
  name_to_stack_indices_.clear();
  std::vector<HloInstruction*> instructions = comp->MakeInstructionPostOrder();

  bool mutated = false;
  for (HloInstruction* inst : instructions) {
    if (IsBeforeMarker(inst)) {
      HandleBeforeMarker(inst);
    } else if (IsAfterMarker(inst)) {
      ASSIGN_OR_RETURN(bool outlined, HandleAfterMarker(module, comp, inst));
      mutated |= outlined;
    } else {
      HandleOtherInstruction(inst);
    }
  }

  // Clean up any orphaned or mismatched 'before' markers left on the stack.
  // This handles cases where a 'before' marker was registered but no matching
  // 'after' marker was found in the computation (e.g., due to malformed HLO,
  // previous optimization passes deleting the 'after' marker, or mismatched
  // nesting).
  //
  // We must safely remove these leftover markers by bypassing them (connecting
  // their users directly to their operands) and deleting them. Leaving these
  // internal custom call markers in the HLO would cause compilation failures
  // in subsequent backend lowering stages.
  for (const OutlineBlock& block : stack_) {
    HloInstruction* before = block.before;
    if (before->shape().IsTuple()) {
      std::vector<HloInstruction*> before_uses = before->users();
      for (HloInstruction* use : before_uses) {
        if (use->opcode() == HloOpcode::kGetTupleElement) {
          int idx = use->tuple_index();
          RETURN_IF_ERROR(
              use->ReplaceAllUsesWith(before->mutable_operand(idx)));
          RETURN_IF_ERROR(comp->RemoveInstruction(use));
        }
      }
    }
    if (!before->IsDead()) {
      RETURN_IF_ERROR(ReplaceMarkerWithOperands(comp, before));
    }
    TF_RET_CHECK(before->IsDead()) << "before marker still has users";
    RETURN_IF_ERROR(comp->RemoveInstruction(before));
    mutated = true;
  }
  stack_.clear();

  if (mutated) {
    const int64_t after_count = comp->instruction_count();
    LOG(INFO) << "Outlining stats for computation " << comp->name()
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
  std::vector<HloComputation*> comps;
  comps.reserve(module->computation_count());
  for (HloComputation* comp : module->computations()) {
    comps.push_back(comp);
  }

  for (HloComputation* comp : comps) {
    ASSIGN_OR_RETURN(bool comp_mutated,
                     OutlineComputation(module, comp, execution_threads));
    mutated |= comp_mutated;
  }

  if (mutated) {
    module->Cleanup();
    int64_t module_after_count = module->instruction_count();
    int64_t module_after_comp_count = module->computation_count();
    LOG(INFO) << "Outlining stats for module " << module->name()
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
