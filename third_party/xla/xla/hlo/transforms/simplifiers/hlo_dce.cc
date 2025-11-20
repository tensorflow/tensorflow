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

#include "xla/hlo/transforms/simplifiers/hlo_dce.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/computation_layout.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

namespace {

const absl::string_view kDceSideEffectFrontendAttribute =
    "xla_allow_dce_side_effecting_op";

// Checks if the instruction is a removable while given
// remove_cross_partition_collective_ops
bool IsRemovableWhile(const HloInstruction* instruction,
                      bool remove_cross_partition_collective_ops) {
  if (instruction->opcode() != HloOpcode::kWhile) {
    return false;
  }
  for (HloComputation* computation : instruction->called_computations()) {
    for (HloInstruction* called_instr : computation->instructions()) {
      auto maybe_collective_op =
          DynCast<HloCollectiveInstruction>(called_instr);
      if (called_instr->HasSideEffect() &&
          (!remove_cross_partition_collective_ops || !maybe_collective_op ||
           maybe_collective_op->constrain_layout())) {
        return false;
      }
    }
  }
  return true;
}

// Updates the users of the fusion instruction.
absl::Status UpdateFusionUsers(HloInstruction* fusion_instruction,
                               const std::set<int64_t>& used_tuple_elements,
                               const std::vector<Shape>& tuple_shapes) {
  if (tuple_shapes.size() > 1) {
    for (HloInstruction* gte : fusion_instruction->users()) {
      auto it = used_tuple_elements.lower_bound(gte->tuple_index());
      int64_t new_tuple_index = std::distance(used_tuple_elements.begin(), it);
      gte->set_tuple_index(new_tuple_index);
    }
  } else {
    // Since we iterate over users while removing them .. make a local copy
    // first.
    std::vector<HloInstruction*> users(fusion_instruction->users());
    for (HloInstruction* gte : users) {
      // Replace and change control successors to be dependent on the fusion
      // instruction itself.
      TF_ASSIGN_OR_RETURN(std::ignore, gte->parent()->ReplaceInstruction(
                                           gte, fusion_instruction,
                                           /*preserve_sharding=*/true,
                                           /*relay_control_dependency=*/true));
    }
  }
  return absl::OkStatus();
}

// Returns true if it found and removed unused outputs.
absl::StatusOr<bool> RemoveMultiOutputFusionsUnusedOutputs(
    HloComputation* computation) {
  HloInstruction* fusion_instruction = computation->FusionInstruction();
  if (!fusion_instruction) {
    return false;
  }

  if (computation->root_instruction()->opcode() != HloOpcode::kTuple ||
      computation->root_instruction()->has_sharding() ||
      !fusion_instruction->output_operand_aliasing().empty() ||
      fusion_instruction->HasControlDependencies() ||
      fusion_instruction->IsCustomFusion()) {
    return false;
  }

  // The order of the used outputs is relevant for the algorithm below.
  std::set<int64_t> used_tuple_elements;

  // We only support this cleanup if all users of the fusion instruction are
  // GetTupleElement ops, and there is at least one user of
  // 'fusion_instruction'.
  if (fusion_instruction->users().empty()) {
    return false;
  }

  for (HloInstruction* gte : fusion_instruction->users()) {
    if (gte->opcode() != HloOpcode::kGetTupleElement) {
      return false;
    }
    used_tuple_elements.insert(gte->tuple_index());
  }

  // If all outputs are used, nothing to clean up.
  if (used_tuple_elements.size() ==
      computation->root_instruction()->operand_count()) {
    return false;
  }

  std::vector<Shape> tuple_shapes;
  tuple_shapes.reserve(used_tuple_elements.size());
  for (int64_t tuple_index : used_tuple_elements) {
    tuple_shapes.push_back(
        fusion_instruction->shape().tuple_shapes(tuple_index));
  }
  Shape new_shape = tuple_shapes.size() == 1
                        ? tuple_shapes[0]
                        : ShapeUtil::MakeTupleShape(tuple_shapes);
  *fusion_instruction->mutable_shape() = std::move(new_shape);

  // Update the users of the old fusion instruction.
  TF_RETURN_IF_ERROR(
      UpdateFusionUsers(fusion_instruction, used_tuple_elements, tuple_shapes));

  // Update the root of the fusion computation.
  if (tuple_shapes.size() > 1) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(used_tuple_elements.size());
    for (int64_t tuple_index : used_tuple_elements) {
      new_operands.push_back(
          computation->root_instruction()->mutable_operand(tuple_index));
    }
    auto new_tuple =
        computation->AddInstruction(HloInstruction::CreateTuple(new_operands));
    TF_RETURN_IF_ERROR(computation->ReplaceInstructionWithDifferentShape(
        computation->root_instruction(), new_tuple));
  } else {
    TF_RETURN_IF_ERROR(
        computation->root_instruction()->ReplaceAllUsesWithDifferentShape(
            computation->root_instruction()->mutable_operand(
                *used_tuple_elements.begin())));
  }

  // We always updated the fusion if we got here.
  return true;
}

bool CanRemoveInstruction(
    const HloInstruction* instruction,
    bool remove_cross_partition_collective_ops,
    const std::function<std::vector<HloInstruction*>(const HloComputation*)>&
        computation_callers) {
  if (!instruction->IsDead()) {
    return false;
  }
  if (!instruction->parent()->IsSafelyRemovable(
          instruction,
          /*ignore_control_dependency=*/false,
          /*computation_callers=*/computation_callers)) {
    return false;
  }
  // We cannot remove a parameter directly, because it may cause a
  // renumbering of other parameters which may invalidate some of the
  // pointers in the worklist.
  if (instruction->opcode() == HloOpcode::kParameter) {
    return false;
  }
  if (instruction->IsCustomCall("Sharding") &&
      (instruction->operand(0)->IsRoot() ||
       instruction->operand(0)->opcode() == HloOpcode::kParameter ||
       instruction->operand(0)->user_count() != 1)) {
    return false;
  }
  if (instruction->HasSideEffect()) {
    auto maybe_collective_op = DynCast<HloCollectiveInstruction>(instruction);
    bool allow_collective = remove_cross_partition_collective_ops &&
                            maybe_collective_op &&
                            !maybe_collective_op->constrain_layout();
    bool allow_while =
        IsRemovableWhile(instruction, remove_cross_partition_collective_ops);
    bool allow_custom_call = instruction->IsCustomCall("tpu_custom_call") &&
                             instruction->frontend_attributes().map().contains(
                                 kDceSideEffectFrontendAttribute) &&
                             instruction->frontend_attributes().map().at(
                                 kDceSideEffectFrontendAttribute) == "true";
    if (!allow_collective && !allow_while && !allow_custom_call) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<bool> RemoveDeadRoots(
    HloComputation* computation, bool remove_cross_partition_collective_ops,
    const std::function<std::vector<HloInstruction*>(const HloComputation*)>&
        computation_callers) {
  bool changed = false;
  std::vector<HloInstruction*> dead_roots;
  for (auto* instruction : computation->instructions()) {
    if (!CanRemoveInstruction(instruction,
                              remove_cross_partition_collective_ops,
                              computation_callers)) {
      continue;
    }
    dead_roots.push_back(instruction);
  }

  for (HloInstruction* dead_root : dead_roots) {
    VLOG(1) << "Removing dead root " << dead_root->ToString()
            << " and its unused operands";
    TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
        dead_root, /*cleanup=*/std::nullopt,
        /*ignore_control_dependencies=*/false,
        /*computation_callers=*/computation_callers));
    changed = true;
  }
  return changed;
}
absl::Status RemoveDeadParametersFromEntryComputationLayout(
    HloModule* module, std::vector<int64_t>& dead_parameter_indexes) {
  if (dead_parameter_indexes.empty()) {
    return absl::OkStatus();
  }
  const ComputationLayout& old_layout = module->entry_computation_layout();
  ShapeLayout result_layout = old_layout.result_layout();
  ComputationLayout new_layout(result_layout);
  for (int i = 0; i < old_layout.parameter_count(); ++i) {
    if (absl::c_linear_search(dead_parameter_indexes, i)) {
      continue;
    }
    new_layout.add_parameter_layout(old_layout.parameter_layout(i));
  }
  *module->mutable_entry_computation_layout() = std::move(new_layout);
  return absl::OkStatus();
}

absl::StatusOr<bool> RemoveDeadParameters(
    HloComputation* computation,
    const std::function<std::vector<HloInstruction*>(const HloComputation*)>&
        computation_callers,
    bool remove_dead_parameters_from_entry_computation) {
  bool changed = false;
  bool update_entry_computation_layout =
      computation->IsEntryComputation() &&
      remove_dead_parameters_from_entry_computation;
  auto parameters = computation->parameter_instructions();
  // Sort into decreasing order by parameter number, otherwise the renumbering
  // of parameters when one parameter is deleted will cause issues.
  absl::c_reverse(parameters);
  std::vector<int64_t> dead_parameters;
  for (HloInstruction* parameter : parameters) {
    if (parameter->IsDead() &&
        computation->IsSafelyRemovable(
            parameter,
            /*ignore_control_dependency=*/false,
            /*computation_callers=*/computation_callers,
            remove_dead_parameters_from_entry_computation)) {
      VLOG(1) << "Removing dead parameter " << parameter->ToString()
              << " and its unused operands";
      int64_t num_parameters = computation->num_parameters();
      int64_t parameter_number = parameter->parameter_number();
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
          parameter, /*cleanup=*/std::nullopt,
          /*ignore_control_dependencies=*/false,
          /*computation_callers=*/computation_callers,
          remove_dead_parameters_from_entry_computation));
      if (computation->num_parameters() < num_parameters) {
        changed = true;
        if (update_entry_computation_layout) {
          dead_parameters.push_back(parameter_number);
        }
      }
    }
  }
  if (update_entry_computation_layout) {
    TF_RETURN_IF_ERROR(RemoveDeadParametersFromEntryComputationLayout(
        computation->parent(), dead_parameters));
  }
  return changed;
}

void PopulateAgenda(HloModule* module, std::stack<HloComputation*>& agenda,
                    absl::flat_hash_set<HloComputation*>& to_remove) {
  // Use computations from all execution threads when determining reachability.
  for (HloComputation* computation : module->computations()) {
    to_remove.insert(computation);
  }
  agenda.push(module->entry_computation());
  to_remove.erase(module->entry_computation());
}

// Run DCE on each computation. Visit callers before callees so that we
// cleanup dead get-tuple-element users of MultiOutput fusions before cleaning
// up the fusion computation. If the same callee is referred to by multiple
// callers we'll only visit the first caller before visiting the callee, but
// that's ok for the use case of fusion computations that should have a unique
// calling instruction anyway.
absl::StatusOr<bool> ProcessAgenda(
    HloModule* module, std::stack<HloComputation*>& agenda,
    absl::flat_hash_set<HloComputation*>& to_remove,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    bool remove_cross_partition_collective_ops, CallGraph* call_graph,
    bool remove_dead_parameters_from_entry_computation) {
  bool changed = false;
  while (!agenda.empty()) {
    HloComputation* computation = agenda.top();
    agenda.pop();

    if (execution_threads.empty() ||
        execution_threads.contains(computation->execution_thread())) {
      TF_ASSIGN_OR_RETURN(
          bool computation_changed,
          xla::HloDCE::RunOnComputation(
              computation, remove_cross_partition_collective_ops, call_graph,
              remove_dead_parameters_from_entry_computation));
      changed |= computation_changed;
    }

    for (auto* instruction : computation->instructions()) {
      for (HloComputation* called_computation :
           instruction->called_computations()) {
        if (to_remove.erase(called_computation) > 0) {
          agenda.push(called_computation);
        }
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> RemoveDanglingComputations(
    HloModule* module, absl::flat_hash_set<HloComputation*>& to_remove,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    const bool use_call_analysis, std::unique_ptr<CallGraph>& call_graph) {
  bool changed = false;
  // Some computations might have been left dangling due to being detached
  // indirectly. We need to rebuild the call graph to find these.
  if (use_call_analysis) {
    call_graph = CallGraph::Build(module);
    for (HloComputation* computation :
         module->computations(execution_threads)) {
      if (!computation->IsEntryComputation() &&
          call_graph->GetComputationCallers(computation).empty()) {
        to_remove.insert(computation);
      }
    }
  }
  for (auto iterator = module->computations().begin();
       iterator != module->computations().end(); ++iterator) {
    // Only remove computations from the specified execution threads.
    auto computation = *iterator;
    if (to_remove.contains(computation)) {
      if (execution_threads.empty() ||
          execution_threads.contains(computation->execution_thread())) {
        TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(
            iterator.underlying_iterator().underlying_iterator()));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace

/*static*/ absl::StatusOr<bool> HloDCE::RunOnComputation(
    HloComputation* computation, bool remove_cross_partition_collective_ops,
    CallGraph* call_graph, bool remove_dead_parameters_from_entry_computation) {
  auto computation_callers =
      [call_graph](
          const HloComputation* computation) -> std::vector<HloInstruction*> {
    if (call_graph == nullptr) {
      return {};
    }
    return call_graph->GetComputationCallers(computation);
  };

  bool changed = false;
  TF_ASSIGN_OR_RETURN(bool fusion_changed,
                      RemoveMultiOutputFusionsUnusedOutputs(computation));
  changed |= fusion_changed;

  TF_ASSIGN_OR_RETURN(
      bool dead_roots_changed,
      RemoveDeadRoots(computation, remove_cross_partition_collective_ops,
                      computation_callers));
  changed |= dead_roots_changed;

  TF_ASSIGN_OR_RETURN(
      bool dead_parameters_changed,
      RemoveDeadParameters(computation, computation_callers,
                           remove_dead_parameters_from_entry_computation));
  changed |= dead_parameters_changed;

  return changed;
}

absl::StatusOr<bool> HloDCE::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Before dce; threads: " << absl::StrJoin(execution_threads, ",");
  XLA_VLOG_LINES(2, module->ToString());

  std::unique_ptr<CallGraph> call_graph;
  if (use_call_analysis_) {
    call_graph = CallGraph::Build(module);
  }

  bool changed = false;

  std::stack<HloComputation*> agenda;
  absl::flat_hash_set<HloComputation*> to_remove;
  PopulateAgenda(module, agenda, to_remove);

  TF_ASSIGN_OR_RETURN(
      bool agenda_changed,
      ProcessAgenda(module, agenda, to_remove, execution_threads,
                    remove_cross_partition_collective_ops_, call_graph.get(),
                    remove_dead_parameters_from_entry_computation_));
  changed |= agenda_changed;

  TF_ASSIGN_OR_RETURN(
      bool dangling_computations_removed,
      RemoveDanglingComputations(module, to_remove, execution_threads,
                                 use_call_analysis_, call_graph));
  changed |= dangling_computations_removed;

  if (changed) {
    // Update the schedule to reflect the removed instructions.
    if (module->has_schedule()) {
      TF_RETURN_IF_ERROR(module->schedule().Update(execution_threads));
    }
    VLOG(2) << "After dce:";
    XLA_VLOG_LINES(2, module->ToString());
  }

  return changed;
}

}  // namespace xla
