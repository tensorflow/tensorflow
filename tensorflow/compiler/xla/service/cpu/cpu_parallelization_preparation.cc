/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_parallelization_preparation.h"

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace cpu {

StatusOr<bool> ParallelizationPreparation::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "ParallelizationPreparation ENTRY");
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  TF_ASSIGN_OR_RETURN(changed, RunParallelTaskAssignment(module));

  HloComputation* entry_computation = module->entry_computation();
  std::unordered_set<HloInstruction*> outlined;
  std::vector<HloInstruction*> instructions_to_outline;
  for (HloInstruction* instruction :
       entry_computation->MakeInstructionPostOrder()) {
    // If the instruction has been outlined, it no longer exists and we must not
    // dereference it.
    if (outlined.count(instruction) > 0) {
      continue;
    }

    // Skip parameters and constants, there is nothing to parallelize.
    if (instruction->opcode() == HloOpcode::kParameter ||
        instruction->opcode() == HloOpcode::kConstant) {
      continue;
    }

    // Outline 'instruction' in isolation if it was assigned parallel tasks.
    if (OutlineParallelizableInstruction(instruction)) {
      outlined.insert(instruction);
      changed = true;
      continue;
    }

    instructions_to_outline.clear();
    HloInstruction* outline_candidate = instruction;
    instructions_to_outline.push_back(outline_candidate);

    // Outline sole users with the current instruction.
    while (CanOutlineWithUser(outline_candidate)) {
      HloInstruction* prior_candidate = outline_candidate;
      outline_candidate = *outline_candidate->users().begin();
      if (std::any_of(outline_candidate->operands().begin(),
                      outline_candidate->operands().end(),
                      [&](const HloInstruction* operand) {
                        // Do not consider any candidates which have operands
                        // other than the prior candidate, constants or
                        // parameters. Otherwise, we'd increase the fan-in which
                        // would reduce parallelism.
                        return operand->opcode() != HloOpcode::kParameter &&
                               operand->opcode() != HloOpcode::kConstant &&
                               operand != prior_candidate;
                      })) {
        break;
      }
      instructions_to_outline.push_back(outline_candidate);
    }

    outlined.insert(instructions_to_outline.begin(),
                    instructions_to_outline.end());

    // Optimization to avoid replacing a single existing kCall with another
    // kCall that just calls the first one.
    if (instructions_to_outline.size() == 1 &&
        instructions_to_outline[0]->opcode() == HloOpcode::kCall) {
      continue;
    }

    module->OutlineExpressionFromComputation(
        instructions_to_outline,
        tensorflow::strings::StrCat("pp_", instruction->name()),
        entry_computation);
    changed = true;
  }

  XLA_VLOG_LINES(2, "ParallelizationPreparation EXIT");
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

StatusOr<bool> ParallelizationPreparation::RunParallelTaskAssignment(
    HloModule* module) {
  VLOG(1) << "RunParallelTaskAssignment max_parallelism_: " << max_parallelism_;
  bool changed = false;
  // Initialize ParallelTaskAssignment.
  ParallelTaskAssignment parallel_task_assignment(max_parallelism_, shape_size_,
                                                  module);
  // Assign parallel tasks to HLOs in entry computation.
  HloComputation* computation = module->entry_computation();
  for (auto* instruction : computation->instructions()) {
    // Calculate target parallel task count in [1, max_parallelism_].
    const int64 target_parallel_task_count =
        parallel_task_assignment.GetTargetParallelTaskCount(instruction);
    if (target_parallel_task_count == 1) {
      continue;
    }

    // Assign feasible dimension partitions (based on actual dimension sizes).
    auto dim_partition_counts = ShapePartitionAssigner(instruction->shape())
                                    .Run(target_parallel_task_count);
    const int64 total_partition_count =
        ShapePartitionAssigner::GetTotalPartitionCount(dim_partition_counts);
    if (total_partition_count <= 1) {
      // Feasible partition calculation resulting in no partitioning, so skip.
      continue;
    }
    VLOG(2) << "Assigning parallel task count: " << total_partition_count
            << " to instruction: " << instruction->name();
    // Map 'instruction' to assigned dimension partitioning.
    instruction->set_outer_dimension_partitions(dim_partition_counts);
  }

  return changed;
}

bool ParallelizationPreparation::OutlineParallelizableInstruction(
    HloInstruction* instruction) {
  if (instruction->outer_dimension_partitions().empty()) {
    return false;
  }
  // Store dimension partition counts before outlining (which clones
  // 'instruction').
  std::vector<int64> dim_partition_counts =
      instruction->outer_dimension_partitions();
  // Outline 'instruction' in its own sub-computation.
  HloModule* module = instruction->parent()->parent();
  auto* call = module->OutlineExpressionFromComputation(
      {instruction}, tensorflow::strings::StrCat("pp_", instruction->name()),
      module->entry_computation());
  // Map previously assigned 'dim_partition_counts' to cloned root instruction.
  VLOG(1) << "Outlining parallelizable"
          << " caller: " << call->name()
          << " callee: " << call->to_apply()->root_instruction()->name();
  call->to_apply()->root_instruction()->set_outer_dimension_partitions(
      dim_partition_counts);
  return true;
}

bool ParallelizationPreparation::CanOutlineWithUser(
    HloInstruction* instruction) {
  if (instruction->users().size() != 1) {
    // Do not outline 'instruction' with multiple users.
    return false;
  }
  if (AssignedParallelTasks(instruction) ||
      AssignedParallelTasks(*instruction->users().begin())) {
    // Do not outline if 'instruction' (or user) were assigned parallel tasks.
    return false;
  }
  return true;
}

bool ParallelizationPreparation::AssignedParallelTasks(
    HloInstruction* instruction) {
  return !instruction->outer_dimension_partitions().empty() ||
         (instruction->opcode() == HloOpcode::kCall &&
          !instruction->to_apply()
               ->root_instruction()
               ->outer_dimension_partitions()
               .empty());
}

}  // namespace cpu
}  // namespace xla
