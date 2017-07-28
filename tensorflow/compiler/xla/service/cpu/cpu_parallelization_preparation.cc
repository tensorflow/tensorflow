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
#include "tensorflow/compiler/xla/service/cpu/shape_partition.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/tuple_points_to_analysis.h"
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
    bool all_bitcasts = outline_candidate->opcode() == HloOpcode::kBitcast;

    // Outline sole users with the current instruction.
    while (CanOutlineWithUser(outline_candidate)) {
      HloInstruction* prior_candidate = outline_candidate;
      outline_candidate = *outline_candidate->users().begin();
      all_bitcasts |= outline_candidate->opcode() == HloOpcode::kBitcast;
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
    // If all instructions in the outline candidates are a bitcast, then create
    // a copy at the head of the bitcasts and include it in the outlined
    // instructions. The underlying problem is that a computation which forwards
    // a parameter buffer to the output is not properly handled by the backends
    // or analysis.
    //
    // This would be better handled by being smarter about choosing outline
    // candidates in the first place.
    if (all_bitcasts) {
      // 'head' is the first instruction in the chain of bitcasts.
      HloInstruction* head = instructions_to_outline[0];
      HloInstruction* head_operand = head->mutable_operand(0);
      HloInstruction* copy =
          entry_computation->AddInstruction(HloInstruction::CreateUnary(
              head_operand->shape(), HloOpcode::kCopy, head_operand));
      TF_RETURN_IF_ERROR(head->ReplaceOperandWith(0, copy));
      instructions_to_outline.insert(instructions_to_outline.begin(), copy);
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

  TF_ASSIGN_OR_RETURN(auto points_to_analysis,
                      TuplePointsToAnalysis::Run(module));
  for (auto& computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    HloInstruction* root = computation->root_instruction();
    // Copy root instruction if it does not define its own top-level buffer.
    // TODO(b/32885001) Remove these copies (at least for the unambiguous case).
    // TODO(b/32885001) Perform shallow copy if root value is a tuple.
    if (!points_to_analysis->InstructionDefinesBufferAtIndex(root,
                                                             /*index=*/{})) {
      HloInstruction* copy = computation->AddInstruction(
          HloInstruction::CreateUnary(root->shape(), HloOpcode::kCopy, root));
      computation->set_root_instruction(copy);
      changed = true;
    }
  }

  XLA_VLOG_LINES(2, "ParallelizationPreparation EXIT");
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

StatusOr<bool> ParallelizationPreparation::RunParallelTaskAssignment(
    HloModule* module) {
  VLOG(1) << "RunParallelTaskAssignment max_parallelism_: " << max_parallelism_;
  bool changed = false;
  // Run cost analysis on entry computation.
  HloCostAnalysis cost_analysis(shape_size_);
  HloComputation* computation = module->entry_computation();
  Status cost_status = computation->root_instruction()->Accept(&cost_analysis);
  for (auto& instruction : computation->instructions()) {
    // Currently, we do not assign parallel tasks to instructions with at least
    // one of the following properties:
    // *) Internal threading (library calls to kConv, kDot, and kCustomCall).
    // *) Emit custom loops (kSelectAndScatter, FusionKind::kTransposeDot).
    // *) Tuple-shaped.
    // TODO(b/27458679) Parallelize instructions which are skipped here.
    if (instruction->opcode() == HloOpcode::kParameter ||
        instruction->opcode() == HloOpcode::kConstant ||
        instruction->opcode() == HloOpcode::kCall ||
        instruction->opcode() == HloOpcode::kCustomCall ||
        instruction->opcode() == HloOpcode::kSelectAndScatter ||
        (instruction->opcode() == HloOpcode::kConvolution &&
         PotentiallyImplementedAsEigenConvolution(*instruction)) ||
        PotentiallyImplementedAsEigenDot(*instruction) ||
        (instruction->opcode() == HloOpcode::kFusion &&
         instruction->fusion_kind() != HloInstruction::FusionKind::kLoop) ||
        ShapeUtil::IsTuple(instruction->shape())) {
      continue;
    }

    // Calculate target parallel task count in [1, max_parallelism_].
    const int64 target_parallel_task_count = GetTargetParallelTaskCount(
        cost_status.ok() ? &cost_analysis : nullptr, instruction.get());
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

int64 ParallelizationPreparation::GetTargetParallelTaskCount(
    const HloCostAnalysis* cost_analysis, HloInstruction* instruction) {
  // Default to a simple cost model based on hlo size and typical L2 cache size.
  // Note that 'cost_analysis' can be 'nullptr' if HloCostAnalysis returns an
  // error status (likely because HLOs like CustomCall are not yet implemented
  // in the HloCostAnalysis).
  int64 instruction_cost = shape_size_(instruction->shape());
  int64 min_cost_per_thread = 256LL << 10;  // 256KB L2 Cache size.
  if (cost_analysis != nullptr) {
    // Calculate the instruction cost in cycles.
    // TODO(29630486) Improve on this linear cost model.
    // Consider making 'min_cost_per_thread' be a function of the target
    // bandwidth limit for instructions with low arithmetic complexity.
    instruction_cost = 1 * cost_analysis->flop_count(*instruction) +
                       2 * cost_analysis->transcendental_count(*instruction) +
                       10 * cost_analysis->bytes_accessed(*instruction);
    // Minimum per-thread cost is 100us of work on a 2GHz core.
    min_cost_per_thread = 100000;
  }
  // Return target parallel task count in [1, max_parallelism_].
  return std::min(max_parallelism_,
                  std::max(1LL, instruction_cost / min_cost_per_thread));
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
