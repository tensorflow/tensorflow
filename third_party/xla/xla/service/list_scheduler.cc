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

#include "xla/service/list_scheduler.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

// Returns the predecessors the provided instruction.
absl::flat_hash_set<HloInstruction*> Predecessors(const HloInstruction* inst) {
  absl::flat_hash_set<HloInstruction*> preds;
  preds.insert(inst->operands().begin(), inst->operands().end());
  preds.insert(inst->control_predecessors().begin(),
               inst->control_predecessors().end());
  return preds;
}

// Returns the successors the provided instruction.
absl::flat_hash_set<HloInstruction*> Successors(const HloInstruction* inst) {
  absl::flat_hash_set<HloInstruction*> succs;
  succs.insert(inst->users().begin(), inst->users().end());
  succs.insert(inst->control_successors().begin(),
               inst->control_successors().end());
  return succs;
}

// Returns the values used by the provided instruction.
absl::flat_hash_set<const HloValue*> UsedValues(
    const HloInstruction* inst, const HloDataflowAnalysis& dataflow) {
  absl::flat_hash_set<const HloValue*> used_values;
  for (const HloInstruction* operand : inst->unique_operands()) {
    HloValueSet operand_values = dataflow.GetFlattenedValueSet(operand);
    for (const HloValue* value : operand_values.values()) {
      // Check if inst actually uses this value.
      for (const HloUse& use : value->GetUses()) {
        if (use.instruction == inst) {
          used_values.insert(value);
          break;
        }
      }
    }
  }
  return used_values;
}

// Returns an estimate of the number of bytes allocated by the instruction minus
// the number of bytes freed by the instruction.
int64_t EstimateMemoryChange(
    const HloInstruction* inst, const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_map<const HloValue*, int>& num_users) {
  constexpr int64_t kPointerSize = 8;

  // Estimate the number of bytes "allocated" by this instruction.
  int64_t allocated = 0;
  HloValueSet values = dataflow.GetFlattenedValueSet(inst);
  for (const HloValue* value : values.values()) {
    allocated += ShapeUtil::ByteSizeOf(value->shape(), kPointerSize);
  }

  // Estimate the number of bytes "freed" by this instruction.
  int64_t freed = 0;
  for (const HloValue* value : UsedValues(inst, dataflow)) {
    // If this instruction is the last user of a value, then the value can be
    // freed after the instruction is run.
    auto it = num_users.find(value);
    if (it != num_users.end() && it->second == 1) {
      freed += ShapeUtil::ByteSizeOf(value->shape(), kPointerSize);
    }
  }

  return allocated - freed;
}

// Picks the next instruction to schedule from the provided set of instructions.
HloInstruction* PickInstructionToSchedule(
    const absl::flat_hash_set<HloInstruction*>& instructions,
    const HloDataflowAnalysis& dataflow,
    const absl::flat_hash_map<const HloValue*, int>& num_users) {
  CHECK(!instructions.empty());

  int64_t min_cost = std::numeric_limits<int64_t>::max();
  HloInstruction* best_instruction = nullptr;
  for (HloInstruction* inst : instructions) {
    // Always schedule async start instructions when possible.
    if (HloDataflowAnalysis::IsAsynchronousOperationStart(inst->opcode())) {
      return inst;
    }

    // Only schedule async done instructions if we have to.
    if (HloDataflowAnalysis::IsAsynchronousOperationDone(inst->opcode())) {
      if (best_instruction == nullptr) {
        best_instruction = inst;
      }
      continue;
    }

    // Pick the instruction with the largest decrease in memory pressure.
    int64_t cost = EstimateMemoryChange(inst, dataflow, num_users);
    if (cost < min_cost) {
      min_cost = cost;
      best_instruction = inst;
    }
  }

  CHECK(best_instruction != nullptr);
  return best_instruction;
}

}  // namespace

absl::StatusOr<bool> ListScheduler::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloSchedule schedule(module);

  // Run dataflow analysis. To compute the change in memory pressure from
  // running an instruction, we have to know which values the instruction might
  // free. Dataflow analysis tells us this.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloDataflowAnalysis> dataflow,
                      HloDataflowAnalysis::Run(*module));

  // Schedule each computation.
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    // Compute the in_degree of every instruction.
    absl::flat_hash_map<HloInstruction*, int> in_degree;
    for (HloInstruction* inst : computation->instructions()) {
      in_degree[inst] = Predecessors(inst).size();
    }

    // Count the number of users of every value.
    absl::flat_hash_map<const HloValue*, int> num_users;
    for (HloInstruction* inst : computation->instructions()) {
      HloValueSet values = dataflow->GetFlattenedValueSet(inst);
      for (const HloValue* value : values.values()) {
        num_users[value] = value->GetUses().size();
      }
    }

    // Populate the frontier of a topological ordering.
    absl::flat_hash_set<HloInstruction*> frontier;
    for (auto& [inst, degree] : in_degree) {
      if (degree == 0) {
        frontier.insert(inst);
      }
    }

    // Perform the topological sort.
    HloInstructionSequence& computation_schedule =
        schedule.GetOrCreateSequence(computation);
    while (!frontier.empty()) {
      // Pick the next instruction to schedule from the frontier.
      HloInstruction* inst =
          PickInstructionToSchedule(frontier, *dataflow, num_users);
      computation_schedule.push_back(inst);
      frontier.erase(inst);

      // Update in-degrees.
      for (HloInstruction* succ : Successors(inst)) {
        in_degree[succ]--;
        if (in_degree[succ] == 0) {
          frontier.insert(succ);
        }
      }

      // Update num_users.
      for (const HloValue* value : UsedValues(inst, *dataflow)) {
        num_users[value]--;
      }
    }

    CHECK_EQ(computation_schedule.size(), computation->instruction_count())
        << "Not all instructions were scheduled in computation "
        << computation->name();
  }

  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  return true;
}

}  // namespace xla
