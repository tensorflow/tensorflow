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

#include "xla/service/cpu/small_while_loop_hoisting_pass.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

static bool InstructionIsUnavailable(const HloInstruction* instr) {
  switch (instr->opcode()) {
    // The following instructions are not currently supported by the call thunk
    // emitter due to how the legacy & thunk emitters interact; specifically,
    // how the run options are passed.
    case HloOpcode::kCustomCall:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kScatter:
    case HloOpcode::kSort:
    case HloOpcode::kFft:
    case HloOpcode::kPartitionId:
    case HloOpcode::kReplicaId:
    case HloOpcode::kAfterAll:
      return true;

    // Legacy call emitter does not support custom fusions.
    case HloOpcode::kFusion:
      return instr->fusion_kind() == HloInstruction::FusionKind::kCustom;

    default:
      return IsCollective(instr);
  }
}

// Simple DFS check to see if a computation contains any instructions that are
// "unavailable" for the call thunk emitter.
static bool ContainsUnavailableInstruction(
    const HloInstruction* instr,
    absl::flat_hash_map<const HloInstruction*, bool>& has_unavailable_instr) {
  if (const auto itr = has_unavailable_instr.find(instr);
      itr != has_unavailable_instr.end()) {
    return itr->second;
  }

  if (InstructionIsUnavailable(instr)) {
    return has_unavailable_instr.insert({instr, true}).first->second;
  }

  for (const HloComputation* called_comp : instr->called_computations()) {
    for (const HloInstruction* sub_instr : called_comp->instructions()) {
      if (ContainsUnavailableInstruction(sub_instr, has_unavailable_instr)) {
        return has_unavailable_instr.insert({instr, true}).first->second;
      }
    }
  }

  return has_unavailable_instr.insert({instr, false}).first->second;
}

static int64_t GetInstructionBytesAccessed(const HloInstruction* instr) {
  HloCostAnalysis cost_analysis(&CpuExecutable::ShapeSizeBytes);
  if (!cost_analysis.RevisitInstruction(instr).ok()) {
    return 0;
  }
  return cost_analysis.bytes_accessed(*instr);
}

static bool IsNonTrivialOp(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kParameter:
    case HloOpcode::kConstant:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kBitcast:
    case HloOpcode::kAfterAll:
      return false;
    default:
      return true;
  }
}

SmallWhileLoopHoistingPass::SmallWhileLoopHoistingPass(
    int64_t small_buffer_access_size)
    : small_buffer_access_size_(small_buffer_access_size) {}

bool SmallWhileLoopHoistingPass::IsBeneficialRun(
    const InstructionRun& run) const {
  if (run.instructions.empty()) return false;
  if (run.total_bytes_accessed > small_buffer_access_size_) return false;
  if (run.contains_while_loop) return true;
  for (const HloInstruction* inst : run.instructions) {
    if (IsNonTrivialOp(inst)) {
      return true;
    }
  }
  return false;
}

std::vector<InstructionRun> SmallWhileLoopHoistingPass::IdentifyCandidateRuns(
    HloComputation* comp,
    absl::flat_hash_map<const HloInstruction*, bool>& unavailable_cache) {
  std::vector<InstructionRun> candidate_runs;
  std::vector<HloInstruction*> topo = comp->MakeInstructionPostOrder();

  InstructionRun current_run;
  for (HloInstruction* instr : topo) {
    if (instr->opcode() == HloOpcode::kParameter ||
        ContainsUnavailableInstruction(instr, unavailable_cache)) {
      if (IsBeneficialRun(current_run)) {
        candidate_runs.push_back(std::move(current_run));
      }
      current_run = InstructionRun();
      continue;
    }

    if (instr->opcode() == HloOpcode::kConstant ||
        (instr->opcode() == HloOpcode::kCopy &&
         instr->operand(0)->opcode() == HloOpcode::kConstant)) {
      continue;
    }

    int64_t bytes = GetInstructionBytesAccessed(instr);
    if (current_run.total_bytes_accessed + bytes > small_buffer_access_size_ &&
        !current_run.instructions.empty()) {
      if (IsBeneficialRun(current_run)) {
        candidate_runs.push_back(std::move(current_run));
      }
      current_run = InstructionRun();
    }

    current_run.instructions.push_back(instr);
    current_run.total_bytes_accessed += bytes;
    if (instr->opcode() == HloOpcode::kWhile) {
      current_run.contains_while_loop = true;
    }
  }

  if (IsBeneficialRun(current_run)) {
    candidate_runs.push_back(std::move(current_run));
  }

  return candidate_runs;
}

absl::StatusOr<bool> SmallWhileLoopHoistingPass::OutlineRun(
    HloComputation* comp, const InstructionRun& run, HloModule* module) {
  absl::flat_hash_set<HloInstruction*> run_set(run.instructions.begin(),
                                               run.instructions.end());

  std::vector<HloInstruction*> external_operands;
  absl::flat_hash_map<HloInstruction*, int64_t> operand_to_param;
  std::vector<HloInstruction*> external_outputs;

  for (HloInstruction* inst : run.instructions) {
    for (HloInstruction* op : inst->operands()) {
      if (!run_set.contains(op) && !operand_to_param.contains(op)) {
        operand_to_param[op] = external_operands.size();
        external_operands.push_back(op);
      }
    }

    bool used_outside = false;
    for (HloInstruction* user : inst->users()) {
      if (!run_set.contains(user)) {
        used_outside = true;
        break;
      }
    }
    if (used_outside || inst == comp->root_instruction()) {
      external_outputs.push_back(inst);
    }
  }

  if (external_outputs.empty()) {
    return false;
  }

  HloComputation::Builder builder(
      absl::StrCat(run.instructions.back()->name(), "_run_computation"));

  std::vector<HloInstruction*> param_instructions;
  param_instructions.reserve(external_operands.size());
  absl::flat_hash_map<HloInstruction*, HloInstruction*> cloned_map;

  for (int64_t i = 0; i < external_operands.size(); ++i) {
    HloInstruction* op = external_operands[i];
    ASSIGN_OR_RETURN(HloInstruction * param,
                     builder.AddParameter(HloInstruction::CreateParameter(
                         i, op->shape(), op->name())));
    param_instructions.push_back(param);
    cloned_map[op] = param;
  }

  for (HloInstruction* inst : run.instructions) {
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(inst->operand_count());
    for (HloInstruction* op : inst->operands()) {
      new_operands.push_back(cloned_map.at(op));
    }
    cloned_map[inst] = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), new_operands));
  }

  for (HloInstruction* inst : run.instructions) {
    for (HloInstruction* pred : inst->control_predecessors()) {
      if (run_set.contains(pred)) {
        RETURN_IF_ERROR(
            cloned_map.at(pred)->AddControlDependencyTo(cloned_map.at(inst)));
      }
    }
  }

  HloInstruction* root_instruction = nullptr;
  Shape call_shape;
  if (external_outputs.size() == 1) {
    root_instruction = cloned_map.at(external_outputs[0]);
    call_shape = external_outputs[0]->shape();
  } else {
    std::vector<HloInstruction*> tuple_elements;
    std::vector<Shape> tuple_shapes;
    tuple_elements.reserve(external_outputs.size());
    tuple_shapes.reserve(external_outputs.size());
    for (HloInstruction* out : external_outputs) {
      tuple_elements.push_back(cloned_map.at(out));
      tuple_shapes.push_back(out->shape());
    }
    root_instruction =
        builder.AddInstruction(HloInstruction::CreateTuple(tuple_elements));
    call_shape = ShapeUtil::MakeTupleShape(tuple_shapes);
  }

  HloComputation* embedded_comp =
      module->AddEmbeddedComputation(builder.Build(root_instruction));
  embedded_comp->SetExecutionThread(comp->execution_thread());

  HloInstruction* call_instruction = comp->AddInstruction(
      HloInstruction::CreateCall(call_shape, external_operands, embedded_comp));
  call_instruction->add_frontend_attribute("xla_cpu_small_call", "true");

  if (external_outputs.size() == 1) {
    HloInstruction* out_inst = external_outputs[0];
    bool is_comp_root = (comp->root_instruction() == out_inst);
    RETURN_IF_ERROR(out_inst->ReplaceAllUsesWith(call_instruction));
    if (is_comp_root) {
      comp->set_root_instruction(call_instruction);
    }
  } else {
    for (int64_t j = 0; j < external_outputs.size(); ++j) {
      HloInstruction* out_inst = external_outputs[j];
      bool is_comp_root = (comp->root_instruction() == out_inst);
      HloInstruction* gte =
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              out_inst->shape(), call_instruction, j));
      RETURN_IF_ERROR(out_inst->ReplaceAllUsesWith(gte));
      if (is_comp_root) {
        comp->set_root_instruction(gte);
      }
    }
  }

  for (HloInstruction* inst : run.instructions) {
    for (HloInstruction* pred : inst->control_predecessors()) {
      if (!run_set.contains(pred)) {
        RETURN_IF_ERROR(pred->AddControlDependencyTo(call_instruction));
      }
    }
    for (HloInstruction* succ : inst->control_successors()) {
      if (!run_set.contains(succ)) {
        RETURN_IF_ERROR(call_instruction->AddControlDependencyTo(succ));
      }
    }
  }

  for (auto it = run.instructions.rbegin(); it != run.instructions.rend();
       ++it) {
    HloInstruction* inst = *it;
    RETURN_IF_ERROR(inst->SafelyDropAllControlDependencies());
    if (inst->parent() == comp) {
      RETURN_IF_ERROR(comp->RemoveInstruction(inst));
    }
  }

  return true;
}

absl::StatusOr<bool> SmallWhileLoopHoistingPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  absl::flat_hash_map<const HloInstruction*, bool> unavailable_cache;

  absl::flat_hash_set<const HloComputation*> subcomputations_to_skip;
  for (const HloComputation* comp : module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kWhile ||
          instr->opcode() == HloOpcode::kFusion ||
          instr->opcode() == HloOpcode::kConditional ||
          instr->opcode() == HloOpcode::kReduce ||
          ContainsUnavailableInstruction(instr, unavailable_cache)) {
        for (const HloComputation* called : instr->called_computations()) {
          subcomputations_to_skip.insert(called);
        }
      }
    }
  }

  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    if (subcomputations_to_skip.contains(comp)) {
      continue;
    }
    std::vector<InstructionRun> candidate_runs =
        IdentifyCandidateRuns(comp, unavailable_cache);
    for (const InstructionRun& run : candidate_runs) {
      ASSIGN_OR_RETURN(bool run_changed, OutlineRun(comp, run, module));
      changed |= run_changed;
    }
  }

  return changed;
}

}  // namespace xla::cpu
