/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/command_buffer_scheduling.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace {

// We categorize HLO instructions into two types.
// 1. Commands: Instructions that correspond to a command that will be
// submitted to a GPU. Fused computations and library calls fall into this
// category.
// 2. Intermediates: Instructions that produce intermediate values that are
// used by commands.
bool IsCommand(const HloInstruction* inst) {
  // TODO(anlunx): Add support for conditionals and while loops.
  return inst->opcode() == HloOpcode::kFusion;
}

bool IsIntermediate(const HloInstruction* inst) {
  switch (inst->opcode()) {
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
      return true;
    default:
      return false;
  }
}

void RemoveTrailingIntermediates(HloInstructionSequence& seq) {
  std::vector<HloInstruction*> instructions = seq.instructions();
  for (int i = instructions.size() - 1; i >= 0; i--) {
    HloInstruction* inst = instructions[i];
    if (IsIntermediate(inst)) {
      seq.remove_instruction(inst);
    } else {
      break;
    }
  }
}

constexpr int kMinNumCommands = 2;

}  // namespace

// The input is a scheduled sequence of instructions. This function collects
// subsequences that will be extracted as command buffers.
std::vector<HloInstructionSequence>
CommandBufferScheduling::CollectCommandBufferSequences(
    const HloInstructionSequence inst_sequence) {
  struct Accumulator {
    std::vector<HloInstructionSequence> sequences;
    HloInstructionSequence current_seq;
    int num_commands_in_current_seq = 0;
  };

  auto start_new_sequence = [](Accumulator* acc) -> Accumulator* {
    if (acc->num_commands_in_current_seq >= kMinNumCommands) {
      RemoveTrailingIntermediates(acc->current_seq);
      acc->sequences.push_back(acc->current_seq);
    }
    acc->current_seq = HloInstructionSequence();
    acc->num_commands_in_current_seq = 0;
    return acc;
  };

  auto process_instruction = [&start_new_sequence](
                                 Accumulator* acc,
                                 HloInstruction* inst) -> Accumulator* {
    if (IsCommand(inst)) {
      acc->current_seq.push_back(inst);
      acc->num_commands_in_current_seq += 1;
      return acc;
    } else if (IsIntermediate(inst)) {
      if (acc->current_seq.size() > 0) {
        acc->current_seq.push_back(inst);
      }
      return acc;
    }
    return start_new_sequence(acc);
  };

  std::vector<HloInstruction*> instructions = inst_sequence.instructions();
  Accumulator acc;
  absl::c_accumulate(instructions, &acc, process_instruction);
  return start_new_sequence(&acc)->sequences;
}

// This function moves kParameter instructions in a computation to the beginning
// of the computation. This simplifies the construction of command buffer
// computations because we don't need to consider kParameter's as intermediates.
void CommandBufferScheduling::MoveParametersToFront(
    HloComputation* computation) {
  HloSchedule& schedule = computation->parent()->schedule();
  HloInstructionSequence& sequence = schedule.GetOrCreateSequence(computation);
  std::vector<HloInstruction*> new_sequence;
  for (HloInstruction* inst : sequence.instructions()) {
    if (inst->opcode() == HloOpcode::kParameter) {
      new_sequence.push_back(inst);
    }
  }

  for (HloInstruction* inst : sequence.instructions()) {
    if (inst->opcode() != HloOpcode::kParameter) {
      new_sequence.push_back(inst);
    }
  }

  schedule.set_sequence(computation, new_sequence);
}

StatusOr<CommandBufferScheduling::BuildCommandBufferResult>
CommandBufferScheduling::BuildCommandBuffer(HloInstructionSequence seq) {
  auto builder = HloComputation::Builder("command_buffer");
  const std::vector<HloInstruction*>& instructions = seq.instructions();

  // The sequence might use results of instructions that are not captured by the
  // sequence. We pass those results as parameters and map the producers of the
  // results to their corresponding parameter instructions.
  absl::flat_hash_map<HloInstruction*, HloParameterInstruction*> parameters_map;
  int64_t parameter_number = 0;
  for (HloInstruction* inst : instructions) {
    for (HloInstruction* operand : inst->operands()) {
      if (absl::c_find(instructions, operand) != instructions.end()) {
        continue;
      }

      if (!parameters_map.contains(operand)) {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * parameter,
            builder.AddParameter(HloInstruction::CreateParameter(
                parameter_number, operand->shape(), "param")));
        parameter_number++;
        parameters_map[operand] =
            static_cast<HloParameterInstruction*>(parameter);
      }
    }
  }

  // We copy instructions from the sequence to the computation and map the
  // original instruction to its clone.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> instructions_map;
  for (HloInstruction* inst : seq.instructions()) {
    switch (inst->opcode()) {
      case HloOpcode::kFusion: {
        std::vector<HloInstruction*> operands;
        for (HloInstruction* operand : inst->operands()) {
          auto it = parameters_map.find(operand);
          if (it != parameters_map.end()) {
            operands.push_back(it->second);
          } else {
            operands.push_back(instructions_map[operand]);
          }
        }
        instructions_map[inst] =
            builder.AddInstruction(HloInstruction::CreateFusion(
                inst->shape(), inst->fusion_kind(), operands,
                inst->fused_instructions_computation()));
        break;
      }
      case HloOpcode::kConstant:
        instructions_map[inst] = builder.AddInstruction(
            HloInstruction::CreateConstant(inst->literal().Clone()));
        break;
      case HloOpcode::kGetTupleElement: {
        HloGetTupleElementInstruction* get_tuple_index =
            static_cast<HloGetTupleElementInstruction*>(inst);
        HloInstruction* original_operand = get_tuple_index->mutable_operand(0);
        auto it = parameters_map.find(original_operand);
        HloInstruction* operand;
        if (it != parameters_map.end()) {
          operand = it->second;
        } else {
          operand = instructions_map[original_operand];
        }
        instructions_map[inst] =
            builder.AddInstruction(HloInstruction::CreateGetTupleElement(
                inst->shape(), operand, get_tuple_index->tuple_index()));
        break;
      }
      default:
        return InternalError("HLO opcode unsupported by command buffers");
    }
  }

  // Build result tuple.
  std::vector<HloInstruction*> new_instructions;
  absl::flat_hash_map<HloInstruction*, int64_t> inst_to_tuple_index_map;
  int64_t index = 0;
  for (HloInstruction* inst : seq.instructions()) {
    new_instructions.push_back(instructions_map[inst]);
    inst_to_tuple_index_map[inst] = index;
    index++;
  }
  builder.AddInstruction(HloInstruction::CreateTuple(new_instructions));

  BuildCommandBufferResult result = {builder.Build(), parameters_map,
                                     inst_to_tuple_index_map};
  return result;
}

StatusOr<bool> CommandBufferScheduling::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    return InternalError("module is not scheduled");
  }
  HloComputation* entry = module->entry_computation();
  MoveParametersToFront(entry);
  std::vector<HloInstructionSequence> sequences =
      CollectCommandBufferSequences(module->schedule().sequence(entry));

  for (const HloInstructionSequence& seq : sequences) {
    TF_ASSIGN_OR_RETURN(BuildCommandBufferResult result,
                        BuildCommandBuffer(seq));

    Shape shape;
    shape.set_element_type(TUPLE);
    shape.mutable_tuple_shapes()->resize(result.inst_to_tuple_index_map.size());
    for (const auto [inst, index] : result.inst_to_tuple_index_map) {
      shape.mutable_tuple_shapes()->at(index) = inst->shape();
    }

    std::vector<HloInstruction*> operands(result.parameters_map.size());
    for (const auto [inst, parameter] : result.parameters_map) {
      operands[parameter->parameter_number()] = inst;
    }

    HloComputation* command_buffer =
        module->AddComputationAndUnifyNamesAndIds(std::move(result.computation),
                                                  /*is_entry=*/false);
    HloInstruction* call_command_buffer = entry->AddInstruction(
        HloInstruction::CreateCall(shape, operands, command_buffer));

    std::vector<HloInstruction*> results(result.inst_to_tuple_index_map.size());
    for (int i = 0; i < result.inst_to_tuple_index_map.size(); i++) {
      results[i] = entry->AddInstruction(
          HloInstruction::CreateGetTupleElement(call_command_buffer, i));
    }

    for (HloInstruction* inst : seq.instructions()) {
      int64_t tuple_index = result.inst_to_tuple_index_map[inst];
      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(results[tuple_index]));
      TF_RETURN_IF_ERROR(entry->RemoveInstruction(inst));
    }
  }

  TF_RETURN_IF_ERROR(module->schedule().Update());
  return true;
}

}  // namespace xla::gpu
