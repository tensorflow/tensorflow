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
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
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

// TODO(ezhulenev): We should use debug options to get this flag.
static constexpr int kMinNumCommands = 2;

//===----------------------------------------------------------------------===//
// Pattern matching HLO instructions to commands
//===----------------------------------------------------------------------===//

using CommandBufferConfig = CommandBufferScheduling::CommandBufferConfig;

// Returns true if instruction is no-op at run time and doesn't have a
// corresponding Thunk or Command (metadata only operation).
static bool IsNoOp(const HloInstruction* inst) {
  return HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kBitcast,
                          HloOpcode::kTuple, HloOpcode::kGetTupleElement>(inst);
};

// Returns true if HLO instruction has a corresponding command buffer command.
static bool IsCommand(const HloInstruction* inst,
                      const CommandBufferConfig& config);

// Returns true if HLO computation can executed as a command buffer.
static bool IsCommand(const HloComputation* computation,
                      const CommandBufferConfig& config) {
  return absl::c_all_of(computation->instructions(),
                        [&](const HloInstruction* inst) {
                          return IsNoOp(inst) || IsCommand(inst, config);
                        });
}

// This is a template to define pattern matching functions for HLO instructions
// that do not have a corresponding class for them.
template <HloOpcode op>
static bool IsCommand(const HloInstruction*, const CommandBufferConfig& config);

// Fusions compiled to device kernels (or lowered to custom kernels) which
// always have a corresponding command buffer command.
static bool IsCommand(const HloFusionInstruction* fusion,
                      const CommandBufferConfig& config) {
  // Certain kinds of fusions have special emitters that we do not support when
  // emitting from HLO.
  auto unsupported = [](const HloInstruction* inst) {
    return inst->opcode() == HloOpcode::kDynamicUpdateSlice;
  };

  return config.contains(DebugOptions::FUSION) &&
         !absl::c_any_of(fusion->called_computation()->instructions(),
                         unsupported);
}

// While loops can be executed inside command buffers only if condition and body
// regions can be executed as command buffers.
template <>
bool IsCommand<HloOpcode::kWhile>(const HloInstruction* inst,
                                  const CommandBufferConfig& config) {
  return config.contains(DebugOptions::WHILE) &&
         IsCommand(inst->while_condition(), config) &&
         IsCommand(inst->while_body(), config);
}

static bool IsCommand(const HloInstruction* inst,
                      const CommandBufferConfig& config) {
  if (auto* fusion = DynCast<HloFusionInstruction>(inst))
    return IsCommand(fusion, config);

  if (inst->opcode() == HloOpcode::kWhile)
    return IsCommand<HloOpcode::kWhile>(inst, config);

  return false;
}

//===----------------------------------------------------------------------===//

static void RemoveTrailingNoOps(HloInstructionSequence& seq) {
  std::vector<HloInstruction*> instructions = seq.instructions();
  for (int i = instructions.size() - 1; i >= 0; i--) {
    if (HloInstruction* inst = instructions[i]; IsNoOp(inst)) {
      seq.remove_instruction(inst);
    } else {
      break;
    }
  }
}

namespace {
struct Accumulator {
  std::vector<HloInstructionSequence> sequences;
  HloInstructionSequence current_seq;
  int num_commands_in_current_seq = 0;
};
}  // namespace

// The input is a scheduled sequence of instructions. This function collects
// subsequences that will be extracted as command buffers.
std::vector<HloInstructionSequence>
CommandBufferScheduling::CollectCommandBufferSequences(
    const HloInstructionSequence inst_sequence,
    const CommandBufferConfig& config) {
  auto start_new_sequence = [](Accumulator* acc) {
    if (acc->num_commands_in_current_seq >= kMinNumCommands) {
      RemoveTrailingNoOps(acc->current_seq);
      acc->sequences.push_back(acc->current_seq);
    }
    acc->current_seq = HloInstructionSequence();
    acc->num_commands_in_current_seq = 0;
    return acc;
  };

  auto process_instruction = [&](Accumulator* acc, HloInstruction* inst) {
    if (IsCommand(inst, config)) {
      acc->current_seq.push_back(inst);
      acc->num_commands_in_current_seq++;
      return acc;
    } else if (IsNoOp(inst)) {
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
  absl::flat_hash_map<HloInstruction*, HloParameterInstruction*> captures;

  // Mapping from command buffer instructions to their clones in the command
  // buffer computation body.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> instr_mapping;

  // Maps HLO instructions in the original computation to instructions in the
  // command buffer: (a) a parameter corresponding to captured value (b) cloned
  // instruction corresponding to a command.
  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (HloInstruction* operand : instr->operands()) {
      if (auto it = captures.find(operand); it != captures.end())
        operands.push_back(it->second);
      if (auto it = instr_mapping.find(operand); it != instr_mapping.end())
        operands.push_back(it->second);
    }
    return operands;
  };

  // Create parameters in the command buffer computation for captures.
  for (HloInstruction* inst : instructions) {
    for (HloInstruction* operand : inst->operands()) {
      // We already mapped instruction to an operand.
      if (captures.contains(operand)) continue;

      // Operand instruction is a part of the command buffer.
      if (absl::c_find(instructions, operand) != instructions.end()) continue;

      int64_t index = captures.size();
      captures[operand] = Cast<HloParameterInstruction>(
          builder.AddInstruction(HloInstruction::CreateParameter(
              index, operand->shape(), absl::StrCat("p", index))));
    }
  }

  // Clone commands into the command buffer body with mapped operands.
  for (HloInstruction* inst : seq.instructions()) {
    HloCloneContext ctx(inst->GetModule());

    // Cloned instructions should call the same computations as original
    // instructions will be dead code eliminated.
    for (HloComputation* called_computation : inst->called_computations()) {
      ctx.MapComputation(called_computation, called_computation);
    }

    instr_mapping[inst] = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), mapped_operands(inst), &ctx));
  }

  // Build result tuple.
  std::vector<HloInstruction*> new_instructions;
  absl::flat_hash_map<HloInstruction*, int64_t> inst_to_tuple_index_map;
  for (HloInstruction* inst : seq.instructions()) {
    inst_to_tuple_index_map[inst] = new_instructions.size();
    new_instructions.push_back(instr_mapping[inst]);
  }
  builder.AddInstruction(HloInstruction::CreateTuple(new_instructions));

  BuildCommandBufferResult result = {builder.Build(), std::move(captures),
                                     inst_to_tuple_index_map,
                                     std::move(instr_mapping)};
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

  auto& debug_options = module->config().debug_options();

  CommandBufferConfig config;
  for (auto cmd_type : debug_options.xla_gpu_enable_command_buffer()) {
    config.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }

  std::vector<HloInstructionSequence> sequences =
      CollectCommandBufferSequences(module->schedule().sequence(entry), config);

  for (const HloInstructionSequence& seq : sequences) {
    TF_ASSIGN_OR_RETURN(BuildCommandBufferResult result,
                        BuildCommandBuffer(seq));

    Shape shape;
    shape.set_element_type(TUPLE);
    shape.mutable_tuple_shapes()->resize(result.inst_to_tuple_index_map.size());
    for (const auto [inst, index] : result.inst_to_tuple_index_map) {
      shape.mutable_tuple_shapes()->at(index) = inst->shape();
    }

    std::vector<HloInstruction*> operands(result.captures.size());
    for (const auto [inst, parameter] : result.captures) {
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

    // Remove instructions in the command buffer sequence.
    bool first_inst = true;
    for (HloInstruction* inst : seq.instructions()) {
      // Replace the first instruction in the sequence by command buffer call.
      // Removal of the rest of the instructions in the sequence is handled by
      // HloSchedule::Update().
      if (first_inst) {
        first_inst = false;
        HloInstructionSequence& sequence =
            module->schedule().GetOrCreateSequence(entry);
        sequence.replace_instruction(inst, call_command_buffer);
      }

      // Forward control dependencies to the new instruction inside command
      // buffer. If the dependent instruction is not captured by the command
      // buffer, forward the dependency to the command buffer call instead.
      HloInstruction* new_inst = result.instr_mapping[inst];
      for (HloInstruction* predecessor : inst->control_predecessors()) {
        if (auto it = result.instr_mapping.find(predecessor);
            it != result.instr_mapping.end()) {
          HloInstruction* new_predecessor = it->second;
          TF_RETURN_IF_ERROR(new_predecessor->AddControlDependencyTo(new_inst));
        } else {
          TF_RETURN_IF_ERROR(
              predecessor->AddControlDependencyTo(call_command_buffer));
        }
      }
      for (HloInstruction* successor : inst->control_successors()) {
        if (auto it = result.instr_mapping.find(successor);
            it != result.instr_mapping.end()) {
          HloInstruction* new_successor = it->second;
          TF_RETURN_IF_ERROR(new_inst->AddControlDependencyTo(new_successor));
        } else {
          TF_RETURN_IF_ERROR(
              call_command_buffer->AddControlDependencyTo(successor));
        }
      }
      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());

      int64_t tuple_index = result.inst_to_tuple_index_map[inst];
      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(results[tuple_index]));
      TF_RETURN_IF_ERROR(entry->RemoveInstruction(inst));
    }
  }

  TF_RETURN_IF_ERROR(module->schedule().Update());
  return true;
}

}  // namespace xla::gpu
