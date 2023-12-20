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
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

// TODO(ezhulenev): We should use debug options to get this flag.
static constexpr int kMinNumCommands = 2;

using CommandBuffer = CommandBufferScheduling::CommandBuffer;
using CommandBufferConfig = CommandBufferScheduling::CommandBufferConfig;

//===----------------------------------------------------------------------===//
// Pattern matching HLO instructions to commands
//===----------------------------------------------------------------------===//

static bool IsConstant(const HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kConstant;
}

static bool IsParameter(const HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kParameter;
}

// Returns true if instruction is no-op at run time and doesn't have a
// corresponding Thunk or Command (metadata only operation).
static bool IsNoOp(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                          HloOpcode::kGetTupleElement>(hlo);
};

// Returns true if HLO instruction has a corresponding command buffer command.
static bool IsCommand(const HloInstruction* hlo,
                      const CommandBufferConfig& config);

// Returns true if HLO computation can be executed as a command buffer.
static bool IsCommand(const HloComputation* computation,
                      const CommandBufferConfig& config) {
  return absl::c_all_of(computation->instructions(),
                        [&](const HloInstruction* inst) {
                          return IsNoOp(inst) || IsConstant(inst) ||
                                 IsParameter(inst) || IsCommand(inst, config);
                        });
}

// This is a template to define pattern matching functions for HLO instructions
// that do not have a corresponding class for them.
template <HloOpcode op>
static bool IsCommand(const HloInstruction*, const CommandBufferConfig&);

// Fusions compiled to device kernels (or lowered to custom kernels) which
// always have a corresponding command buffer command.
static bool IsCommand(const HloFusionInstruction* fusion,
                      const CommandBufferConfig& config) {
  // TODO(vuson): Support custom kernels as command buffer commands.
  auto backend_config = fusion->backend_config<FusionBackendConfig>();
  return config.contains(DebugOptions::FUSION) && backend_config.ok() &&
         backend_config->kind() != kCustomFusionKind;
}

// Sort operations lowered to memcpy and device kernels and we have a
// corresponding command buffer commands for them.
static bool IsCommand(const HloSortInstruction* sort,
                      const CommandBufferConfig& config) {
  return config.contains(DebugOptions::FUSION);
}

// While loops can be executed inside command buffers only if condition and body
// regions can be executed as command buffers.
template <>
bool IsCommand<HloOpcode::kWhile>(const HloInstruction* hlo,
                                  const CommandBufferConfig& config) {
  return config.contains(DebugOptions::WHILE) &&
         IsCommand(hlo->while_condition(), config) &&
         IsCommand(hlo->while_body(), config);
}

static bool IsCommand(const HloCustomCallInstruction* hlo,
                      const CommandBufferConfig& config) {
  return config.contains(DebugOptions::CUBLAS) && IsLegacyCublasMatmul(*hlo);
}

static bool IsCommand(const HloInstruction* hlo,
                      const CommandBufferConfig& config) {
  if (auto* fusion = DynCast<HloFusionInstruction>(hlo))
    return IsCommand(fusion, config);

  if (auto* sort = DynCast<HloSortInstruction>(hlo))
    return IsCommand(sort, config);

  if (auto* custom_call = DynCast<HloCustomCallInstruction>(hlo))
    return IsCommand(custom_call, config);

  if (hlo->opcode() == HloOpcode::kWhile)
    return IsCommand<HloOpcode::kWhile>(hlo, config);

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

//===----------------------------------------------------------------------===//
// Discovering sequences of compatible Hlo instructions
//===----------------------------------------------------------------------===//

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

// This function moves kParameter and kConstant instructions in a computation to
// the beginning of the computation. This simplifies the construction of command
// buffer computations because we don't need to deal with parameters and
// constants that have users outside of a command buffer.
Status CommandBufferScheduling::MoveParametersAndConstantsToFront(
    HloComputation* computation) {
  HloInstructionSequence new_sequence;
  HloSchedule& schedule = computation->parent()->schedule();
  HloInstructionSequence& sequence = schedule.GetOrCreateSequence(computation);

  for (HloInstruction* inst : sequence.instructions()) {
    if (IsParameter(inst) || IsConstant(inst)) {
      new_sequence.push_back(inst);

      // Because we move instruction to the front of the computation we can't
      // have any control predecessors, however silently dropping them is unsafe
      // as we can have transitive dependencies that define schedule order, so
      // we forward control predecessors to all users.
      for (HloInstruction* control_predecessor : inst->control_predecessors()) {
        for (HloInstruction* user : inst->users()) {
          TF_RETURN_IF_ERROR(control_predecessor->AddControlDependencyTo(user));
        }
      }
      TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
    }
  }

  for (HloInstruction* inst : sequence.instructions()) {
    if (!IsParameter(inst) && !IsConstant(inst)) {
      new_sequence.push_back(inst);
    }
  }

  schedule.set_sequence(computation, new_sequence);
  return OkStatus();
}

//===----------------------------------------------------------------------===//
// Prepares command buffer from sequence of instructions
//===----------------------------------------------------------------------===//

StatusOr<CommandBuffer> CommandBufferScheduling::PrepareCommandBuffer(
    const HloInstructionSequence& seq) {
  auto builder = HloComputation::Builder("command_buffer");

  absl::Span<HloInstruction* const> instructions =
      absl::MakeSpan(seq.instructions());

  // A set of instructions that will be moved into command buffer computation.
  absl::flat_hash_set<HloInstruction*> in_command_buffer(instructions.begin(),
                                                         instructions.end());

  // The sequence might use results of instructions that are not captured by the
  // sequence. We pass those results as parameters and map the producers of the
  // results to their corresponding parameter instructions.
  absl::flat_hash_map<HloInstruction*, HloParameterInstruction*> parameters;

  // Mapping from command buffer instructions to their clones in the command
  // buffer computation body.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> inst_mapping;

  // Maps HLO instructions in the original computation to instructions in the
  // command buffer: (a) a parameter corresponding to captured value (b) cloned
  // instruction corresponding to a command.
  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (HloInstruction* operand : instr->operands()) {
      if (auto it = inst_mapping.find(operand); it != inst_mapping.end())
        operands.push_back(it->second);
    }
    return operands;
  };

  // Create parameters in the command buffer computation for captured values.
  for (HloInstruction* inst : instructions) {
    for (HloInstruction* operand : inst->operands()) {
      // We already mapped instruction to a parameter.
      if (parameters.contains(operand)) continue;

      // Operand instruction is a part of the command buffer.
      if (in_command_buffer.contains(operand)) continue;

      // Create a new parameter for value defined outside of a command buffer.
      int64_t parameter_id = parameters.size();
      auto* parameter = Cast<HloParameterInstruction>(builder.AddInstruction(
          HloInstruction::CreateParameter(parameter_id, operand->shape(),
                                          absl::StrCat("p", parameter_id))));
      inst_mapping[operand] = parameters[operand] = parameter;
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

    inst_mapping[inst] = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), mapped_operands(inst), &ctx));
  }

  // Convert parameters to command buffer arguments.
  std::vector<HloInstruction*> arguments(parameters.size());
  for (auto& [argument, parameter] : parameters) {
    arguments[parameter->parameter_number()] = argument;
  }

  // Collect command buffer `results` (instructions replaced in the original
  // computation) and `results` (instructions in the command buffer).
  std::vector<HloInstruction*> results;
  std::vector<HloInstruction*> returned;

  auto has_external_users = [&](HloInstruction* inst) {
    return inst->IsRoot() || absl::c_any_of(inst->users(), [&](auto* user) {
             return !in_command_buffer.contains(user);
           });
  };

  for (HloInstruction* inst : instructions) {
    if (has_external_users(inst)) {
      results.push_back(inst);
      returned.push_back(inst_mapping[inst]);
    }
  }

  // If we return multiple results wrap them into tuple.
  if (returned.size() > 1) {
    builder.AddInstruction(HloInstruction::CreateTuple(returned));
  }

  return CommandBuffer{std::move(arguments), std::move(results),
                       builder.Build(), std::move(inst_mapping)};
}

//===----------------------------------------------------------------------===//
// Rewrites original computation into command buffer call
//===----------------------------------------------------------------------===//

Status CommandBufferScheduling::RewriteCommandBuffer(
    HloComputation* parent, const HloInstructionSequence& seq,
    CommandBuffer command_buffer) {
  if (command_buffer.results.empty())
    return absl::InternalError("command buffer rsults must be not empty");

  // If we have more than one result we return them as tuple, and get individual
  // values using `get-tuple-element` instructions. Otherwise we simply return
  // a result from a command buffer computation.
  Shape cmd_buffer_result_shape;
  bool has_single_result = command_buffer.results.size() == 1;

  if (has_single_result) {
    cmd_buffer_result_shape = command_buffer.results[0]->shape();
  } else {
    absl::InlinedVector<Shape, 4> shapes;
    shapes.reserve(command_buffer.results.size());
    for (auto* res : command_buffer.results) shapes.push_back(res->shape());
    cmd_buffer_result_shape = ShapeUtil::MakeTupleShape(shapes);
  }

  HloComputation* computation =
      parent->parent()->AddComputationAndUnifyNamesAndIds(
          std::move(command_buffer.computation),
          /*is_entry=*/false);

  HloInstruction* call = parent->AddInstruction(HloInstruction::CreateCall(
      cmd_buffer_result_shape, command_buffer.arguments, computation));

  // Replace all users or original results with a command buffer results.
  if (has_single_result) {
    TF_RETURN_IF_ERROR(command_buffer.results[0]->ReplaceAllUsesWith(call));
  } else {
    for (int i = 0; i < command_buffer.results.size(); i++) {
      TF_RETURN_IF_ERROR(
          command_buffer.results[i]->ReplaceAllUsesWith(parent->AddInstruction(
              HloInstruction::CreateGetTupleElement(call, i))));
    }
  }

  // As we are running after scheduling we have to keep it valid.
  HloSchedule& schedule = parent->parent()->schedule();

  // Update schedule to replace the last instruction with a command buffer call.
  // Removal of the rest of the instructions in the sequence is handled by
  // schedule update below.
  HloInstructionSequence& sequence = schedule.GetOrCreateSequence(parent);
  sequence.replace_instruction(seq.instructions().back(), call);

  // Rebuild original instruction sequence schedule in a newly created
  // command buffer computation to guarantee that we'll get exactly the same
  // buffer assignment result as if we were running without command buffers.
  HloInstructionSequence cmd_buffer_schedule;
  for (auto* argument : command_buffer.arguments) {
    cmd_buffer_schedule.push_back(command_buffer.inst_mapping[argument]);
  }
  for (auto* inst : seq.instructions()) {
    cmd_buffer_schedule.push_back(command_buffer.inst_mapping[inst]);
  }
  if (!has_single_result) {
    cmd_buffer_schedule.push_back(computation->root_instruction());
  }
  schedule.set_sequence(computation, cmd_buffer_schedule);

  // Forward control dependencies between original instructions to instruction
  // in the command buffer computation.
  auto& inst_mapping = command_buffer.inst_mapping;
  for (HloInstruction* inst : seq.instructions()) {
    HloInstruction* cmd_inst = inst_mapping[inst];

    // Forward control dependencies to the new instruction inside command
    // buffer. If the dependent instruction is not captured by the command
    // buffer, forward the dependency to the command buffer call instead.
    for (HloInstruction* predecessor : inst->control_predecessors()) {
      if (auto it = inst_mapping.find(predecessor); it != inst_mapping.end()) {
        HloInstruction* cmd_predecessor = it->second;
        TF_RETURN_IF_ERROR(cmd_predecessor->AddControlDependencyTo(cmd_inst));
      } else {
        TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(call));
      }
    }

    for (HloInstruction* successor : inst->control_successors()) {
      if (auto it = inst_mapping.find(successor); it != inst_mapping.end()) {
        HloInstruction* cmd_successor = it->second;
        TF_RETURN_IF_ERROR(cmd_inst->AddControlDependencyTo(cmd_successor));
      } else {
        TF_RETURN_IF_ERROR(call->AddControlDependencyTo(successor));
      }
    }

    TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
  }

  // Traverse in reverse order as original sequence was topologically sorted and
  // we can't remove instructions with users.
  for (int32_t i = seq.instructions().size() - 1; i >= 0; i--) {
    TF_RETURN_IF_ERROR(parent->RemoveInstruction(seq.instructions()[i]));
  }

  return OkStatus();
}

//===----------------------------------------------------------------------===//

StatusOr<bool> CommandBufferScheduling::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // We run command buffer scheduling after a regular scheduling to guarantee
  // that command buffers will not change execution order and buffer assignment
  // compared to a regular execution. Some operations (i.e. async collectives)
  // can't be captured into command buffers, and forming too large command
  // buffers too early can impact async operations scheduling.
  if (!module->has_schedule()) return InternalError("module is not scheduled");

  CommandBufferConfig config;
  for (auto cmd_type :
       module->config().debug_options().xla_gpu_enable_command_buffer()) {
    config.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }

  // TODO(b/315874495): We should traverse all computations in topological order
  // to discover command buffers inside nested control flow computations.
  HloComputation* entry = module->entry_computation();
  TF_RETURN_IF_ERROR(MoveParametersAndConstantsToFront(entry));

  std::vector<HloInstructionSequence> sequences =
      CollectCommandBufferSequences(module->schedule().sequence(entry), config);

  for (const HloInstructionSequence& seq : sequences) {
    TF_ASSIGN_OR_RETURN(CommandBuffer command_buffer,
                        PrepareCommandBuffer(seq));
    TF_RETURN_IF_ERROR(
        RewriteCommandBuffer(entry, seq, std::move(command_buffer)));
  }

  TF_RETURN_IF_ERROR(module->schedule().Update());

  return true;
}

}  // namespace xla::gpu
