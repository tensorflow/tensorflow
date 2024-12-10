/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/command_buffer_scheduling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/variant_visitor.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

using CommandBuffer = CommandBufferScheduling::CommandBuffer;
using CommandBufferConfig = CommandBufferScheduling::CommandBufferConfig;

// Returns true if HLO computation can be executed as a command buffer.
static bool IsCommand(const HloComputation* computation,
                      const CommandBufferConfig& config);

//===----------------------------------------------------------------------===//
// No-op HLO operations.
//===----------------------------------------------------------------------===//

// Some of the HLO operations do not have corresponding operations at run time
// and they can be safely wrapped into command buffers together with load
// bearing commands.

static bool IsConstant(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kConstant>(hlo);
}

static bool IsParameter(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kParameter>(hlo);
}

// Returns true if instruction is no-op at run time and doesn't have a
// corresponding Thunk or Command (metadata only operation).
static bool IsNoOp(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                          HloOpcode::kGetTupleElement>(hlo);
};

//===----------------------------------------------------------------------===//
// Asynchronous HLO operations mapped to commands.
//===----------------------------------------------------------------------===//

// Asynchronous HLO operations can be wrapped into command buffers only when
// both start and done operations can be put into the same command buffer.
// Command buffer semantics implies that when command buffer execution
// completes, all recorded commands are also completed, which means that if
// done operation is not part of the same command buffer, we would change the
// execution semantics and create additional synchronization point.

static bool IsAsyncStartCommand(const HloInstruction* hlo,
                                const CommandBufferConfig& config) {
  if (HloPredicateIsOp<HloOpcode::kAllReduceStart, HloOpcode::kAllGatherStart>(
          hlo)) {
    return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
  }

  if (HloPredicateIsOp<HloOpcode::kAsyncStart>(hlo)) {
    if (IsCublasGemm(*hlo->async_wrapped_instruction())) {
      return config.enabled_commands.contains(DebugOptions::CUBLAS);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kFusion) {
      return config.enabled_commands.contains(DebugOptions::FUSION);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kReduceScatter ||
        hlo->async_wrapped_opcode() == HloOpcode::kAllToAll) {
      return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
    }
  }

  if (HloPredicateIsOp<HloOpcode::kReduceScatter, HloOpcode::kAllToAll>(hlo)) {
    return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
  }

  return false;
}

static bool IsAsyncDoneCommand(const HloInstruction* hlo,
                               const CommandBufferConfig& config) {
  if (HloPredicateIsOp<HloOpcode::kAllReduceDone, HloOpcode::kAllGatherDone>(
          hlo)) {
    return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
  }

  if (HloPredicateIsOp<HloOpcode::kAsyncDone>(hlo)) {
    if (IsCublasGemm(*hlo->async_wrapped_instruction())) {
      return config.enabled_commands.contains(DebugOptions::CUBLAS);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kFusion) {
      return config.enabled_commands.contains(DebugOptions::FUSION);
    }
    if (hlo->async_wrapped_opcode() == HloOpcode::kReduceScatter ||
        hlo->async_wrapped_opcode() == HloOpcode::kAllToAll) {
      return config.enabled_commands.contains(DebugOptions::COLLECTIVES);
    }
  }

  return false;
}

// Finds an async-done HLO operation corresponding on an async-start one.
static HloInstruction* FindAsyncDoneCommand(const HloInstruction* start) {
  if (HloPredicateIsOp<HloOpcode::kAllReduceStart, HloOpcode::kAllGatherStart>(
          start)) {
    CHECK(start->users().size() == 1);  // NOLINT, checked by HLO verifier
    return start->users().front();
  } else if (HloPredicateIsOp<HloOpcode::kAsyncStart>(start)) {
    return start->async_chain_done();
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Synchronous HLO operations mapped to commands.
//===----------------------------------------------------------------------===//

// Synchronous HLO operations can be wrapped into command buffers when they have
// a corresponding commands.

// This is a template to define pattern matching functions for HLO instructions
// that do not have a corresponding class for them.
template <HloOpcode op>
static bool IsCommand(const HloInstruction*, const CommandBufferConfig&);

// While loops can be executed inside command buffers only if condition and body
// regions can be executed as command buffers.
template <>
bool IsCommand<HloOpcode::kWhile>(const HloInstruction* hlo,
                                  const CommandBufferConfig& config) {
  return config.enabled_commands.contains(DebugOptions::WHILE) &&
         IsCommand(hlo->while_body(), config) &&
         IsCommand(hlo->while_condition(), config);
}

// Conditional can be executed inside command buffers only if all regions of its
// branches can be executed as command buffers.
template <>
bool IsCommand<HloOpcode::kConditional>(const HloInstruction* hlo,
                                        const CommandBufferConfig& config) {
  return config.enabled_commands.contains(DebugOptions::CONDITIONAL) &&
         absl::c_all_of(hlo->branch_computations(),
                        [&](const HloComputation* comp) {
                          return IsCommand(comp, config);
                        });
}

static bool IsCommand(const HloCustomCallInstruction* hlo,
                      const CommandBufferConfig& config) {
  // cuBLAS gemms represented in the HLO as custom call instructions.
  if (config.enabled_commands.contains(DebugOptions::CUBLAS) &&
      IsLegacyCublasMatmul(*hlo)) {
    return true;
  }

  if (config.enabled_commands.contains(DebugOptions::CUBLASLT) &&
      (IsCublasLtMatmul(*hlo) || IsCublasLtMatmulF8(*hlo))) {
    return true;
  }

  if (config.enabled_commands.contains(DebugOptions::CUDNN) &&
      IsCustomCallTofMHA(*hlo)) {
    VLOG(3) << "Recording FusedMHA, target " << hlo->custom_call_target()
            << " into command buffer.";
    return true;
  }

  if (!config.enabled_commands.contains(DebugOptions::CUSTOM_CALL)) {
    return false;
  }

  if (config.enabled_legacy_custom_call_targets.contains(
          hlo->custom_call_target())) {
    VLOG(3) << "Recording legacy custom call target "
            << hlo->custom_call_target() << " into command buffer.";
    return true;
  }

  // Check if FFI handler is compatible with command buffers.
  auto registration = ffi::FindHandler(hlo->custom_call_target(), "gpu");
  return registration.ok()
             ? ffi::IsCommandBufferCompatible(registration->traits)
             : false;
}

static bool IsCommand(const HloInstruction* hlo,
                      const CommandBufferConfig& config) {
  if (auto* fusion = DynCast<HloFusionInstruction>(hlo)) {
    auto gpu_config = fusion->backend_config<GpuBackendConfig>();
    const FusionBackendConfig& backend_config =
        gpu_config->fusion_backend_config();
    if (backend_config.kind() == kCuDnnFusionKind) {
      return config.enabled_commands.contains(DebugOptions::CUDNN);
    }
    const auto& custom_config = backend_config.custom_fusion_config();
    if ((custom_config.name() == "address_computation") ||
        (custom_config.name() == "dynamic_address_computation")) {
      auto fusion_analysis =
          HloFusionAnalysis::Create(*hlo, config.device_description);
      const HloFusionAdaptor& adaptor = fusion_analysis.fusion();
      auto hero_adaptor =
          HloBfsFindIf(adaptor.GetRoots(), adaptor, [](auto node) {
            return node.opcode() == HloOpcode::kCustomCall ||
                   node.opcode() == HloOpcode::kReduceScatter;
          });
      const HloInstruction* hero = &hero_adaptor->instruction();

      if (custom_config.name() == "address_computation") {
        return IsCommand(hero, config) || IsAsyncStartCommand(hero, config);
      } else {
        // DynamicSliceFusionRewriter currently only rewrites for dynamic slice
        // fusion with constant or loop iteration offset values, which are all
        // supported by command buffer.
        return (config.enabled_commands.contains(
                    DebugOptions::DYNAMIC_SLICE_FUSION) &&
                (IsCommand(hero, config) || IsAsyncStartCommand(hero, config)));
      }
    }

    // Cuda has a bug that when the cuda kernel's parameter size is larger than
    // 4KB, then cudaGraphAddKernelNode will have segment fault, we disable the
    // command buffer lowering for kernels which has over 512 parameters.
    // TODO(shawnw): remove this when cuda driver has release a fix.
    int64_t total_args = fusion->operands().size();
    ShapeUtil::ForEachLeafShape(
        fusion->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsArray()) {
            total_args++;
          }
        });
    if (total_args > 512) {
      // shared memory allocation needs a pointer inside kernel's packed
      // arguments, and PackKernelArgs will round to 1024 args for the kernel
      // has arg count between 512 and 1024.
      VLOG(2) << "disable fusion kernel due to large argument count (>512)";
      return false;
    }

    return config.enabled_commands.contains(DebugOptions::FUSION);
  }

  if (auto* sort = DynCast<HloSortInstruction>(hlo))
    return config.enabled_commands.contains(DebugOptions::FUSION);

  if (HloPredicateIsOp<HloOpcode::kCopy>(hlo))
    return config.enabled_commands.contains(DebugOptions::FUSION);

  if (HloPredicateIsOp<HloOpcode::kPartitionId, HloOpcode::kReplicaId>(hlo)) {
    return config.enabled_commands.contains(DebugOptions::FUSION);
  }

  if (auto* custom_call = DynCast<HloCustomCallInstruction>(hlo))
    return IsCommand(custom_call, config);

  if (HloPredicateIsOp<HloOpcode::kWhile>(hlo))
    return IsCommand<HloOpcode::kWhile>(hlo, config);

  if (HloPredicateIsOp<HloOpcode::kConditional>(hlo))
    return IsCommand<HloOpcode::kConditional>(hlo, config);

  return false;
}

//===----------------------------------------------------------------------===//
// HLO computations mapped to command buffers.
//===----------------------------------------------------------------------===//

// Returns true if HLO computation can be executed as a command buffer.
static bool IsCommand(const HloComputation* computation,
                      const CommandBufferConfig& config) {
  return absl::c_all_of(
      computation->instructions(), [&](const HloInstruction* inst) {
        return IsNoOp(inst) || IsConstant(inst) || IsParameter(inst) ||
               IsCommand(inst, config) || IsAsyncStartCommand(inst, config) ||
               IsAsyncDoneCommand(inst, config);
      });
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

// The input is a scheduled sequence of instructions. This function collects
// subsequences that will be extracted as command buffers.
std::vector<HloInstructionSequence>
CommandBufferScheduling::CollectCommandBufferSequences(
    const HloInstructionSequence schedule, const CommandBufferConfig& config,
    int32_t min_num_commands) {
  std::vector<HloInstructionSequence> sequences;

  HloInstructionSequence current_seq;
  int64_t num_commands_in_current_seq = 0;

  // Adds `current_seq` to `sequences` if it has enough commands in it.
  auto collect_current_seq = [&]() {
    if (num_commands_in_current_seq >= std::max(1, min_num_commands)) {
      RemoveTrailingNoOps(current_seq);
      sequences.push_back(std::move(current_seq));
    }
    current_seq = HloInstructionSequence();
    num_commands_in_current_seq = 0;
  };

  auto& instructions = schedule.instructions();

  // we currently require that when lowering DynamicSliceFusion, the offset
  // value should not come from the output of operators that are already
  // captured in command buffer.
  auto check_dynamic_slice_operand_not_from_seq =
      [&](const HloInstructionSequence& seq, const HloInstruction* inst) {
        if (!config.enabled_commands.contains(
                DebugOptions::DYNAMIC_SLICE_FUSION))
          return true;
        const auto* fusion = DynCast<HloFusionInstruction>(inst);
        if (!fusion) return true;

        auto gpu_config = fusion->backend_config<GpuBackendConfig>();
        const FusionBackendConfig& backend_config =
            gpu_config->fusion_backend_config();
        const auto& custom_config = backend_config.custom_fusion_config();
        if (custom_config.name() != "dynamic_address_computation") return true;

        auto* fused_computation = fusion->called_computation();
        return !absl::c_any_of(
            fused_computation->instructions(), [&](const HloInstruction* inst) {
              const auto* dynamic_inst =
                  DynCast<HloDynamicIndexInstruction>(inst);
              if (!dynamic_inst) return false;
              for (auto* operand : dynamic_inst->index_operands()) {
                const auto* param = DynCast<HloParameterInstruction>(operand);
                const auto* fusion_operand =
                    fusion->operand(param->parameter_number());
                if (seq.contains(fusion_operand)) {
                  return true;
                }
              }
              return false;
            });
      };

  // Collect the sequence of instructions that contains the async start and its
  // corresponding done instruction. If there is another start instruction
  // between the original start and done, we may potentially extend the sequence
  // to include its corresponding done instruction. For example, if we call this
  // function on async-start_a in the following sequence:
  //
  // async_start_a
  // async_start_b
  // async_done_a
  // async_done_b
  //
  // The returned sequence will contain async_done_b. So that all async pairs
  // are captured by the same command buffer.
  auto collect_async_region = [&](const HloInstruction* start) {
    auto get_index = [&](const HloInstruction* inst) -> size_t {
      auto it = std::find(instructions.begin(), instructions.end(), inst);
      return std::distance(instructions.begin(), it);
    };

    HloInstructionSequence seq;
    size_t done_index = get_index(FindAsyncDoneCommand(start));
    for (size_t i = get_index(start); i <= done_index; i++) {
      HloInstruction* inst = instructions.at(i);
      if (IsAsyncStartCommand(inst, config)) {
        const HloInstruction* done = FindAsyncDoneCommand(inst);
        done_index = std::max(done_index, get_index(done));
      }
      seq.push_back(inst);
    }
    return seq;
  };

  // Check that instructions are safe to be captured by command buffer, and that
  // we do not capture unmatched async done instruction.
  auto check_async_region = [&](const HloInstructionSequence& seq) {
    if (!absl::c_all_of(seq.instructions(), [&](HloInstruction* inst) {
          return IsNoOp(inst) ||
                 (IsCommand(inst, config) &&
                  check_dynamic_slice_operand_not_from_seq(seq, inst) &&
                  check_dynamic_slice_operand_not_from_seq(current_seq,
                                                           inst)) ||
                 IsAsyncStartCommand(inst, config) ||
                 IsAsyncDoneCommand(inst, config);
        })) {
      return false;
    }

    absl::flat_hash_set<HloInstruction*> done_instructions;
    for (const HloInstruction* inst : seq.instructions()) {
      if (IsAsyncStartCommand(inst, config)) {
        done_instructions.insert(FindAsyncDoneCommand(inst));
      }
      if (IsAsyncDoneCommand(inst, config)) {
        if (!done_instructions.contains(inst)) {
          return false;
        }
      }
    }
    return true;
  };

  for (size_t i = 0; i < instructions.size(); i++) {
    HloInstruction* inst = instructions.at(i);

    // We add no-op instructions to current sequence only if they act as a glue
    // between commands. We do not create command sequences consisting only from
    // no-op instruction. First and last instruction in the command buffer is
    // always a load-bearing command.
    if (IsNoOp(inst) && num_commands_in_current_seq) {
      current_seq.push_back(inst);
      continue;
    }

    // Synchronous commands always can be added to instruction sequence.
    if (IsCommand(inst, config) &&
        check_dynamic_slice_operand_not_from_seq(current_seq, inst)) {
      num_commands_in_current_seq++;
      current_seq.push_back(inst);
      continue;
    }

    // We capture async commands if all instruction between start and done can
    // be outlined into a command buffer.
    if (IsAsyncStartCommand(inst, config)) {
      HloInstructionSequence seq = collect_async_region(inst);
      if (check_async_region(seq)) {
        num_commands_in_current_seq += seq.instructions().size();
        for (HloInstruction* inst : seq.instructions()) {
          current_seq.push_back(inst);
        }
        i += seq.instructions().size() - 1;
        continue;
      }
    }

    // If we didn't find the next command, collect the current sequence and
    // start a new one.
    collect_current_seq();
  }

  // Don't forget to collect the final command sequence.
  collect_current_seq();
  return sequences;
}

// This function moves kParameter and kConstant instructions in a computation to
// the beginning of the computation. This simplifies the construction of command
// buffer computations because we don't need to deal with parameters and
// constants that have users outside of a command buffer.
// Returns true if there is a change in the order of instructions, false
// otherwise.
absl::StatusOr<bool> CommandBufferScheduling::MoveParametersAndConstantsToFront(
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
  for (auto [old_i, new_i] :
       llvm::zip(sequence.instructions(), new_sequence.instructions())) {
    if (old_i != new_i) return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Prepares command buffer from sequence of instructions
//===----------------------------------------------------------------------===//

absl::StatusOr<CommandBuffer> CommandBufferScheduling::PrepareCommandBuffer(
    const HloInstructionSequence& seq, HloModule* module) {
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
      auto* parameter = Cast<HloParameterInstruction>(
          builder.AddInstruction(HloInstruction::CreateParameter(
              parameter_id, operand->shape(), "p")));

      parameter->UniquifyName(module);
      parameter->UniquifyId(module);
      inst_mapping[operand] = parameters[operand] = parameter;
    }
  }

  // Clone commands into the command buffer body with mapped operands.
  for (HloInstruction* inst : seq.instructions()) {
    HloCloneContext ctx(inst->GetModule());

    // Cloned instructions should call the same computations as original
    // instructions will be dead code eliminated.
    for (HloComputation* called_computation : inst->called_computations()) {
      // Async computations can only be referenced by a single async chain at
      // a time. Detach the current chain to let its copy bind to the
      // computation.
      if (called_computation->IsAsyncComputation()) {
        called_computation->RemoveAsyncStart();
      }
      ctx.MapComputation(called_computation, called_computation);
    }

    inst_mapping[inst] = builder.AddInstruction(
        inst->CloneWithNewOperands(inst->shape(), mapped_operands(inst), &ctx));
    inst_mapping[inst]->UniquifyId(module);
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
    HloInstruction* inst =
        builder.AddInstruction(HloInstruction::CreateTuple(returned));
    inst->UniquifyName(module);
    inst->UniquifyId(module);
  }

  std::unique_ptr<HloComputation> comp = builder.Build();
  comp->UniquifyName(module);
  comp->SetUniqueId(comp->root_instruction()->unique_id());

  return CommandBuffer{std::move(arguments), std::move(results),
                       std::move(comp), std::move(inst_mapping)};
}

//===----------------------------------------------------------------------===//
// Rewrites original computation into command buffer call
//===----------------------------------------------------------------------===//

absl::StatusOr<HloComputation*> CommandBufferScheduling::RewriteCommandBuffer(
    HloComputation* parent, const HloInstructionSequence& seq,
    CommandBuffer command_buffer) {
  if (command_buffer.results.empty())
    return absl::InternalError("command buffer results must not be empty");

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
      parent->parent()->AddComputation(std::move(command_buffer.computation),
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
        // If predecessor mapped to a parameter instruction it means that we
        // need to forward control dependency to a call operation, otherwise
        // we add control dependency between commands in the command buffer.
        HloInstruction* cmd_predecessor = it->second;
        if (IsParameter(cmd_predecessor)) {
          TF_RETURN_IF_ERROR(predecessor->AddControlDependencyTo(call));
        } else {
          TF_RETURN_IF_ERROR(cmd_predecessor->AddControlDependencyTo(cmd_inst));
        }
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

  return computation;
}

//===----------------------------------------------------------------------===//

CommandBufferScheduling::CommandBufferScheduling(
    const se::DeviceDescription& device_description)
    : device_description_(device_description) {}

absl::StatusOr<bool> CommandBufferScheduling::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // We run command buffer scheduling after a regular scheduling to guarantee
  // that command buffers will not change execution order and buffer assignment
  // compared to a regular execution. Some operations (i.e. async collectives)
  // can't be captured into command buffers, and forming too large command
  // buffers too early can impact async operations scheduling.
  if (!module->has_schedule()) return Internal("module is not scheduled");

  const DebugOptions& debug_options = module->config().debug_options();

  absl::flat_hash_set<DebugOptions::CommandBufferCmdType> commands;
  for (auto cmd_type : debug_options.xla_gpu_enable_command_buffer()) {
    commands.insert(static_cast<DebugOptions::CommandBufferCmdType>(cmd_type));
  }

  absl::flat_hash_set<std::string> legacy_custom_call_targets;
  for (const auto& target :
       debug_options.legacy_command_buffer_custom_call_targets()) {
    legacy_custom_call_targets.insert(target);
  }

  CommandBufferConfig config{std::move(commands),
                             std::move(legacy_custom_call_targets),
                             device_description_};

  // Erase command buffer cmd types that are not supported by the gpu runtime.
  static constexpr auto kRequireConditionals = {DebugOptions::CONDITIONAL,
                                                DebugOptions::WHILE};
  static constexpr auto kRequireTracing = {
      DebugOptions::CUBLAS, DebugOptions::CUBLASLT, DebugOptions::CUDNN,
      DebugOptions::CUSTOM_CALL, DebugOptions::COLLECTIVES};

  auto erase = [&](absl::Span<const DebugOptions::CommandBufferCmdType> cmds) {
    for (auto cmd : cmds) {
      if (config.enabled_commands.erase(cmd)) {
        VLOG(1) << "Removed command buffer support for "
                << DebugOptions::CommandBufferCmdType_Name(cmd)
                << " as it's not supported with gpu toolkit version "
                << device_description_.runtime_version()
                << " and driver version "
                << device_description_.driver_version()
                << ". This might negatively impact peformance. To enable "
                << DebugOptions::CommandBufferCmdType_Name(cmd)
                << " support in command buffers use cuda-compat package: "
#if defined(PLATFORM_GOOGLE)
                << "set CUDA_COMPAT_LOAD=1 env variable.";
#else
                << "https://docs.nvidia.com/deploy/cuda-compatibility/.";
#endif
      }
    }
  };

  // Check if CUDA/ROCM driver supports required features.
  auto erase_cuda = [&](const se::CudaComputeCapability& cuda_comp) {
    if (std::min(device_description_.runtime_version(),
                 device_description_.driver_version()) <
        se::SemanticVersion{12, 3, 0}) {
      erase(kRequireTracing);       // cuStreamBeginCaptureToGraph
      erase(kRequireConditionals);  // on-device control flow
    }
  };
  auto erase_rocm = [&](const se::RocmComputeCapability& rocm_comp) {
    erase(kRequireConditionals);  // on-device control flow
  };

  std::visit(VariantVisitor{erase_cuda, erase_rocm},
             device_description_.gpu_compute_capability());

  auto order = module->MakeComputationPostOrder();
  std::reverse(order.begin(), order.end());
  absl::flat_hash_set<HloComputation*> processed_command_buffers;

  auto changed = false;
  for (HloComputation* comp : order) {
    // Skip special computations that do not have lowering to thunks.
    if (comp->IsFusionComputation() || comp->IsAsyncComputation() ||
        comp->IsCustomCallComputation())
      continue;

    // Skip computations that already part of command buffers.
    if (processed_command_buffers.contains(comp)) continue;

    TF_ASSIGN_OR_RETURN(bool changed_, MoveParametersAndConstantsToFront(comp));
    changed |= changed_;

    std::vector<HloInstructionSequence> sequences =
        CollectCommandBufferSequences(
            module->schedule().sequence(comp), config,
            debug_options.xla_gpu_graph_min_graph_size());

    for (const HloInstructionSequence& seq : sequences) {
      TF_ASSIGN_OR_RETURN(CommandBuffer command_buffer,
                          PrepareCommandBuffer(seq, comp->parent()));
      TF_ASSIGN_OR_RETURN(
          HloComputation * command_buffer_computation,
          RewriteCommandBuffer(comp, seq, std::move(command_buffer)));
      changed = true;

      // All computations reachable from a command buffer computation are nested
      // command buffers (i.e. body computations attached to a while operation).
      for (HloComputation* called :
           command_buffer_computation->MakeEmbeddedComputationsList()) {
        processed_command_buffers.insert(called);
      }
    }
  }
  TF_RETURN_IF_ERROR(module->schedule().Update());

  return changed;
}

}  // namespace xla::gpu
