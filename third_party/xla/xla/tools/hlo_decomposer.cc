/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/tools/hlo_decomposer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/hlo_module_config.h"

namespace xla {
namespace {

constexpr absl::string_view kSparseCoreThread = "sparsecore";

// Appends the leaf elements of `instruction` to `elements`, decomposing any
// tuple-shaped instruction via GetTupleElement instructions so that a flat
// tuple can be constructed without nesting tuples.
void AppendFlattenedTupleElements(HloInstruction* instruction,
                                  HloComputation::Builder& builder,
                                  std::vector<HloInstruction*>& elements) {
  if (!instruction->shape().IsTuple()) {
    elements.push_back(instruction);
    return;
  }
  for (int64_t i = 0; i < instruction->shape().tuple_shapes().size(); ++i) {
    HloInstruction* gte = builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(instruction, i));
    AppendFlattenedTupleElements(gte, builder, elements);
  }
}

void AddFlatTupleRoot(const std::vector<HloInstruction*>& instructions,
                      HloComputation::Builder& builder) {
  std::vector<HloInstruction*> tuple_elements;
  for (HloInstruction* instruction : instructions) {
    AppendFlattenedTupleElements(instruction, builder, tuple_elements);
  }
  builder.AddInstruction(HloInstruction::CreateTuple(tuple_elements));
}

// Returns whether it makes sense to run the given instruction in isolation
// (e.g. whether it can run without dependent instructions).
bool ShouldIsolateOpcode(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
      return false;
    default:
      return true;
  }
}

// Walks forward from `async_start` along direct `operand(0)` async links within
// `async_start.parent()` and returns `[async-start, async-update...,
// async-done]`.
absl::StatusOr<std::vector<const HloInstruction*>>
FindSparseCoreOffloadChainFromStart(const HloInstruction& async_start) {
  const HloComputation* computation = async_start.parent();
  std::vector<const HloInstruction*> chain;
  const HloInstruction* current = &async_start;
  while (current != nullptr) {
    chain.push_back(current);
    if (chain.size() > computation->instruction_count()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cycle in async chain starting at ", async_start.name()));
    }
    if (current->opcode() == HloOpcode::kAsyncDone) {
      return chain;
    }
    const HloInstruction* next = nullptr;
    for (const HloInstruction* user : current->users()) {
      if (user->parent() != computation || user->operands().empty() ||
          user->operand(0) != current) {
        continue;
      }
      if (user->opcode() == HloOpcode::kAsyncUpdate ||
          user->opcode() == HloOpcode::kAsyncDone) {
        if (next != nullptr) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Branching async chain starting at ", async_start.name()));
        }
        next = user;
      }
    }
    current = next;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Incomplete or loop crossing async chain starting at ",
                   async_start.name()));
}

// Returns `[async-start, async-update..., async-done]` when `async_done` ends a
// complete main thread "sparsecore" direct operand async chain.
absl::StatusOr<std::vector<const HloInstruction*>> FindSparseCoreOffloadChain(
    const HloInstruction& async_done) {
  if (async_done.opcode() != HloOpcode::kAsyncDone) {
    return absl::InvalidArgumentError(
        absl::StrCat(async_done.name(), " is not an async-done instruction."));
  }
  const HloComputation* computation = async_done.parent();
  if (computation == nullptr ||
      computation->execution_thread() != HloInstruction::kMainExecutionThread) {
    return absl::InvalidArgumentError(absl::StrCat(
        async_done.name(), " is not in a main thread computation."));
  }

  const HloInstruction* current = &async_done;
  for (int64_t steps = 0; current->opcode() != HloOpcode::kAsyncStart;
       ++steps) {
    if (steps > computation->instruction_count()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cycle in async chain at ", async_done.name()));
    }
    if ((current != &async_done &&
         current->opcode() != HloOpcode::kAsyncUpdate) ||
        current->operands().empty() ||
        current->operand(0)->parent() != computation) {
      return absl::InvalidArgumentError(
          absl::StrCat("Incomplete async chain at ", async_done.name()));
    }
    current = current->operand(0);
  }
  if (current->async_execution_thread() != kSparseCoreThread) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Async chain at ", async_done.name(), " is not on sparsecore thread."));
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<const HloInstruction*> chain,
                   FindSparseCoreOffloadChainFromStart(*current));
  if (chain.back() != &async_done) {
    return absl::InvalidArgumentError(
        absl::StrCat("Async chain from ", current->name(),
                     " does not uniquely end at ", async_done.name()));
  }
  return chain;
}

// Restores an embedded program `custom-call` wrapper on
// `async_wrapped_computation` back to a standard `call` instruction so
// reference evaluators can execute it.
absl::Status RestoreSparseCoreProgramCall(HloInstruction* async_done) {
  HloComputation* wrapped =
      Cast<HloAsyncInstruction>(async_done)->async_wrapped_computation();
  if (wrapped == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat(async_done->name(), " was extracted without its chain."));
  }
  HloInstruction* caller = wrapped->root_instruction();
  if (caller->opcode() != HloOpcode::kCustomCall ||
      !absl::StartsWith(caller->custom_call_target(), "TODO(")) {
    return absl::OkStatus();
  }
  if (caller->called_computations().size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat(caller->name(), " does not call one SparseCore program."));
  }
  HloInstruction* call = wrapped->AddInstruction(HloInstruction::CreateCall(
      caller->shape(), caller->operands(), caller->to_apply()));
  call->set_raw_backend_config_string(caller->raw_backend_config_string());
  call->set_metadata(caller->metadata());
  wrapped->set_root_instruction(call);
  return wrapped->RemoveInstruction(caller);
}

absl::StatusOr<std::unique_ptr<HloModule>> ExtractSparseCoreOffloadChain(
    const std::vector<const HloInstruction*>& chain) {
  const HloInstruction& async_done = *chain.back();
  const HloModule* source_module = async_done.GetModule();
  HloModuleConfig config;
  config.set_debug_options(source_module->config().debug_options());
  auto new_hlo_module = std::make_unique<HloModule>(
      std::string(async_done.name()), config,
      std::make_unique<CompilationEnvironments>(source_module->comp_envs()));

  HloComputation::Builder builder("entry_computation");
  HloCloneContext clone_context(new_hlo_module.get());
  absl::flat_hash_set<const HloInstruction*> chain_set(chain.begin(),
                                                       chain.end());
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> param_map;
  int64_t parameter_number = 0;
  for (const HloInstruction* inst : chain) {
    for (const HloInstruction* operand : inst->operands()) {
      if (chain_set.contains(operand)) {
        continue;
      }
      auto [param_it, inserted] = param_map.try_emplace(operand, nullptr);
      if (inserted) {
        param_it->second =
            builder.AddInstruction(HloInstruction::CreateParameter(
                parameter_number++, operand->shape(), operand->name()));
      }
    }
  }

  absl::flat_hash_map<const HloInstruction*, HloInstruction*> chain_map;
  for (const HloInstruction* inst : chain) {
    std::vector<HloInstruction*> mapped_operands;
    mapped_operands.reserve(inst->operand_count());
    for (const HloInstruction* operand : inst->operands()) {
      if (auto it = chain_map.find(operand); it != chain_map.end()) {
        mapped_operands.push_back(it->second);
      } else {
        mapped_operands.push_back(param_map.at(operand));
      }
    }
    HloInstruction* cloned = builder.AddInstruction(inst->CloneWithNewOperands(
        inst->shape(), mapped_operands, &clone_context));
    chain_map[inst] = cloned;
  }

  HloComputation* entry_computation =
      new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
  entry_computation->SetExecutionThread(
      async_done.parent()->execution_thread());

  ABSL_RETURN_IF_ERROR(
      RestoreSparseCoreProgramCall(entry_computation->root_instruction()));
  if (source_module->has_schedule()) {
    ABSL_RETURN_IF_ERROR(ScheduleIsolatedModule(new_hlo_module.get()));
  }
  return new_hlo_module;
}

absl::StatusOr<std::vector<std::unique_ptr<HloModule>>> Decompose(
    const HloModule& module) {
  std::vector<std::unique_ptr<HloModule>> modules;

  std::vector<const HloComputation*> computations_to_visit{
      module.entry_computation()};
  absl::flat_hash_set<const HloComputation*> visited_computations{
      module.entry_computation()};

  // Traverse the computation tree, starting from the entry computation, and
  // recursing into the called computations.
  for (size_t visit_idx = 0; visit_idx < computations_to_visit.size();
       ++visit_idx) {
    const HloComputation* computation = computations_to_visit[visit_idx];

    for (const HloInstruction* instruction : computation->instructions()) {
      // Skip called computations in the embedded context (fusion, reduce, map,
      // etc), as within these computations instructions are not lowered
      // individually and it doesn't make sense to test them in isolation.
      if (GetInstructionCallContext(instruction->opcode()) !=
          CallContext::kEmbedded) {
        for (const HloComputation* called_computation :
             instruction->called_computations()) {
          if (called_computation->execution_thread() == kSparseCoreThread) {
            continue;
          }
          if (visited_computations.insert(called_computation).second) {
            computations_to_visit.push_back(called_computation);
          }
        }
      }
      if (computation->execution_thread() ==
          HloInstruction::kMainExecutionThread) {
        if (instruction->opcode() == HloOpcode::kAsyncStart &&
            instruction->async_execution_thread() == kSparseCoreThread) {
          if (absl::Status status =
                  FindSparseCoreOffloadChainFromStart(*instruction).status();
              !status.ok()) {
            LOG(WARNING) << "Skipping SparseCore offload: " << status;
          }
          continue;
        }
        if (instruction->opcode() == HloOpcode::kAsyncDone) {
          absl::StatusOr<std::vector<const HloInstruction*>> chain =
              FindSparseCoreOffloadChain(*instruction);
          if (chain.ok()) {
            absl::StatusOr<std::unique_ptr<HloModule>> offload_module =
                ExtractSparseCoreOffloadChain(*chain);
            if (!offload_module.ok()) {
              LOG(WARNING) << "Skipping SparseCore offload "
                           << instruction->name() << ": "
                           << offload_module.status();
              continue;
            }
            modules.push_back(*std::move(offload_module));
          }
          continue;
        }
      }
      if (ShouldIsolateOpcode(instruction->opcode())) {
        modules.push_back(ExtractInstructionIntoNewModule(*instruction));
      }
    }
  }

  return modules;
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<HloModule>>> DecomposeHloModule(
    const HloModule& module, bool deduplicate_modules) {
  std::vector<std::unique_ptr<HloModule>> modules;
  absl::flat_hash_set<std::string> module_fingerprints;

  auto should_add_module = [&](const HloModule* module) {
    if (!deduplicate_modules) {
      return true;
    }
    HloPrintOptions print_options = HloPrintOptions::ModuleFingerprint();
    if (IsSparseCoreOffloadUnit(*module)) {
      print_options.set_print_backend_config(true);
    }
    const std::string fingerprint = module->GetFingerprint128(print_options);
    if (!module_fingerprints.insert(fingerprint).second) {
      return false;
    }
    return true;
  };

  ABSL_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<HloModule>> isolated_modules,
                   Decompose(module));
  for (auto& module : isolated_modules) {
    if (should_add_module(module.get())) {
      modules.push_back(std::move(module));
    }
  }
  return modules;
}

bool IsSparseCoreOffloadUnit(const HloModule& module) {
  const HloInstruction* root = module.entry_computation()->root_instruction();
  return root->opcode() == HloOpcode::kAsyncDone &&
         root->async_execution_thread() == kSparseCoreThread;
}

absl::Status ScheduleIsolatedModule(HloModule* module) {
  ABSL_RETURN_IF_ERROR(HloTrivialScheduler().Run(module).status());
  if (IsSparseCoreOffloadUnit(*module)) {
    HloComputation* entry = module->entry_computation();
    std::vector<HloInstruction*> sequence =
        module->schedule().sequence(entry).instructions();
    absl::c_stable_partition(sequence, [](const HloInstruction* instruction) {
      return instruction->opcode() == HloOpcode::kParameter;
    });
    module->schedule().set_sequence(entry, std::move(sequence));
  }
  return absl::OkStatus();
}

std::unique_ptr<HloModule> ExtractCollectiveOperationsIntoNewModule(
    const std::vector<HloInstruction*>& instructions,
    const absl::flat_hash_set<HloOpcode>& done_ops,
    const absl::flat_hash_set<HloOpcode>& non_optimized_ops,
    bool return_tuple) {
  CHECK(!instructions.empty());
  HloInstruction& first_instruction = *instructions[0];
  auto new_hlo_module = std::make_unique<HloModule>(
      first_instruction.GetModule()->name() + "_collective_ops",
      HloModuleConfig{},
      std::make_unique<CompilationEnvironments>(
          first_instruction.GetModule()->comp_envs()));
  int parameter_number = 0;
  HloComputation::Builder builder("entry_computation");
  HloCloneContext clone_context(new_hlo_module.get());
  std::vector<HloInstruction*> result_instructions;
  absl::flat_hash_map<std::string, HloInstruction*> start_op_map;
  for (auto* hlo : instructions) {
    if (done_ops.contains(hlo->opcode())) {
      std::vector<HloInstruction*> new_operands;
      for (const HloInstruction* operand : hlo->operands()) {
        if (start_op_map.contains(operand->name())) {
          new_operands.push_back(start_op_map[operand->name()]);
        }
      }
      result_instructions.push_back(
          builder.AddInstruction(hlo->CloneWithNewOperands(
              hlo->shape(), new_operands, &clone_context)));
    } else {
      std::vector<HloInstruction*> new_operands;
      for (const HloInstruction* operand : hlo->operands()) {
        std::unique_ptr<HloInstruction> new_parameter =
            HloInstruction::CreateParameter(parameter_number, operand->shape(),
                                            operand->name());
        ++parameter_number;
        new_operands.push_back(
            builder.AddInstruction(std::move(new_parameter)));
      }
      std::unique_ptr<HloInstruction> new_instruction =
          hlo->CloneWithNewOperands(hlo->shape(), new_operands, &clone_context);
      HloInstruction* new_instr_ptr =
          builder.AddInstruction(std::move(new_instruction));
      if (non_optimized_ops.contains(hlo->opcode())) {
        result_instructions.push_back(new_instr_ptr);
      }
      start_op_map[hlo->name()] = new_instr_ptr;
    }
  }

  if (return_tuple) {
    AddFlatTupleRoot(result_instructions, builder);
  }
  HloComputation* entry_computation =
      new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
  // Preserve the execution thread so the extracted module stays internally
  // consistent: a cloned instruction embedded below the entry (e.g. a fusion
  // wrapping an op that lives on a non-"main" thread) still carries its
  // original execution_thread, and HloVerifier requires the calling
  // computation's thread to match.
  entry_computation->SetExecutionThread(
      first_instruction.parent()->execution_thread());
  return new_hlo_module;
}

std::unique_ptr<HloModule> ExtractInstructionIntoNewModule(
    const HloInstruction& hlo) {
  auto new_hlo_module = std::make_unique<HloModule>(
      std::string(hlo.name()), HloModuleConfig{},
      std::make_unique<CompilationEnvironments>(hlo.GetModule()->comp_envs()));
  int parameter_number = 0;
  HloComputation::Builder builder("entry_computation");
  HloCloneContext clone_context(new_hlo_module.get());
  std::vector<HloInstruction*> new_operands;
  for (const HloInstruction* operand : hlo.operands()) {
    std::unique_ptr<HloInstruction> new_parameter =
        HloInstruction::CreateParameter(parameter_number, operand->shape(),
                                        operand->name());
    ++parameter_number;
    new_operands.push_back(builder.AddInstruction(std::move(new_parameter)));
  }
  std::unique_ptr<HloInstruction> new_instruction =
      hlo.CloneWithNewOperands(hlo.shape(), new_operands, &clone_context);
  builder.AddInstruction(std::move(new_instruction));
  HloComputation* entry_computation =
      new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
  // See the comment in ExtractCollectiveOperationsIntoNewModule above.
  entry_computation->SetExecutionThread(hlo.parent()->execution_thread());
  return new_hlo_module;
}

std::unique_ptr<HloModule> ExtractProducerConsumersIntoNewModule(
    const HloInstruction& producer) {
  auto new_hlo_module =
      std::make_unique<HloModule>("extracted", HloModuleConfig{},
                                  std::make_unique<CompilationEnvironments>(
                                      producer.GetModule()->comp_envs()));
  int parameter_number = 0;
  HloComputation::Builder builder("entry_computation");
  HloCloneContext clone_context(new_hlo_module.get());
  absl::InlinedVector<HloInstruction*, 8> producer_operands;
  for (const HloInstruction* operand : producer.operands()) {
    HloInstruction* new_parameter =
        builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_number, operand->shape(), operand->name()));
    ++parameter_number;

    producer_operands.push_back(new_parameter);
  }

  HloInstruction* new_producer =
      builder.AddInstruction(producer.CloneWithNewOperands(
          producer.shape(), producer_operands, &clone_context));

  absl::flat_hash_map<const HloInstruction*, HloInstruction*> operand_map;
  operand_map.emplace(&producer, new_producer);

  std::vector<HloInstruction*> consumer_outputs;
  for (const HloInstruction* c : producer.users()) {
    absl::InlinedVector<HloInstruction*, 8> consumer_operands;
    for (const HloInstruction* operand : c->operands()) {
      auto it = operand_map.find(operand);
      if (it != operand_map.end()) {
        consumer_operands.push_back(it->second);
      } else {
        HloInstruction* new_parameter =
            builder.AddInstruction(HloInstruction::CreateParameter(
                parameter_number, operand->shape(), operand->name()));
        ++parameter_number;
        consumer_operands.push_back(new_parameter);
      }
    }
    consumer_outputs.push_back(builder.AddInstruction(c->CloneWithNewOperands(
        c->shape(), consumer_operands, &clone_context)));
  }

  if (consumer_outputs.size() > 1) {
    AddFlatTupleRoot(consumer_outputs, builder);
  }

  HloComputation* entry_computation =
      new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
  // See the comment in ExtractCollectiveOperationsIntoNewModule above.
  entry_computation->SetExecutionThread(producer.parent()->execution_thread());
  return new_hlo_module;
}

std::unique_ptr<HloModule> ExtractComputationIntoNewModule(
    const HloComputation& computation) {
  auto new_hlo_module = std::make_unique<HloModule>(
      std::string(computation.name()), HloModuleConfig{},
      std::make_unique<CompilationEnvironments>(
          computation.parent()->comp_envs()));
  HloCloneContext clone_context(new_hlo_module.get());
  new_hlo_module->AddEntryComputationWithLayouts(
      computation.CloneInContext(clone_context));
  return new_hlo_module;
}

}  // namespace xla
