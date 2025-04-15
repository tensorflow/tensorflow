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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/compilation_environments.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

// Returns whether it makes sense to run the given instruction in isolation
// (e.g. whether it can run without dependent instructions).
bool ShouldIsolateOpcode(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kConstant:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
      return false;
    default:
      return true;
  }
}

absl::StatusOr<std::vector<std::unique_ptr<HloModule>>> Decompose(
    const HloModule& module) {
  std::vector<std::unique_ptr<HloModule>> modules;

  absl::flat_hash_set<const HloComputation*> computations_to_visit{
      module.entry_computation()};
  absl::flat_hash_set<const HloComputation*> visited_computations;

  // Traverse the computation tree, starting from the entry computation, and
  // recursing into the called computations.
  while (!computations_to_visit.empty()) {
    const HloComputation* computation = *computations_to_visit.begin();
    computations_to_visit.erase(computations_to_visit.begin());
    visited_computations.insert(computation);

    for (const HloInstruction* instruction : computation->instructions()) {
      // Skip called computations in the embedded context (fusion, reduce, map,
      // etc), as within these computations instructions are not lowered
      // individually and it doesn't make sense to test them in isolation.
      if (GetInstructionCallContext(instruction->opcode()) !=
          CallContext::kEmbedded) {
        for (const HloComputation* called_computation :
             instruction->called_computations()) {
          if (!visited_computations.contains(called_computation)) {
            computations_to_visit.insert(called_computation);
          }
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
    const std::string fingerprint = module->GetFingerprint128();
    if (module_fingerprints.contains(fingerprint)) {
      return false;
    }
    module_fingerprints.insert(fingerprint);
    return true;
  };

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<HloModule>> isolated_modules,
                      Decompose(module));
  for (auto& module : isolated_modules) {
    if (should_add_module(module.get())) {
      modules.push_back(std::move(module));
    }
  }
  return modules;
}

std::unique_ptr<HloModule> ExtractCollectiveOperationsIntoNewModule(
    const std::vector<HloInstruction*>& instructions,
    const absl::flat_hash_set<HloOpcode>& done_ops,
    const absl::flat_hash_set<HloOpcode>& non_optimized_ops) {
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

  std::unique_ptr<HloInstruction> tuple_instruction =
      HloInstruction::CreateTuple(result_instructions);
  builder.AddInstruction(std::move(tuple_instruction));
  new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
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
  new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
  return new_hlo_module;
}

std::unique_ptr<HloModule> ExtractProducerConsumerIntoNewModule(
    const HloInstruction& producer, const HloInstruction& consumer) {
  auto new_hlo_module =
      std::make_unique<HloModule>("extracted", HloModuleConfig{},
                                  std::make_unique<CompilationEnvironments>(
                                      consumer.GetModule()->comp_envs()));
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

  absl::InlinedVector<HloInstruction*, 8> consumer_operands;
  for (const HloInstruction* operand : consumer.operands()) {
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
  builder.AddInstruction(consumer.CloneWithNewOperands(
      consumer.shape(), consumer_operands, &clone_context));

  new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
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
