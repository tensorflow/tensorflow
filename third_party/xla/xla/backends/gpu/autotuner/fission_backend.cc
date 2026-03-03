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

#include "xla/backends/gpu/autotuner/fission_backend.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace gpu {

namespace {

// Replaces the fusion instruction with the instructions from the fissioned
// computation.
absl::Status InlineFissionedComputation(HloInstruction* fusion_instr,
                                        HloComputation* fissioned_computation) {
  if (fusion_instr->opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError("Not a fusion instruction.");
  }
  HloModule* original_module = fusion_instr->GetModule();
  HloCloneContext clone_context(original_module);
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      cloned_instructions;
  HloComputation* parent_computation = fusion_instr->parent();

  for (HloInstruction* instruction_to_clone :
       fissioned_computation->MakeInstructionPostOrder()) {
    if (instruction_to_clone->opcode() == HloOpcode::kParameter) {
      cloned_instructions[instruction_to_clone] = fusion_instr->mutable_operand(
          instruction_to_clone->parameter_number());
      continue;
    }

    std::vector<HloInstruction*> new_operands;
    for (const HloInstruction* operand : instruction_to_clone->operands()) {
      new_operands.push_back(cloned_instructions.at(operand));
    }
    HloInstruction* new_instruction = parent_computation->AddInstruction(
        instruction_to_clone->CloneWithNewOperands(
            instruction_to_clone->shape(), new_operands, &clone_context));
    cloned_instructions[instruction_to_clone] = new_instruction;
  }
  HloInstruction* new_root =
      cloned_instructions.at(fissioned_computation->root_instruction());
  return parent_computation->ReplaceInstruction(fusion_instr, new_root);
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
FissionBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    VLOG(3) << "Instruction not supported by " << name() << ": "
            << instr.ToString();
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      GetFissionedAndRewrittenModule(instr));
  absl::StatusOr<HloInstruction*> supported_instr =
      FindFirstSupportedInstruction(hlo_module.get());
  if (supported_instr.status().code() == absl::StatusCode::kNotFound) {
    VLOG(3) << "No supported instructions found by " << name() << ": "
            << instr.ToString();
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  TF_RETURN_IF_ERROR(supported_instr.status());
  return codegen_backend_->GetSupportedConfigs(**supported_instr);

  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::StatusOr<std::unique_ptr<BackendConfig>> FissionBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError("Not a fusion instruction.");
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      GetFissionedAndRewrittenModule(instr));
  TF_ASSIGN_OR_RETURN(HloInstruction * supported_instr,
                      FindFirstSupportedInstruction(hlo_module.get()));
  return codegen_backend_->GetDefaultConfig(*supported_instr);
}

absl::StatusOr<std::unique_ptr<HloModule>> FissionBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      codegen_backend_->RunHloPasses(std::move(hlo_module), options));

  // Run priority fusion to fuse the fissioned HLOs.
  HloCostAnalysis::Options priority_fusion_options;
  priority_fusion_options.count_multiple_input_accesses = true;
  // TODO: b/407494653 - Get rid of PriorityFusion.
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, target_config().device_description, alias_info_,
      priority_fusion_options, mlir_context_);
  TF_RETURN_IF_ERROR(priority_fusion.Run(module.get()).status());
  return module;
}

absl::Status FissionBackend::ApplyConfig(HloInstruction& instr,
                                         const BackendConfig& config) {
  HloModule* module = instr.GetModule();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      GetFissionedAndRewrittenModule(instr));
  TF_ASSIGN_OR_RETURN(HloInstruction * supported_instr,
                      FindFirstSupportedInstruction(hlo_module.get()));
  TF_RETURN_IF_ERROR(codegen_backend_->ApplyConfig(*supported_instr, config));
  TF_RETURN_IF_ERROR(
      InlineFissionedComputation(&instr, hlo_module->entry_computation()));
  return module->RemoveUnusedComputations();
}

bool FissionBackend::IsSupported(const HloInstruction& instr) {
  return instr.opcode() == HloOpcode::kFusion;
}

absl::StatusOr<std::unique_ptr<HloModule>>
FissionBackend::GetFissionedAndRewrittenModule(
    const HloInstruction& fusion_instr) {
  const auto* fusion = Cast<HloFusionInstruction>(&fusion_instr);
  std::unique_ptr<HloModule> hlo_module =
      ExtractComputationIntoNewModule(*fusion->called_computation());
  TF_RETURN_IF_ERROR(rewriter_pipeline_->Run(hlo_module.get()).status());
  return hlo_module;
}

absl::StatusOr<HloInstruction*> FissionBackend::FindFirstSupportedInstruction(
    const HloModule* module) {
  std::vector<HloInstruction*> supported_instructions;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (codegen_backend_->IsSupported(*instruction)) {
        supported_instructions.push_back(instruction);
      }
    }
  }
  if (supported_instructions.empty()) {
    return absl::NotFoundError("No supported instructions found.");
  }
  if (supported_instructions.size() > 1) {
    LOG(WARNING) << "Backend " << name()
                 << " found multiple supported instructions found. Using the "
                    "first one.";
  }
  return supported_instructions[0];
}

}  // namespace gpu

}  // namespace xla
