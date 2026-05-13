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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/transforms/priority_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/fingerprint.h"

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

absl::Status FissionBackend::RunPriorityFusion(HloModule* module) {
  HloCostAnalysis::Options priority_fusion_options;
  priority_fusion_options.count_multiple_input_accesses = true;
  PriorityFusion priority_fusion(
      /*thread_pool=*/nullptr, target_config().device_description, alias_info_,
      priority_fusion_options, mlir_context_);
  return priority_fusion.Run(module).status();
}

absl::StatusOr<std::unique_ptr<HloModule>> FissionBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      codegen_backend_->RunHloPasses(std::move(hlo_module), options));

  TF_RETURN_IF_ERROR(RunPriorityFusion(module.get()));
  return module;
}

absl::Status FissionBackend::ApplyConfig(HloInstruction& instr,
                                         const BackendConfig& config) {
  HloModule* module = instr.GetModule();
  BackendConfig fissioned_hero_config = config;
  autotuner::FissionConfig fission_config;
  bool has_fission_config = config.UnpackTo(&fission_config);
  if (has_fission_config) {
    fissioned_hero_config = fission_config.fissioned_hero_config();
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      GetFissionedAndRewrittenModule(instr));
  TF_ASSIGN_OR_RETURN(HloInstruction * supported_instr,
                      FindFirstSupportedInstruction(hlo_module.get()));
  TF_RETURN_IF_ERROR(
      codegen_backend_->ApplyConfig(*supported_instr, fissioned_hero_config));

  // Given that the autotuner runs post fusion, we have to run priority fusion
  // again to fuse the epilogue and prologues.
  if (debug_options().xla_gpu_experimental_autotune_post_fusion()) {
    TF_RETURN_IF_ERROR(RunPriorityFusion(hlo_module.get()));

    if (has_fission_config) {
      HloPrintOptions options = HloPrintOptions::Fingerprint();
      options.set_print_backend_config(true);
      options.set_sort_backend_config(true);
      options.set_print_operand_shape(true);

      for (HloInstruction* fragment_instr :
           hlo_module->entry_computation()->instructions()) {
        if (fragment_instr->opcode() != HloOpcode::kFusion) {
          continue;
        }
        tsl::Fprint128 fp =
            tsl::Fingerprint128(fragment_instr->ToString(options));
        std::string fp_str = absl::StrCat(fp.high64, "_", fp.low64);
        if (fission_config.sub_fusion_configs().contains(fp_str)) {
          const auto& sub_config =
              fission_config.sub_fusion_configs().at(fp_str);
          if (sub_config.Is<BlockLevelFusionConfig>()) {
            BlockLevelFusionConfig block_level_fusion_config;
            sub_config.UnpackTo(&block_level_fusion_config);
            GpuBackendConfig gpu_config;
            *gpu_config.mutable_fusion_backend_config()
                 ->mutable_block_level_fusion_config() =
                block_level_fusion_config;
            gpu_config.mutable_fusion_backend_config()->set_kind(
                kTritonFusionKind);
            TF_RETURN_IF_ERROR(fragment_instr->set_backend_config(gpu_config));
            fragment_instr->set_fusion_kind(
                HloInstruction::FusionKind::kCustom);
          } else if (sub_config.Is<NativeEmitterBackendConfig>()) {
            NativeEmitterBackendConfig native_emitter_backend_config;
            sub_config.UnpackTo(&native_emitter_backend_config);
            GpuBackendConfig gpu_config;
            *gpu_config.mutable_native_emitter_backend_config() =
                native_emitter_backend_config;
            TF_RETURN_IF_ERROR(fragment_instr->set_backend_config(gpu_config));
            fragment_instr->set_fusion_kind(HloInstruction::FusionKind::kInput);
          }
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(
      InlineFissionedComputation(&instr, hlo_module->entry_computation()));
  return module->RemoveUnusedComputations();
}

bool FissionBackend::RequiresSubFusionTuning(
    const HloInstruction& instr) const {
  // Only supported when we autotune post fusion.
  if (debug_options().xla_gpu_experimental_autotune_post_fusion()) {
    return instr.opcode() == HloOpcode::kFusion;
  }
  return false;
}

absl::StatusOr<std::vector<std::unique_ptr<HloModule>>>
FissionBackend::GenerateSubFusions(const HloInstruction& instr,
                                   const BackendConfig& config) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      GetFissionedAndRewrittenModule(instr));
  TF_ASSIGN_OR_RETURN(HloInstruction * supported_instr,
                      FindFirstSupportedInstruction(hlo_module.get()));
  BackendConfig fissioned_hero_config = config;
  if (config.Is<autotuner::FissionConfig>()) {
    autotuner::FissionConfig fission_config;
    config.UnpackTo(&fission_config);
    fissioned_hero_config = fission_config.fissioned_hero_config();
  }
  TF_RETURN_IF_ERROR(
      codegen_backend_->ApplyConfig(*supported_instr, fissioned_hero_config));
  TF_RETURN_IF_ERROR(RunPriorityFusion(hlo_module.get()));

  std::vector<std::unique_ptr<HloModule>> sub_fusions;
  for (const HloInstruction* fragment_instr :
       hlo_module->entry_computation()->instructions()) {
    // Skip the supported instruction itself, i.e. the custom-call.
    if (fragment_instr == supported_instr) {
      continue;
    }
    // Skip non-fusion instructions.
    if (fragment_instr->opcode() != HloOpcode::kFusion) {
      continue;
    }
    sub_fusions.push_back(ExtractInstructionIntoNewModule(*fragment_instr));
  }
  return sub_fusions;
}

absl::Status FissionBackend::StoreSubFusionConfigs(
    BackendConfig& config,
    const absl::flat_hash_map<tsl::Fprint128, BackendConfig,
                              tsl::Fprint128Hasher>& sub_fusion_configs) {
  autotuner::FissionConfig fission_config;
  *fission_config.mutable_fissioned_hero_config() = config;
  for (const auto& [fp, child_config] : sub_fusion_configs) {
    (*fission_config.mutable_sub_fusion_configs())[absl::StrCat(
        fp.high64, "_", fp.low64)] = child_config;
  }
  config.PackFrom(fission_config);
  return absl::OkStatus();
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
