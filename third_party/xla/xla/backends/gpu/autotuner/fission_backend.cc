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

#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
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

// Applies stored sub-fusion configurations to the fragments in the fissioned
// module.
absl::Status ApplySubFusionConfigs(
    HloModule* hlo_module, const AutotuneResult::FissionKey& fission_key) {
  HloPrintOptions options = HloPrintOptions::Fingerprint();
  options.set_print_backend_config(true);
  options.set_sort_backend_config(true);
  options.set_print_operand_shape(true);

  for (HloInstruction* fragment_instr :
       hlo_module->entry_computation()->instructions()) {
    if (fragment_instr->opcode() != HloOpcode::kFusion) {
      continue;
    }
    tsl::Fprint128 fp = tsl::Fingerprint128(fragment_instr->ToString(options));
    std::string fp_str = absl::StrCat(fp.high64, "_", fp.low64);
    if (fission_key.sub_fusion_configs().contains(fp_str)) {
      const auto& sub_config = fission_key.sub_fusion_configs().at(fp_str);
      if (sub_config.Is<BlockLevelFusionConfig>()) {
        BlockLevelFusionConfig block_level_fusion_config;
        sub_config.UnpackTo(&block_level_fusion_config);
        GpuBackendConfig gpu_config;
        *gpu_config.mutable_fusion_backend_config()
             ->mutable_block_level_fusion_config() = block_level_fusion_config;
        gpu_config.mutable_fusion_backend_config()->set_kind(kTritonFusionKind);
        TF_RETURN_IF_ERROR(fragment_instr->set_backend_config(gpu_config));
        fragment_instr->set_fusion_kind(HloInstruction::FusionKind::kCustom);
      } else if (sub_config.Is<NativeEmitterBackendConfig>()) {
        NativeEmitterBackendConfig native_emitter_backend_config;
        sub_config.UnpackTo(&native_emitter_backend_config);
        GpuBackendConfig gpu_config;
        *gpu_config.mutable_native_emitter_backend_config() =
            native_emitter_backend_config;
        TF_RETURN_IF_ERROR(fragment_instr->set_backend_config(gpu_config));
      }
    }
  }
  return absl::OkStatus();
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

  // 1. Get Hero configs. E.g. Cublas GEMM.
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> hero_configs,
                      codegen_backend_->GetSupportedConfigs(**supported_instr));
  if (!debug_options().xla_gpu_experimental_autotune_post_fusion()) {
    return hero_configs;
  }

  if (hero_configs.empty()) {
    return hero_configs;
  }

  // 2. Fuse the fragments from the epilogue and prologues.
  TF_RETURN_IF_ERROR(RunPriorityFusion(hlo_module.get()));
  TF_ASSIGN_OR_RETURN(std::vector<FragmentConfigs> fragments_to_process,
                      CollectFragmentConfigs(hlo_module.get()));

  // 3. Generate configuration combinations: Hero configs x Fragment configs.
  return GenerateFissionKeyCombinations(hero_configs, fragments_to_process);
}

absl::StatusOr<std::vector<FissionBackend::FragmentConfigs>>
FissionBackend::CollectFragmentConfigs(HloModule* hlo_module) {
  std::vector<FragmentConfigs> fragments;
  HloPrintOptions options = HloPrintOptions::Fingerprint();
  options.set_print_backend_config(true);
  options.set_sort_backend_config(true);
  options.set_print_operand_shape(true);

  for (HloInstruction* fragment_instr :
       hlo_module->entry_computation()->instructions()) {
    if (fragment_instr->opcode() != HloOpcode::kFusion) {
      continue;
    }

    tsl::Fprint128 fp = tsl::Fingerprint128(fragment_instr->ToString(options));
    std::string fp_str = absl::StrCat(fp.high64, "_", fp.low64);

    std::vector<std::unique_ptr<BackendConfig>> configs;

    for (const auto& backend : fragment_backends_) {
      if (!backend->IsSupported(*fragment_instr)) {
        continue;
      }

      // 1. Try to get supported configs if autotuning is enabled.
      if (should_autotune_(*fragment_instr)) {
        if (auto supported_configs =
                backend->GetSupportedConfigs(*fragment_instr);
            supported_configs.ok() && !supported_configs->empty()) {
          absl::c_move(*supported_configs, std::back_inserter(configs));
          continue;  // Skip default config since we got supported ones.
        }
      }

      // 2. Fallback to default config if no supported configs were found.
      if (auto default_config = backend->GetDefaultConfig(*fragment_instr);
          default_config.ok()) {
        configs.push_back(std::move(*default_config));
      }
    }

    if (!configs.empty()) {
      fragments.push_back({fp_str, std::move(configs)});
    }
  }
  return fragments;
}

std::vector<std::unique_ptr<BackendConfig>>
FissionBackend::GenerateFissionKeyCombinations(
    const std::vector<std::unique_ptr<BackendConfig>>& hero_configs,
    const std::vector<FragmentConfigs>& fragments_to_process) {
  // 1. Unpack Hero configs and use them as the base combinations.
  std::vector<AutotuneResult::FissionKey> combinations;
  for (const auto& hero_cfg : hero_configs) {
    AutotuneResult::FissionKey key;
    key.set_backend(backend());

    if (AutotuneResult::GemmKey gemm_key; hero_cfg->UnpackTo(&gemm_key)) {
      *key.mutable_gemm() = std::move(gemm_key);
    } else if (AutotuneResult::CustomKernelFusionKey custom_key;
               hero_cfg->UnpackTo(&custom_key)) {
      *key.mutable_custom_kernel_fusion() = std::move(custom_key);
    } else {
      continue;
    }
    combinations.push_back(std::move(key));
  }

  // 2. Iteratively expand combinations with each supplementary fragment.
  for (const auto& fragment : fragments_to_process) {
    std::vector<AutotuneResult::FissionKey> next_combinations;
    // For every existing combination, append every possible config of the
    // current fragment.
    for (const auto& key : combinations) {
      for (const auto& config : fragment.configs) {
        AutotuneResult::FissionKey new_key = key;
        (*new_key.mutable_sub_fusion_configs())[fragment.fp_str] = *config;
        next_combinations.push_back(std::move(new_key));
      }
    }
    combinations = std::move(next_combinations);
  }

  // 3. Pack configurations to the output format (Any).
  std::vector<std::unique_ptr<BackendConfig>> result_configs;
  result_configs.reserve(combinations.size());
  for (const auto& key : combinations) {
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(key);
    result_configs.push_back(std::move(any));
  }

  return result_configs;
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
  AutotuneResult::FissionKey fission_key;
  bool has_fission_config = config.UnpackTo(&fission_key);
  if (has_fission_config) {
    if (fission_key.has_gemm()) {
      fissioned_hero_config.PackFrom(fission_key.gemm());
    } else if (fission_key.has_custom_kernel_fusion()) {
      fissioned_hero_config.PackFrom(fission_key.custom_kernel_fusion());
    }
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      GetFissionedAndRewrittenModule(instr));
  TF_ASSIGN_OR_RETURN(HloInstruction * supported_instr,
                      FindFirstSupportedInstruction(hlo_module.get()));
  TF_RETURN_IF_ERROR(
      codegen_backend_->ApplyConfig(*supported_instr, fissioned_hero_config));

  // We need to run priority fusion if we are in post-fusion autotuning mode
  // and we have a fission config (e.g., from cache) that needs to be applied
  // to the fused fragments.
  if (debug_options().xla_gpu_experimental_autotune_post_fusion() &&
      has_fission_config) {
    TF_RETURN_IF_ERROR(RunPriorityFusion(hlo_module.get()));
    TF_RETURN_IF_ERROR(ApplySubFusionConfigs(hlo_module.get(), fission_key));
  }

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
