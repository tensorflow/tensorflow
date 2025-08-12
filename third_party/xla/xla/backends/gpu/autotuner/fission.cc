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

#include "xla/backends/gpu/autotuner/fission.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/gpu/autotuner/cublas.h"
#include "xla/backends/gpu/autotuner/cublaslt.h"
#include "xla/backends/gpu/autotuner/custom_kernel.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/call_inliner.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;
using CublasBackendConfig = AutotuneResult::GemmKey;
using CublasLtBackendConfig = AutotuneResult::GemmKey;
using CustomKernelBackendConfig = AutotuneResult::CustomKernelFusionKey;

namespace {
HloCostAnalysis::Options PriorityFusionOptions() {
  // The real pointer size is set in GpuCompiler. In HloCostAnalysis, the
  // pointer size is used only to determine the size of tuple types. We
  // shouldn't have any tuples in the autotuned module, so it's safe to use
  // the default value here, instead of piping the real value.
  return {.count_multiple_input_accesses = true};
}
}  // namespace

// Unfuses a fusion instruction and rewrites it to using a cublas or cublasLt
// custom call for the dot operation.
// If rewrite_to_cublaslt is true, we will try to rewrite the dot to a cublasLt
// custom call, otherwise we will try to rewrite it to a cublas custom call.
absl::Status FissionToCublas(HloModule* hlo_module,
                             const se::DeviceDescription& device_description,
                             bool rewrite_to_cublaslt) {
  if (rewrite_to_cublaslt) {
    hlo_module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_enable_cublaslt(true);
  }
  HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *hlo_module->entry_computation(), HloOpcode::kDot);

  if (dot == nullptr) {
    return absl::InvalidArgumentError(
        "No dot instruction found in the fusion.");
  }

  // Substitute algorithms, which are not supported by cuBLAS for the check, but
  // don't use cuBlas in the end. This assumes that the substituting algorithm
  // has result which are close enough for the check in this file.
  if (dot->precision_config().algorithm() ==
      PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3) {
    dot->mutable_precision_config()->set_algorithm(
        PrecisionConfig::ALG_DOT_F32_F32_F32);
  }

  bool is_rewritten_to_cublas_custom_call = false;
  for (GemmRewriterOptions::DType dtype :
       {GemmRewriterOptions::DType::kFp8Only,
        GemmRewriterOptions::DType::kNonFp8Only}) {
    DotAlgorithmRewriter dot_algorithm_rewriter;

    TF_RETURN_IF_ERROR(dot_algorithm_rewriter.Run(hlo_module).status());

    GemmRewriter gemm_rewriter(device_description.gpu_compute_capability(),
                               device_description.runtime_version(),
                               GemmRewriterOptions{dtype});
    TF_ASSIGN_OR_RETURN(bool changed, gemm_rewriter.Run(hlo_module));
    is_rewritten_to_cublas_custom_call |= changed;

    PriorityFusion fusion_pass(
        /*thread_pool=*/nullptr, device_description, PriorityFusionOptions());
    TF_RETURN_IF_ERROR(fusion_pass.Run(hlo_module).status());
  }

  if (is_rewritten_to_cublas_custom_call) {
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Failed to rewrite fusion to cuBLAS.");
}

absl::Status FissionToCustomKernel(
    HloModule* hlo_module, const se::DeviceDescription& device_description) {
  CustomKernelFusionRewriter custom_kernel_fusion_rewriter(&device_description);
  PriorityFusion fusion_pass(
      /*thread_pool=*/nullptr, device_description, PriorityFusionOptions());
  TF_ASSIGN_OR_RETURN(bool is_rewritten_to_custom_kernel,
                      custom_kernel_fusion_rewriter.Run(hlo_module));
  TF_RETURN_IF_ERROR(fusion_pass.Run(hlo_module).status());

  if (is_rewritten_to_custom_kernel) {
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "Failed to rewrite fusion to custom kernel.");
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> GetCublasConfigs(
    CublasBackend& cublas_backend, std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_executor) {
  std::vector<std::unique_ptr<BackendConfig>> configs;

  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsLegacyCublasMatmul(*instruction)) {
        TF_ASSIGN_OR_RETURN(configs,
                            cublas_backend.GetSupportedConfigs(*instruction));
        return configs;
      }
    }
  }

  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> GetCublasLtConfigs(
    CublasLtBackend& cublaslt_backend, std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_executor) {
  std::vector<std::unique_ptr<BackendConfig>> configs;

  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (IsCublasLtMatmul(*instruction) || IsCublasLtMatmulF8(*instruction)) {
        TF_ASSIGN_OR_RETURN(configs,
                            cublaslt_backend.GetSupportedConfigs(*instruction));
        return configs;
      }
    }
  }

  return configs;
}

bool IsCustomKernel(const HloComputation* computation) {
  if (!computation->IsFusionComputation()) {
    return false;
  }

  HloInstruction* instruction = computation->FusionInstruction();
  absl::StatusOr<GpuBackendConfig> gpu_backend_config =
      instruction->backend_config<GpuBackendConfig>();
  if (!gpu_backend_config.ok()) {
    return false;
  }

  if (instruction->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return false;
  }

  if (!gpu_backend_config->has_fusion_backend_config()) {
    return false;
  }

  return gpu_backend_config->fusion_backend_config().kind() ==
         kCustomFusionKind;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetCustomKernelConfigs(CustomKernelBackend& custom_kernel_backend,
                       std::unique_ptr<HloModule> hlo_module,
                       se::StreamExecutor* stream_executor) {
  std::vector<std::unique_ptr<BackendConfig>> configs;

  for (HloComputation* computation : hlo_module->computations()) {
    if (IsCustomKernel(computation)) {
      TF_ASSIGN_OR_RETURN(configs, custom_kernel_backend.GetSupportedConfigs(
                                       *computation->FusionInstruction()));
    }
  }

  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
FissionBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError("Not a fusion instruction.");
  }

  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(&instr);
  const HloComputation* fusion_computation = fusion->called_computation();
  std::unique_ptr<HloModule> hlo_module =
      ExtractComputationIntoNewModule(*fusion_computation);

  std::vector<std::unique_ptr<BackendConfig>> configs;
  std::unique_ptr<HloModule> cublas_hlo_module = hlo_module->Clone();
  if (FissionToCublas(cublas_hlo_module.get(),
                      target_config().device_description,
                      /*rewrite_to_cublaslt=*/false)
          .ok()) {
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<BackendConfig>> cublas_configs,
        GetCublasConfigs(cublas_backend_, std::move(cublas_hlo_module),
                         stream_executor()));
    configs.insert(configs.end(),
                   std::make_move_iterator(cublas_configs.begin()),
                   std::make_move_iterator(cublas_configs.end()));
  }

  std::unique_ptr<HloModule> cublaslt_hlo_module = hlo_module->Clone();
  if (FissionToCublas(cublaslt_hlo_module.get(),
                      target_config().device_description,
                      /*rewrite_to_cublaslt=*/true)
          .ok()) {
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<BackendConfig>> cublaslt_configs,
        GetCublasLtConfigs(cublaslt_backend_, std::move(cublaslt_hlo_module),
                           stream_executor()));
    configs.insert(configs.end(),
                   std::make_move_iterator(cublaslt_configs.begin()),
                   std::make_move_iterator(cublaslt_configs.end()));
  }

  std::unique_ptr<HloModule> custom_kernel_hlo_module = hlo_module->Clone();
  if (FissionToCustomKernel(custom_kernel_hlo_module.get(),
                            target_config().device_description)
          .ok()) {
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<BackendConfig>> custom_kernel_configs,
        GetCustomKernelConfigs(custom_kernel_backend_,
                               std::move(custom_kernel_hlo_module),
                               stream_executor()));
    configs.insert(configs.end(),
                   std::make_move_iterator(custom_kernel_configs.begin()),
                   std::make_move_iterator(custom_kernel_configs.end()));
  }

  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> FissionBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  return absl::InvalidArgumentError(
      "FissionBackend doesn't support getting a default config.");
}

absl::Status FissionBackend::ApplyConfig(HloInstruction& instr,
                                         const BackendConfig& config) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError("Not a fusion instruction.");
  }

  // Inline the fusion into a new module.
  HloComputation* computation = instr.parent();
  HloInstruction* call = computation->AddInstruction(HloInstruction::CreateCall(
      instr.shape(), instr.operands(), instr.fused_instructions_computation()));
  TF_RETURN_IF_ERROR(computation->ReplaceInstruction(&instr, call));
  HloModule* hlo_module = call->GetModule();

  CallInliner call_inliner(
      /*single_call_site=*/false, /*update_domain=*/false,
      /*composites_to_preserve=*/absl::flat_hash_set<std::string>(),
      /*uniquify_channel_ids=*/true);
  TF_RETURN_IF_ERROR(call_inliner.Run(hlo_module).status());

  if (config.GetDescriptor() == CublasBackendConfig::descriptor()) {
    TF_RETURN_IF_ERROR(FissionToCublas(hlo_module,
                                       target_config().device_description,
                                       /*rewrite_to_cublaslt=*/false));
    for (HloComputation* computation :
         hlo_module->MakeNonfusionComputations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (IsLegacyCublasMatmul(*instruction)) {
          TF_RETURN_IF_ERROR(cublas_backend_.ApplyConfig(*instruction, config));
        }
      }
    }

    return absl::OkStatus();
  }

  if (config.GetDescriptor() == CublasLtBackendConfig::descriptor()) {
    TF_RETURN_IF_ERROR(FissionToCublas(hlo_module,
                                       target_config().device_description,
                                       /*rewrite_to_cublaslt=*/true));
    for (HloComputation* computation :
         hlo_module->MakeNonfusionComputations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (IsCublasLtMatmul(*instruction) ||
            IsCublasLtMatmulF8(*instruction)) {
          TF_RETURN_IF_ERROR(
              cublaslt_backend_.ApplyConfig(*instruction, config));
        }
      }
    }

    return absl::OkStatus();
  }

  if (config.GetDescriptor() == CustomKernelBackendConfig::descriptor()) {
    TF_RETURN_IF_ERROR(
        FissionToCustomKernel(hlo_module, target_config().device_description));
    for (HloComputation* computation : hlo_module->computations()) {
      if (IsCustomKernel(computation)) {
        TF_RETURN_IF_ERROR(custom_kernel_backend_.ApplyConfig(
            *computation->FusionInstruction(), config));
      }
    }

    return absl::OkStatus();
  }
  return absl::UnimplementedError("Not implemented.");
}

}  // namespace gpu
}  // namespace xla
