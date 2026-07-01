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

#include "xla/service/gpu/autotuning/autotuner_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_assigner.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/backends/gpu/autotuner/legacy_cache.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/compiler.h"
#include "xla/service/decision.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

namespace {

using AutotuneDecision = Decision;

// Register spilling is currently not allowed for GEMM/Conv fusions, but allowed
// for non-GEMM fusions. Register spilling configurations can be expensive to
// run and might lead to increased memory usage and potential compile-time
// regressions. These are allowed on non-GEMM fusions because sometimes these
// are the only viable configurations to try.
AutotuneDecision AllowRegSpillsForGpuInstruction(
    const HloInstruction& instruction) {
  if (instruction.opcode() == HloOpcode::kCustomCall) {
    if (IsCublasLtGemm(instruction) ||
        IsCustomCallToDnnConvolution(instruction)) {
      return AutotuneDecision::Forbid(
          "Register spilling is not allowed for GEMM/Conv custom calls");
    }
  }
  if (instruction.opcode() == HloOpcode::kFusion) {
    auto gpu_config = instruction.backend_config<GpuBackendConfig>();
    if (gpu_config.ok()) {
      const FusionBackendConfig& backend_config =
          gpu_config->fusion_backend_config();
      if (backend_config.kind() == kTritonGemmFusionKind ||
          backend_config.kind() == kCuDnnFusionKind ||
          backend_config.kind() == kCustomFusionKind) {
        return AutotuneDecision::Forbid(
            "Register spilling is not allowed for GEMM/Conv fusions");
      }
    }
  }
  return AutotuneDecision::Allow();
}

AutotuneDecision ShouldAutotuneCustomCall(bool do_not_autotune_cublas,
                                          bool do_not_autotune_cudnn,
                                          const HloInstruction& instruction) {
  auto gpu_config = instruction.backend_config<GpuBackendConfig>();
  if (IsCublasLtGemm(instruction)) {
    if (do_not_autotune_cublas) {
      return AutotuneDecision::Forbid("Autotuning cuBLAS is disabled");
    }
    if (gpu_config.ok()) {
      // Grouped matmul stores the selected algorithm in the nested
      // grouped_gemm_backend_config, not the top-level gemm_backend_config.
      const GemmBackendConfig& gemm_config =
          IsCublasLtGroupedMatmul(instruction)
              ? gpu_config->grouped_gemm_backend_config().gemm_backend_config()
              : gpu_config->gemm_backend_config();
      if (gemm_config.has_selected_algorithm()) {
        return AutotuneDecision::Forbid(
            "cuBLAS GEMM already has a selected algorithm");
      }
    }
    return AutotuneDecision::Allow();
  }
  if (IsCustomCallToDnnConvolution(instruction)) {
    if (do_not_autotune_cudnn) {
      return AutotuneDecision::Forbid("Autotuning cuDNN is disabled");
    }
    if (gpu_config.ok() &&
        gpu_config->cudnn_conv_backend_config().has_algorithm()) {
      return AutotuneDecision::Forbid(
          "cuDNN convolution already has a selected algorithm");
    }
    return AutotuneDecision::Allow();
  }
  return AutotuneDecision::Forbid(
      "Instruction is not a supported custom call (GEMM or Conv)");
}

AutotuneDecision ShouldAutotuneGemmFusion(const HloInstruction& instruction) {
  auto gpu_config = instruction.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok()) {
    return AutotuneDecision::Forbid(absl::StrCat(
        "Failed to get GPU backend config: ", gpu_config.status().message()));
  }
  const FusionBackendConfig& backend_config =
      gpu_config->fusion_backend_config();
  if (backend_config.kind() == kTritonGemmFusionKind) {
    if (backend_config.has_triton_gemm_config()) {
      return AutotuneDecision::Forbid(
          "Triton GEMM fusion already has a config");
    }
    return AutotuneDecision::Allow();
  }
  if (backend_config.kind() == kCuDnnFusionKind) {
    if (backend_config.has_cudnn_fusion_config()) {
      return AutotuneDecision::Forbid("cuDNN fusion already has a config");
    }
    return AutotuneDecision::Allow();
  }
  if (backend_config.kind() == kCustomFusionKind) {
    if (backend_config.has_custom_fusion_config()) {
      return AutotuneDecision::Forbid("Custom fusion already has a config");
    }
    return AutotuneDecision::Allow();
  }
  return AutotuneDecision::Forbid(
      "Fusion kind is not supported for GEMM autotuning");
}

AutotuneDecision ShouldAutotunGenericFusion(bool enable_fusion_autotuner,
                                            const HloInstruction& instruction) {
  if (!enable_fusion_autotuner) {
    return AutotuneDecision::Forbid("Fusion autotuner is disabled");
  }
  auto fusion = Cast<const HloFusionInstruction>(&instruction);
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return AutotuneDecision::Forbid(
        "Custom fusions are not supported for generic fusion autotuning");
  }
  if (absl::c_any_of(fusion->fused_instructions_computation()->instructions(),
                     HloPredicateIsOp<HloOpcode::kScatter>)) {
    return AutotuneDecision::Forbid("Fusions with Scatter are not supported");
  }
  return AutotuneDecision::Allow();
}

AutotuneDecision ShouldAutotuneInstruction(bool do_not_autotune_cublas,
                                           bool do_not_autotune_cudnn,
                                           bool enable_fusion_autotuner,
                                           const HloInstruction& instruction) {
  // 1. Custom calls.
  if (instruction.opcode() == HloOpcode::kCustomCall) {
    return ShouldAutotuneCustomCall(do_not_autotune_cublas,
                                    do_not_autotune_cudnn, instruction);
  }
  if (instruction.opcode() == HloOpcode::kFusion) {
    // 2. GEMM fusions.
    auto gpu_config = instruction.backend_config<GpuBackendConfig>();
    if (!gpu_config.ok()) {
      return AutotuneDecision::Forbid(absl::StrCat(
          "Failed to get GPU backend config: ", gpu_config.status().message()));
    }
    const FusionBackendConfig& backend_config =
        gpu_config->fusion_backend_config();
    if (backend_config.kind() == kTritonGemmFusionKind ||
        backend_config.kind() == kCuDnnFusionKind ||
        backend_config.kind() == kCustomFusionKind) {
      return ShouldAutotuneGemmFusion(instruction);
    }
    // 3. Generic fusions.
    return ShouldAutotunGenericFusion(enable_fusion_autotuner, instruction);
  }
  return AutotuneDecision::Forbid(
      "Instruction is neither custom call nor fusion");
}

}  // namespace

ConfigAssigner::Options GetConfigAssignerOptions(
    const DebugOptions& debug_options, bool is_deviceless) {
  ConfigAssigner::Options options;
  options.check_buffers = debug_options.xla_gpu_autotune_level() >= 4;
  options.relative_tolerance = debug_options.xla_gpu_autotune_gemm_rtol();
  options.crash_on_check_failure =
      debug_options.xla_gpu_crash_on_verification_failures();
  options.dump_logs_to = debug_options.xla_gpu_dump_autotune_logs_to();
  options.select_first_config =
      debug_options.xla_gpu_deterministic_ops() ||
      debug_options.xla_gpu_exclude_nondeterministic_ops() ||
      debug_options.xla_gpu_autotune_level() == 0;

  if (is_deviceless) {
    // If we are running on a deviceless target, we want to use default configs.
    options.use_default_config = true;
  }

  options.expect_all_instructions_in_cache =
      debug_options.xla_gpu_require_complete_aot_autotune_results();
  options.dump_hlos = debug_options.xla_gpu_dump_autotuned_gemm_fusions() ||
                      debug_options.xla_gpu_dump_autotuned_instructions();

  return options;
}

CodegenOrchestrator::Options GetCodegenOrchestratorOptions(
    const DebugOptions& debug_options) {
  CodegenOrchestrator::Options options;
  options.exclude_cublas_config = !debug_options.xla_gpu_cublas_fallback();
  if (!debug_options.xla_gpu_fail_ptx_compilation_on_register_spilling()) {
    options.allow_reg_spills_fn = [](const HloInstruction& instr) {
      return static_cast<bool>(AllowRegSpillsForGpuInstruction(instr));
    };
  }
  // xla_gpu_filter_kernels_spilling_registers_on_autotuning is true by default,
  // but some autotuner passes need to set it to false explicitly as there
  // aren't enough configs to guarantee that no config will spill. So we allow
  // allow_reg_spills to override the autotuner config unless this flag is
  // explicitly set to false.
  if (!debug_options
           .xla_gpu_filter_kernels_spilling_registers_on_autotuning()) {
    options.allow_reg_spills_fn = [](const HloInstruction&) { return false; };
  }
  return options;
}

ProfileOptions GetProfileOptions(
    const DebugOptions& debug_options,
    const ConfigAssigner::Options& config_assigner_options) {
  ProfileOptions profile_options;
  profile_options.redzone_padding_bytes =
      debug_options.xla_gpu_redzone_padding_bytes();
  profile_options.should_init_buffers = config_assigner_options.check_buffers;
  return profile_options;
}

absl::StatusOr<std::vector<std::unique_ptr<CodegenBackend>>>
AutotunerPass::GetGpuAutotunerBackends(
    se::StreamExecutor* stream_exec,
    se::DeviceAddressAllocator* device_allocator,
    const Compiler::GpuTargetConfig* target_config, const AliasInfo* alias_info,
    const DebugOptions& debug_options, mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn, Compiler* compiler,
    se::PlatformId platform_id) {
  std::vector<autotuner::Backend> autotune_backends;
  if (!debug_options.xla_gpu_experimental_autotune_backends().empty()) {
    for (const auto& backend :
         debug_options.xla_gpu_experimental_autotune_backends()) {
      autotune_backends.push_back(static_cast<autotuner::Backend>(backend));
    }
  } else {
    for (int i = 0; i < autotuner::Backend_descriptor()->value_count(); ++i) {
      const auto backend = static_cast<autotuner::Backend>(
          autotuner::Backend_descriptor()->value(i)->number());
      if (backend != autotuner::Backend::UNSPECIFIED_BACKEND) {
        autotune_backends.push_back(backend);
      }
    }
  }

  std::vector<autotuner::Backend> disabled_autotune_backends;
  if (debug_options.xla_gpu_experimental_disable_binary_libraries()) {
    disabled_autotune_backends.push_back(autotuner::Backend::CUBLASLT);
    disabled_autotune_backends.push_back(autotuner::Backend::CUDNN);
    disabled_autotune_backends.push_back(autotuner::Backend::HIPBLASLT);
    disabled_autotune_backends.push_back(autotuner::Backend::MIOPEN);
    disabled_autotune_backends.push_back(autotuner::Backend::HIPBLASLT_FISSION);
  }

  if (debug_options.xla_gpu_autotune_level() == 0 ||
      debug_options.xla_gpu_exclude_nondeterministic_ops() ||
      !debug_options.xla_gpu_experimental_enable_fusion_autotuner()) {
    disabled_autotune_backends.push_back(autotuner::Backend::NATIVE_EMITTER);
    disabled_autotune_backends.push_back(
        autotuner::Backend::BLOCK_LEVEL_EMITTER);
  }

  if (debug_options.xla_gpu_exclude_nondeterministic_ops() ||
      debug_options.xla_gpu_deterministic_ops()) {
    disabled_autotune_backends.push_back(autotuner::Backend::TRITON);
  }

  autotune_backends.erase(
      std::remove_if(autotune_backends.begin(), autotune_backends.end(),
                     [&](autotuner::Backend backend) {
                       return absl::c_linear_search(disabled_autotune_backends,
                                                    backend);
                     }),
      autotune_backends.end());

  auto& registry = stream_executor::PlatformObjectRegistry::GetGlobalRegistry();
  ASSIGN_OR_RETURN(const GetCodegenBackends::Type& get_codegen_backends,
                   registry.FindObject<GetCodegenBackends>(platform_id));
  std::vector<std::unique_ptr<CodegenBackend>> backends = get_codegen_backends(
      stream_exec, device_allocator, &debug_options, compiler, target_config,
      alias_info, mlir_context, shape_size_fn, autotune_backends);

  return backends;
}

absl::StatusOr<std::unique_ptr<AutotunerPass>> AutotunerPass::Create(
    AutotunerPass::GetBackendsFn get_backends_fn,
    const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version,
    se::StreamExecutor* stream_executor, tsl::thread::ThreadPool* thread_pool,
    const Compiler::GpuTargetConfig* target_config, const AliasInfo* alias_info,
    mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    se::DeviceAddressAllocator* allocator,
    MultiProcessKeyValueStore key_value_store) {
  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<CodegenBackend>> backends,
                   get_backends_fn());

  // 1. Assessing whether to autotune custom calls.
  bool do_not_autotune_cublas =
      debug_options.xla_gpu_experimental_disable_binary_libraries() ||
      debug_options.xla_gpu_autotune_level() == 0;
  bool do_not_autotune_cudnn =
      debug_options.xla_gpu_experimental_disable_binary_libraries() ||
      (do_not_autotune_cublas && !gpu_version.IsRocm());

  // 3. Assessing whether to autotune generic fusions.
  bool enable_fusion_autotuner =
      debug_options.xla_gpu_autotune_level() != 0 &&
      !debug_options.xla_gpu_exclude_nondeterministic_ops() &&
      debug_options.xla_gpu_experimental_enable_fusion_autotuner();

  auto should_autotune =
      [do_not_autotune_cublas, do_not_autotune_cudnn,
       enable_fusion_autotuner](const HloInstruction& instruction) -> bool {
    AutotuneDecision decision =
        ShouldAutotuneInstruction(do_not_autotune_cublas, do_not_autotune_cudnn,
                                  enable_fusion_autotuner, instruction);
    if (!decision) {
      VLOG(3) << "Not autotuning " << instruction.name() << ": "
              << decision.Explain();
    }
    return decision.IsAllowed();
  };

  bool is_deviceless = stream_executor == nullptr;
  ConfigAssigner::Options assigner_options =
      GetConfigAssignerOptions(debug_options, is_deviceless);
  CodegenOrchestrator::Options orchestrator_options =
      GetCodegenOrchestratorOptions(debug_options);

  std::unique_ptr<Profiler> profiler = nullptr;
  if (!is_deviceless) {
    if (stream_executor->GetPlatform()->id() ==
        stream_executor::sycl::kSyclPlatformId) {
      // TODO(intel-tf): Enable buffer checking for SYCL once
      // BufferComparatorKernel and RedzoneAllocatorKernel are registered for
      // SYCL platform.
      assigner_options.check_buffers = false;
    }
    profiler = GpuProfiler::Create(
        stream_executor, GetProfileOptions(debug_options, assigner_options),
        allocator);
  }

  VLOG(1) << "ConfigAssigner options: " << assigner_options.ToString();

  std::string cache_dir = debug_options.xla_gpu_per_fusion_autotune_cache_dir();
  if (cache_dir.empty()) {
    cache_dir = debug_options.xla_gpu_experimental_autotuner_cache_dir();
  }
  std::unique_ptr<AutotunerCacheInterface> cache =
      std::make_unique<LegacyCache>(
          cache_dir, debug_options.xla_gpu_experimental_autotune_cache_mode(),
          target_config->device_description);

  ASSIGN_OR_RETURN(auto orchestrator,
                   CodegenOrchestrator::Create(
                       std::move(backends), orchestrator_options, thread_pool));

  ASSIGN_OR_RETURN(
      auto config_assigner,
      ConfigAssigner::Create(assigner_options, std::move(cache),
                             std::move(orchestrator), std::move(profiler)));

  return absl::WrapUnique(new AutotunerPass(
      std::move(config_assigner), std::move(should_autotune),
      std::move(key_value_store), debug_options.xla_gpu_shard_autotuning()));
}

absl::StatusOr<bool> AutotunerPass::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER("AutotunerPass");
  VLOG(1) << "Running Autotuner Pass";

  bool shard_autotuning =
      enable_sharding_ && key_value_store_.process_count > 1;
  if (shard_autotuning) {
    RETURN_IF_ERROR(config_assigner_->AssignConfigs(module, should_autotune_,
                                                    key_value_store_));
  } else {
    RETURN_IF_ERROR(config_assigner_->AssignConfigs(module, should_autotune_));
  }
  VLOG(1) << "Autotuner cache stats: hits="
          << config_assigner_->GetCacheStats().hits
          << ", misses=" << config_assigner_->GetCacheStats().misses;
  return true;
}

}  // namespace gpu
}  // namespace xla
