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

#include "xla/backends/gpu/autotuner/cublaslt.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/triton/dot_search_space.h"
#include "xla/backends/gpu/autotuner/triton/triton_configs.h"
#include "xla/backends/gpu/transforms/convert_triton_gemm_config.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/gpu_dot_fusion_cost_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;
using se::gpu::BlasLt;

using CublasLtBackendConfig = AutotuneResult::GemmKey;

namespace {

absl::StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return Internal("Unsupported Epilogue.");
  }
}

absl::StatusOr<std::unique_ptr<HloModule>> CreateSyntheticTritonDotModule(
    const HloInstruction& instr, const GemmBackendConfig& backend_config,
    const DebugOptions& debug_options) {
  HloModuleConfig module_config;
  module_config.set_debug_options(debug_options);
  auto isolated_module =
      std::make_unique<HloModule>("synthetic_dot_module", module_config);
  HloComputation::Builder builder("synthetic_dot_computation");
  HloInstruction* lhs_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, instr.operand(0)->shape(), "lhs"));
  HloInstruction* rhs_param = builder.AddInstruction(
      HloInstruction::CreateParameter(1, instr.operand(1)->shape(), "rhs"));
  HloInstruction* dot_instr = builder.AddInstruction(HloInstruction::CreateDot(
      instr.shape().tuple_shapes(0), lhs_param, rhs_param,
      backend_config.dot_dimension_numbers(),
      backend_config.precision_config()));
  HloComputation* entry = isolated_module->AddEntryComputation(builder.Build());
  HloInstruction* fusion = entry->CreateFusionInstruction(
      {dot_instr}, HloInstruction::FusionKind::kCustom);
  entry->set_root_instruction(fusion);
  GpuBackendConfig fusion_gpu_backend_config;
  fusion_gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      kTritonGemmFusionKind);
  ABSL_RETURN_IF_ERROR(fusion->set_backend_config(fusion_gpu_backend_config));
  return isolated_module;
}

absl::StatusOr<std::optional<absl::Duration>> EstimateTritonDotRuntime(
    const HloInstruction& instr, const GemmBackendConfig& backend_config,
    const stream_executor::DeviceDescription& device_info,
    const DebugOptions& debug_options, mlir::MLIRContext* mlir_context) {
  if (mlir_context == nullptr) {
    return std::nullopt;
  }

  // The dot search space and the cost model assume the dot is a part of a
  // module. So we create one.
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> isolated_module,
      CreateSyntheticTritonDotModule(instr, backend_config, debug_options));
  const auto* fusion_instr = Cast<HloFusionInstruction>(
      isolated_module->entry_computation()->root_instruction());
  const auto* dot =
      Cast<HloDotInstruction>(fusion_instr->fused_expression_root());

  // Why do we create a default set? Because the cost model is currently better
  // tuned for the default set and has better quality on it.
  TritonDotFusionSearchSpace search_space(device_info, dot);
  std::vector<TritonGemmConfig> configs =
      search_space.GenerateAndOptimizeConfigs(
          GetDefaultTritonConfigs(device_info.gpu_compute_capability()));

  std::optional<absl::Duration> best_exec_time;
  for (const auto& config : configs) {
    absl::StatusOr<xtile::BlockLevelParameters> block_params =
        FindBlockLevelParameters(dot, config, mlir_context, device_info);
    if (!block_params.ok()) {
      continue;
    }
    absl::StatusOr<EstimateRunTimeData> estimate =
        gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
            dot, *block_params, device_info, config.block_k);
    if (!estimate.ok()) {
      continue;
    }

    if (!best_exec_time.has_value() || estimate->exec_time < *best_exec_time) {
      best_exec_time = estimate->exec_time;
    }
  }

  return best_exec_time;
}

}  // namespace

bool CublasLtBackend::IsSupported(const HloInstruction& instr) {
  return IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr);
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CublasLtBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }

  if (stream_executor() == nullptr) {
    return absl::InvalidArgumentError(
        "CublasLtBackend cannot enumerate configs in deviceless mode.");
  }

  GpuBackendConfig gpu_config =
      instr.backend_config<GpuBackendConfig>().value();
  const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

  ABSL_ASSIGN_OR_RETURN(
      GemmConfig gemm_config,
      GemmConfig::For(
          &instr, target_config().device_description.gpu_compute_capability()));

  ABSL_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                   AsBlasLtEpilogue(backend_config.epilogue()));

  ABSL_ASSIGN_OR_RETURN(BlasLt * blas_lt, se::gpu::BlasLt::Get(stream_executor()));

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<BlasLt::MatmulPlan> plan,
                   blas_lt->GetMatmulPlan(gemm_config, epilogue));

  const Shape& output_shape = instr.shape();
  if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
    return Internal(
        "Invalid shape for CublasLt matmul: output is not a non-empty tuple.");
  }
  // The last element of the output tuple is the workspace.
  const int64_t workspace_size =
      ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

  int max_algorithms = debug_options().xla_gpu_blas_max_algorithms();
  if (max_algorithms <= 0) {
    max_algorithms = GemmConfig::kNumAlgorithms;
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<BlasLt::MatmulAlgorithm> algorithms,
                   plan->GetAlgorithms(max_algorithms, workspace_size));
  int num_algorithms = algorithms.size();
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(num_algorithms);
  for (int i = 0; i < num_algorithms; ++i) {
    auto config = std::make_unique<BackendConfig>();
    auto* gemm_key = config->mutable_gemm();
    gemm_key->set_algorithm(i);
    gemm_key->set_autotune_workspace_size(workspace_size);
    configs.push_back(std::move(config));
  }

  return configs;
}

absl::StatusOr<std::vector<CodegenBackend::EstimatedConfig>>
CublasLtBackend::GetSupportedConfigsWithEstimates(const HloInstruction& instr) {
  ABSL_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                   GetSupportedConfigs(instr));
  if (configs.empty()) {
    return std::vector<CodegenBackend::EstimatedConfig>{};
  }

  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

  std::optional<absl::Duration> estimated_duration = std::nullopt;
  if (backend_config.epilogue() == GemmBackendConfig::DEFAULT) {
    // There is no specialized cuBLASLt cost model that we can use (yet?) but
    // we expect it to be somewhere in the ballpark of the fastest Triton dot.
    absl::StatusOr<std::optional<absl::Duration>> estimate_or =
        EstimateTritonDotRuntime(instr, backend_config,
                                 target_config().device_description,
                                 debug_options(), mlir_context_);
    if (estimate_or.ok()) {
      estimated_duration = *estimate_or;
    } else {
      VLOG(1) << "Failed to estimate cuBLASLt dot runtime: "
              << estimate_or.status();
    }
  }

  // We do not have a good way to estimate each cublaslt config, so we use the
  // same dot estimate for all of them.
  std::vector<EstimatedConfig> results;
  results.reserve(configs.size());
  for (std::unique_ptr<BackendConfig>& config : configs) {
    results.push_back(EstimatedConfig{std::move(config), estimated_duration});
  }
  return results;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
CublasLtBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "Not a CublasLt custom call instruction.");
  }

  auto config = std::make_unique<BackendConfig>();
  auto* gemm_key = config->mutable_gemm();
  gemm_key->set_algorithm(0);
  // We don't know the workspace size in advance, so we pick a reasonably large
  // value that is likely to be sufficient.
  gemm_key->set_autotune_workspace_size(4194304);  // 4MiB
  return config;
}

absl::Status CublasLtBackend::ApplyConfig(HloInstruction& instr,
                                          const BackendConfig& config) {
  if (!config.has_gemm()) {
    return absl::InvalidArgumentError(
        "Expected GemmKey config for CublasLtBackend.");
  }
  const AutotuneResult::GemmKey& gemm_key = config.gemm();
  ABSL_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();
  backend_config.set_selected_algorithm(gemm_key.algorithm());
  backend_config.set_autotune_workspace_size(
      gemm_key.autotune_workspace_size());
  ABSL_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));

  if (instr.shape().IsTuple() && !instr.shape().tuple_shapes().empty()) {
    Shape* workspace_shape = instr.mutable_shape()->mutable_tuple_shapes(
        instr.shape().tuple_shapes().size() - 1);
    if (workspace_shape->element_type() == S8 &&
        workspace_shape->dimensions().size() == 1) {
      workspace_shape->set_dimensions(0, gemm_key.autotune_workspace_size());
      if (HloModule* module = instr.GetModule()) {
        if (module->entry_computation() &&
            module->entry_computation()->root_instruction() == &instr) {
          *module->mutable_entry_computation_layout()->mutable_result_layout() =
              ShapeLayout(instr.shape());
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
