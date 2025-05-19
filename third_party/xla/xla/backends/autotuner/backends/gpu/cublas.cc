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

#include "xla/backends/autotuner/backends/gpu/cublas.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/autotuning/redzone_buffers.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CublasBackendConfig = AutotuneResult::GemmKey;

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CublasBackend::GetSupportedConfigs(
    const HloInstruction& instr,
    stream_executor::StreamExecutor* stream_executor) {
  if (!IsLegacyCublasMatmul(instr)) {
    return absl::InvalidArgumentError(
        "CublasBackend does not support this instruction.");
  }

  std::unique_ptr<se::DeviceMemoryAllocator> allocator =
      std::make_unique<se::StreamExecutorMemoryAllocator>(stream_executor);
  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      allocator->GetStream(stream_executor->device_ordinal()));

  // We use GemmConfig::For with GemmBackendConfig as a fallback because
  // Matmul_utils.cc relies on backend config to determine gemm contracting
  // dimensions.
  GemmBackendConfig backend_config;
  backend_config =
      instr.backend_config<GpuBackendConfig>()->gemm_backend_config();
  TF_ASSIGN_OR_RETURN(
      GemmConfig gemm_config,
      GemmConfig::For(
          &instr, backend_config,
          target_config().device_description.gpu_compute_capability()));

  TF_ASSIGN_OR_RETURN(RedzoneBuffers rz_buffers,
                      RedzoneBuffers::FromInstruction(
                          instr, allocator.get(), stream,
                          RedzoneBuffers::kAllInputsAllOutputs, true, true,
                          instr.GetModule()
                              ->config()
                              .debug_options()
                              .xla_gpu_redzone_padding_bytes()));

  TF_ASSIGN_OR_RETURN(
      GemmConfig::DescriptorsTuple desc,
      gemm_config.GetMatrixDescriptors(rz_buffers.input_buffers().at(0),
                                       rz_buffers.input_buffers().at(1),
                                       rz_buffers.output_buffers().at(0)));

  se::blas::BlasSupport* blas = stream_executor->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("Failed to getBlas support.");
  }
  std::vector<se::blas::AlgorithmType> algorithms;
  blas->GetBlasGemmAlgorithms(stream, desc.lhs, desc.rhs, &desc.output,
                              &gemm_config.alpha, &gemm_config.beta,
                              &algorithms);

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(algorithms.size());
  for (se::blas::AlgorithmType algorithm : algorithms) {
    std::unique_ptr<CublasBackendConfig> gemm_key =
        std::make_unique<CublasBackendConfig>();
    gemm_key->set_algorithm(algorithm);
    configs.push_back(std::move(gemm_key));
  }
  return configs;
}

HloCostAnalysis::Options PriorityFusionOptions() {
  // The real pointer size is set in GpuCompiler. In HloCostAnalysis, the
  // pointer size is used only to determine the size of tuple types. We
  // shouldn't have any tuples in the autotuned module, so it's safe to use
  // the default value here, instead of piping the real value.
  HloCostAnalysis::Options options;
  options.count_multiple_input_accesses = true;
  return options;
}

absl::StatusOr<std::unique_ptr<HloModule>> RewriteToCublasCustomCall(
    std::unique_ptr<HloModule> hlo_module,
    const se::DeviceDescription& gpu_device_info) {
  HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *hlo_module->entry_computation(), HloOpcode::kDot);
  // Substitute algorithms, which are not supported by cuBLAS for the check, but
  // don't use cuBlas in the end. This assumes that the substituting algorithm
  // has result which are close enough for the check in this file.
  if (dot->precision_config().algorithm() ==
      PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3) {
    dot->mutable_precision_config()->set_algorithm(
        PrecisionConfig::ALG_DOT_F32_F32_F32);
  }

  for (GemmRewriterOptions::DType dtype :
       {GemmRewriterOptions::DType::kFp8Only,
        GemmRewriterOptions::DType::kNonFp8Only}) {
    GemmRewriter gemm_rewriter(gpu_device_info.cuda_compute_capability(),
                               gpu_device_info.runtime_version(),
                               GemmRewriterOptions{dtype});
    DotAlgorithmRewriter dot_algorithm_rewriter;
    PriorityFusion fusion_pass(
        /*thread_pool=*/nullptr, gpu_device_info, PriorityFusionOptions());
    TF_RETURN_IF_ERROR(dot_algorithm_rewriter.Run(hlo_module.get()).status());
    TF_RETURN_IF_ERROR(gemm_rewriter.Run(hlo_module.get()).status());
    TF_RETURN_IF_ERROR(fusion_pass.Run(hlo_module.get()).status());
  }

  return hlo_module;
}

void SubstituteCublasAlgorithms(const HloInstruction* gemm,
                                se::blas::AlgorithmType algorithm) {
  GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();
  backend_config.set_selected_algorithm(algorithm);
}

absl::StatusOr<std::unique_ptr<BackendConfig>> CublasBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (!IsLegacyCublasMatmul(instr)) {
    return absl::InvalidArgumentError(
        "CublasBackend does not support this instruction.");
  }

  std::unique_ptr<CublasBackendConfig> gemm_key =
      std::make_unique<CublasBackendConfig>();
  gemm_key->set_algorithm(se::blas::kDefaultAlgorithm);
  return gemm_key;
}

absl::StatusOr<std::unique_ptr<HloModule>> CublasBackend::WrapInModule(
    const HloInstruction& hlo_instruction, const BackendConfig& config) {
  return absl::UnimplementedError("Not implemented.");
}

absl::StatusOr<std::unique_ptr<HloModule>> CublasBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  return absl::UnimplementedError("Not implemented.");
}

}  // namespace gpu
}  // namespace xla
