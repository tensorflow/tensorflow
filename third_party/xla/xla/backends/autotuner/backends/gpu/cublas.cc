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

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CublasBackendConfig = AutotuneResult::GemmKey;

absl::StatusOr<const HloInstruction*> GetGemmInstruction(
    const HloInstruction* instr) {
  // Is dot instruction.
  if (instr->opcode() == HloOpcode::kDot) {
    return instr;
  }

  // Is Cublas custom call instruction.
  if (IsLegacyCublasMatmul(*instr)) {
    return instr;
  }

  // Is fusion instruction containing a GEMM instruction.
  if (instr->opcode() == HloOpcode::kFusion) {
    auto fused_instructions = instr->fused_instructions();
    auto gemm_instr_it =
        absl::c_find_if(fused_instructions, HloPredicateIsOp<HloOpcode::kDot>);
    if (gemm_instr_it != std::end(fused_instructions)) {
      return *gemm_instr_it;
    }
  }

  return absl::InvalidArgumentError(
      "CublasBackend didn't find a compatible Cublas GEMM.");
}

std::vector<std::unique_ptr<BackendConfig>> CublasBackend::GetSupportedConfigs(
    const HloInstruction& instr,
    stream_executor::StreamExecutor* stream_executor) {
  absl::StatusOr<const HloInstruction*> gemm_instr = GetGemmInstruction(&instr);
  if (!gemm_instr.ok()) {
    LOG(ERROR) << "Failed to get GEMM instruction: " << gemm_instr.status();
    return {};
  }
  std::unique_ptr<se::DeviceMemoryAllocator> allocator =
      std::make_unique<se::StreamExecutorMemoryAllocator>(stream_executor);
  absl::StatusOr<se::Stream*> stream =
      allocator->GetStream(stream_executor->device_ordinal());
  if (!stream.ok()) {
    LOG(ERROR) << "Failed to get stream: " << stream.status();
    return {};
  }

  // We use GemmConfig::For with GemmBackendConfig as a fallback because
  // Matmul_utils.cc relies on backend config to determine gemm contracting
  // dimensions.
  GemmBackendConfig backend_config;
  // For custom call, we use the backend config from the custom call.
  if ((*gemm_instr)->opcode() == HloOpcode::kCustomCall) {
    backend_config = (*gemm_instr)
                         ->backend_config<GpuBackendConfig>()
                         ->gemm_backend_config();
  } else {
    *backend_config.mutable_dot_dimension_numbers() =
        (*gemm_instr)->dot_dimension_numbers();
  }
  absl::StatusOr<GemmConfig> gemm_config = GemmConfig::For(
      *gemm_instr, backend_config,
      target_config().device_description.gpu_compute_capability());
  if (!gemm_config.ok()) {
    LOG(ERROR) << "Failed to get GEMM config: " << gemm_config.status();
    return {};
  }

  // Get dummy buffers for the GEMM instruction.
  const HloInstruction* lhs_operand = (*gemm_instr)->operand(0);
  se::DeviceMemoryBase lhs_buffer = se::DeviceMemoryBase(
      nullptr, xla::ShapeUtil::ByteSizeOf(lhs_operand->shape()));
  const HloInstruction* rhs_operand = (*gemm_instr)->operand(1);
  se::DeviceMemoryBase rhs_buffer = se::DeviceMemoryBase(
      nullptr, xla::ShapeUtil::ByteSizeOf(rhs_operand->shape()));
  // For custom call, the output buffer is the first tuple element.
  const Shape& output_shape = (*gemm_instr)->opcode() == HloOpcode::kCustomCall
                                  ? (*gemm_instr)->shape().tuple_shapes(0)
                                  : (*gemm_instr)->shape();
  se::DeviceMemoryBase output_buffer =
      se::DeviceMemoryBase(nullptr, xla::ShapeUtil::ByteSizeOf(output_shape));

  absl::StatusOr<GemmConfig::DescriptorsTuple> desc =
      gemm_config->GetMatrixDescriptors(lhs_buffer, rhs_buffer, output_buffer);
  if (!desc.ok()) {
    LOG(ERROR) << "Failed to get GEMM descriptors: " << desc.status();
    return {};
  }

  se::blas::BlasSupport* blas = stream_executor->AsBlas();
  if (blas == nullptr) {
    LOG(ERROR) << "Failed to getBlas support.";
    return {};
  }
  std::vector<se::blas::AlgorithmType> algorithms;
  blas->GetBlasGemmAlgorithms(*stream, (*desc).lhs, (*desc).rhs,
                              &(*desc).output, &(*gemm_config).alpha,
                              &(*gemm_config).beta, &algorithms);

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
  LOG(ERROR) << "YOLO: SUBSTITUTING ALGORITHM " << gemm->ToString();
  GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();
  backend_config.set_selected_algorithm(algorithm);
}

absl::StatusOr<std::unique_ptr<BackendConfig>> CublasBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  TF_RETURN_IF_ERROR(GetGemmInstruction(&instr).status());

  std::unique_ptr<CublasBackendConfig> gemm_key =
      std::make_unique<CublasBackendConfig>();
  gemm_key->set_algorithm(se::blas::kDefaultAlgorithm);
  return gemm_key;
}

absl::StatusOr<std::unique_ptr<HloModule>> CublasBackend::WrapInModule(
    const HloInstruction& hlo_instruction, const BackendConfig& config) {
  const se::DeviceDescription& gpu_device_info =
      target_config().device_description;
  CublasBackendConfig cublas_config =
      static_cast<const CublasBackendConfig&>(config);

  // Handle dot instruction.
  if (hlo_instruction.opcode() == HloOpcode::kDot) {
    const HloComputation* computation = hlo_instruction.parent();
    std::unique_ptr<HloModule> hlo_module =
        ExtractComputationIntoNewModule(*computation);
    TF_ASSIGN_OR_RETURN(
        auto rewritten_module,
        RewriteToCublasCustomCall(std::move(hlo_module), gpu_device_info));
    SubstituteCublasAlgorithms(
        rewritten_module->entry_computation()->root_instruction(),
        cublas_config.algorithm());
    return rewritten_module;
  }

  // Handle Cublas custom call instruction.
  if (IsLegacyCublasMatmul(hlo_instruction)) {
    const HloComputation* computation = hlo_instruction.parent();
    std::unique_ptr<HloModule> hlo_module =
        ExtractComputationIntoNewModule(*computation);
    SubstituteCublasAlgorithms(
        hlo_module->entry_computation()->root_instruction()->operand(0),
        cublas_config.algorithm());
    return hlo_module;
  }

  // Handle fusion instruction.
  if (hlo_instruction.opcode() == HloOpcode::kFusion) {
    const HloFusionInstruction* fusion_instruction =
        Cast<HloFusionInstruction>(&hlo_instruction);
    const HloComputation* fusion_computation =
        fusion_instruction->called_computation();
    std::unique_ptr<HloModule> hlo_module =
        ExtractComputationIntoNewModule(*fusion_computation);
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModule> rewritten_module,
        RewriteToCublasCustomCall(std::move(hlo_module), gpu_device_info));
    HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
        *rewritten_module->entry_computation(), HloOpcode::kCustomCall);
    SubstituteCublasAlgorithms(dot, cublas_config.algorithm());
    return rewritten_module;
  }

  return absl::InvalidArgumentError(
      "CublasBackend didn't find a compatible cuBLAS GEMM.");
}

absl::StatusOr<std::unique_ptr<HloModule>> CublasBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  // Noop;

  return hlo_module;
}

}  // namespace gpu
}  // namespace xla
