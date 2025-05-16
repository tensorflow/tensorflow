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

#include "xla/backends/autotuner/backends/gpu/fission.h"

#include <iterator>
#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/transforms/custom_kernel_fusion_rewriter.h"
#include "xla/service/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/service/gpu/transforms/gemm_rewriter.h"
#include "xla/service/gpu/transforms/priority_fusion.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using FissionBackendConfig = HloModuleProto;

HloCostAnalysis::Options PriorityFusionOptions() {
  // The real pointer size is set in GpuCompiler. In HloCostAnalysis, the
  // pointer size is used only to determine the size of tuple types. We
  // shouldn't have any tuples in the autotuned module, so it's safe to use
  // the default value here, instead of piping the real value.
  return {.count_multiple_input_accesses = true};
}

// Logs the error message if the status is not ok, otherwise returns true.
template <typename T>
bool contains_error(absl::StatusOr<T> status_or,
                    absl::string_view error_message) {
  if (!status_or.ok()) {
    LOG(WARNING) << error_message << ": " << status_or.status();
    return true;
  }
  return false;
}

// Unfuses a fusion instruction and rewrites it to using a cublas or cublasLt
// custom call for the dot operation.
// If rewrite_to_cublaslt is true, we will try to rewrite the dot to a cublasLt
// custom call, otherwise we will try to rewrite it to a cublas custom call.
std::vector<std::unique_ptr<FissionBackendConfig>> FissionFusionToCublas(
    const HloFusionInstruction* fusion, se::StreamExecutor* stream_executor,
    bool rewrite_to_cublaslt) {
  const HloComputation* fusion_computation = fusion->called_computation();
  std::unique_ptr<HloModule> hlo_module =
      ExtractComputationIntoNewModule(*fusion_computation);
  if (rewrite_to_cublaslt) {
    hlo_module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_enable_cublaslt(true);
  }
  HloInstruction* dot = hlo_query::GetFirstInstructionWithOpcode(
      *hlo_module->entry_computation(), HloOpcode::kDot);

  if (dot == nullptr) {
    LOG(WARNING) << "No dot instruction found in the fusion.";
    return {};
  }

  // Substitute algorithms, which are not supported by cuBLAS for the check, but
  // don't use cuBlas in the end. This assumes that the substituting algorithm
  // has result which are close enough for the check in this file.
  if (dot->precision_config().algorithm() ==
      PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3) {
    dot->mutable_precision_config()->set_algorithm(
        PrecisionConfig::ALG_DOT_F32_F32_F32);
  }

  const se::DeviceDescription device_description =
      stream_executor->GetDeviceDescription();
  bool is_rewritten_to_cublas_custom_call = false;
  for (GemmRewriterOptions::DType dtype :
       {GemmRewriterOptions::DType::kFp8Only,
        GemmRewriterOptions::DType::kNonFp8Only}) {
    DotAlgorithmRewriter dot_algorithm_rewriter;

    if (contains_error(dot_algorithm_rewriter.Run(hlo_module.get()),
                       "Dot algorithm rewriter failed")) {
      return {};
    };

    GemmRewriter gemm_rewriter(device_description.gpu_compute_capability(),
                               device_description.runtime_version(),
                               GemmRewriterOptions{dtype});
    absl::StatusOr<bool> changed = gemm_rewriter.Run(hlo_module.get());
    if (contains_error(changed, "Gemm rewriter failed")) {
      return {};
    }

    is_rewritten_to_cublas_custom_call |= *changed;

    PriorityFusion fusion_pass(
        /*thread_pool=*/nullptr, device_description, PriorityFusionOptions());
    if (contains_error(fusion_pass.Run(hlo_module.get()),
                       "Priority fusion failed")) {
      return {};
    }
  }

  if (is_rewritten_to_cublas_custom_call) {
    std::vector<std::unique_ptr<FissionBackendConfig>> configs;
    configs.push_back(std::make_unique<HloModuleProto>(hlo_module->ToProto()));
    return configs;
  }

  return {};
}

std::vector<std::unique_ptr<FissionBackendConfig>> FissionFusionToCustomKernel(
    const HloFusionInstruction* fusion, se::StreamExecutor* stream_executor) {
  const HloComputation* fusion_computation = fusion->called_computation();
  std::unique_ptr<HloModule> hlo_module =
      ExtractComputationIntoNewModule(*fusion_computation);
  CustomKernelFusionRewriter custom_kernel_fusion_rewriter(
      &stream_executor->GetDeviceDescription());
  PriorityFusion fusion_pass(
      /*thread_pool=*/nullptr, stream_executor->GetDeviceDescription(),
      PriorityFusionOptions());
  absl::StatusOr<bool> is_rewritten_to_custom_kernel =
      custom_kernel_fusion_rewriter.Run(hlo_module.get());
  if (contains_error(is_rewritten_to_custom_kernel,
                     "Custom kernel fusion rewriter failed")) {
    return {};
  }
  if (contains_error(fusion_pass.Run(hlo_module.get()),
                     "Priority fusion failed")) {
    return {};
  };
  if (*is_rewritten_to_custom_kernel) {
    std::vector<std::unique_ptr<FissionBackendConfig>> configs;
    configs.push_back(std::make_unique<HloModuleProto>(hlo_module->ToProto()));
    return configs;
  }

  return {};
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
FissionBackend::GetSupportedConfigs(const HloInstruction& instr,
                                    se::StreamExecutor* stream_executor) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return absl::InvalidArgumentError("Not a fusion instruction.");
  }

  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(&instr);

  std::vector<std::unique_ptr<FissionBackendConfig>> cublas_configs =
      FissionFusionToCublas(fusion, stream_executor,
                            /*rewrite_to_cublaslt=*/false);
  std::vector<std::unique_ptr<FissionBackendConfig>> cublaslt_configs =
      FissionFusionToCublas(fusion, stream_executor,
                            /*rewrite_to_cublaslt=*/true);
  std::vector<std::unique_ptr<FissionBackendConfig>> custom_kernel_configs =
      FissionFusionToCustomKernel(fusion, stream_executor);

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(cublas_configs.size() + cublaslt_configs.size() +
                  custom_kernel_configs.size());
  configs.insert(configs.end(), std::make_move_iterator(cublas_configs.begin()),
                 std::make_move_iterator(cublas_configs.end()));
  configs.insert(configs.end(),
                 std::make_move_iterator(cublaslt_configs.begin()),
                 std::make_move_iterator(cublaslt_configs.end()));
  configs.insert(configs.end(),
                 std::make_move_iterator(custom_kernel_configs.begin()),
                 std::make_move_iterator(custom_kernel_configs.end()));

  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> FissionBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  return absl::InvalidArgumentError(
      "FissionBackend doesn't support getting a default config.");
}

absl::StatusOr<std::unique_ptr<HloModule>> FissionBackend::WrapInModule(
    const HloInstruction& hlo_instruction, const BackendConfig& config) {
  return absl::UnimplementedError("Not implemented.");
}

absl::StatusOr<std::unique_ptr<HloModule>> FissionBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  return absl::UnimplementedError("Not implemented.");
}

}  // namespace gpu
}  // namespace xla
