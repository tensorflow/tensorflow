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
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

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

}  // namespace

bool CublasLtBackend::IsSupported(const HloInstruction& instr) {
  return IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr);
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CublasLtBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }

  GpuBackendConfig gpu_config =
      instr.backend_config<GpuBackendConfig>().value();
  const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

  TF_ASSIGN_OR_RETURN(
      GemmConfig gemm_config,
      GemmConfig::For(
          &instr, target_config().device_description.gpu_compute_capability()));

  TF_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                      AsBlasLtEpilogue(backend_config.epilogue()));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                      stream_executor()->CreateStream());

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BlasLt::MatmulPlan> plan,
      se::gpu::BlasLt::GetMatmulPlan(stream.get(), gemm_config, epilogue));

  const Shape& output_shape = instr.shape();
  if (!output_shape.IsTuple() || output_shape.tuple_shapes().empty()) {
    return Internal(
        "Invalid shape for CublasLt matmul: output is not a non-empty tuple.");
  }
  // The last element of the output tuple is the workspace.
  const int64_t workspace_size =
      ShapeUtil::ByteSizeOf(output_shape.tuple_shapes().back());

  TF_ASSIGN_OR_RETURN(
      std::vector<BlasLt::MatmulAlgorithm> algorithms,
      plan->GetAlgorithms(stream.get(), GemmConfig::kNumAlgorithms,
                          workspace_size));
  int num_algorithms = algorithms.size();
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(num_algorithms);
  for (int i = 0; i < num_algorithms; ++i) {
    CublasLtBackendConfig gemm_key;
    gemm_key.set_algorithm(i);
    gemm_key.set_autotune_workspace_size(workspace_size);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(gemm_key);
    configs.push_back(std::move(any));
  }

  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
CublasLtBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "Not a CublasLt custom call instruction.");
  }

  AutotuneResult::GemmKey gemm_key;
  gemm_key.set_algorithm(0);
  // We don't know the workspace size in advance, so we pick a reasonably large
  // value that is likely to be sufficient.
  gemm_key.set_autotune_workspace_size(4194304);  // 4MiB
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(gemm_key);
  return any;
}

absl::Status CublasLtBackend::ApplyConfig(HloInstruction& instr,
                                          const BackendConfig& config) {
  CublasLtBackendConfig gemm_key;
  if (!config.UnpackTo(&gemm_key)) {
    return absl::InvalidArgumentError(
        "Failed to unpack CublasLtBackendConfig from Any.");
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();
  backend_config.set_selected_algorithm(gemm_key.algorithm());
  backend_config.set_autotune_workspace_size(
      gemm_key.autotune_workspace_size());
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));

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
