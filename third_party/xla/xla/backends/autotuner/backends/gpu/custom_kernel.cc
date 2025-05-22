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

#include "xla/backends/autotuner/backends/gpu/custom_kernel.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CustomKernelBackendConfig = AutotuneResult::CustomKernelFusionKey;

namespace {
bool IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    LOG(ERROR)
        << "CustomKernelBackend doesn't support non-fusion instructions.";
    return false;
  }

  if (instr.backend_config<GpuBackendConfig>()
          ->fusion_backend_config()
          .kind() != kCustomFusionKind) {
    LOG(ERROR) << "CustomKernelBackend expected a custom fusion.";
    return false;
  }

  return true;
}
}  // namespace

absl::StatusOr<std::vector<CustomKernel>> LoadKernels(
    const HloInstruction* fusion_instruction,
    const se::DeviceDescription& device_description) {
  CustomFusionConfig config =
      fusion_instruction->backend_config<GpuBackendConfig>()
          ->fusion_backend_config()
          .custom_fusion_config();
  CustomKernelFusionRegistry* registry = CustomKernelFusionRegistry::Default();
  CustomKernelFusion* custom_kernel_fusion = registry->Lookup(config.name());

  // If custom fusion is not found it means that some of the build targets might
  // not be statically linked into the binary.
  if (custom_kernel_fusion == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel fusion ", config.name(),
                     " not found in a default registry."));
  }

  // Load custom kernels that can implement a fusion computation.
  TF_ASSIGN_OR_RETURN(
      std::vector<CustomKernel> kernels,
      custom_kernel_fusion->LoadKernels(
          device_description,
          fusion_instruction->fused_instructions_computation()));

  return kernels;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CustomKernelBackend::GetSupportedConfigs(
    const HloInstruction& instr,
    stream_executor::StreamExecutor* stream_executor) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "CustomKernelBackend does not support this instruction.");
  }
  TF_ASSIGN_OR_RETURN(
      std::vector<CustomKernel> kernels,
      LoadKernels(&instr, stream_executor->GetDeviceDescription()));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  int num_kernels = kernels.size();
  configs.reserve(num_kernels);
  for (int i = 0; i < num_kernels; ++i) {
    auto config = std::make_unique<CustomKernelBackendConfig>();
    config->set_kernel_index(i);
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
CustomKernelBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "CustomKernelBackend does not support this instruction.");
  }

  auto config = std::make_unique<CustomKernelBackendConfig>();
  config->set_kernel_index(0);
  return config;
}

absl::Status CustomKernelBackend::ApplyConfig(HloInstruction& instr,
                                              const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "CustomKernelBackend does not support this instruction.");
  }

  const CustomKernelBackendConfig custom_kernel_config =
      static_cast<const CustomKernelBackendConfig&>(config);

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  FusionBackendConfig* backend_config =
      gpu_config.mutable_fusion_backend_config();
  backend_config->mutable_custom_fusion_config()->set_kernel_index(
      custom_kernel_config.kernel_index());
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));

  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
