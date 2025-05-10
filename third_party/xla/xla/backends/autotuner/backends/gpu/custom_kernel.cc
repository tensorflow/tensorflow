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
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CustomKernelBackendConfig = AutotuneResult::CustomKernelFusionKey;

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

std::vector<std::unique_ptr<BackendConfig>>
CustomKernelBackend::GetSupportedConfigs(
    const HloInstruction& instr,
    stream_executor::StreamExecutor* stream_executor) {
  if (instr.opcode() != HloOpcode::kFusion) {
    LOG(ERROR)
        << "CustomKernelBackend doesn't support non-fusion instructions.";
    return {};
  }

  absl::StatusOr<std::vector<CustomKernel>> kernels =
      LoadKernels(&instr, stream_executor->GetDeviceDescription());
  if (!kernels.ok()) {
    LOG(ERROR) << "Failed to load kernels: " << kernels.status();
    return {};
  }

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve((*kernels).size());
  for (int i = 0; i < (*kernels).size(); ++i) {
    std::unique_ptr<CustomKernelBackendConfig> config =
        std::make_unique<CustomKernelBackendConfig>();
    config->set_kernel_index(i);
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
CustomKernelBackend::GetDefaultConfig(const HloInstruction& instr) {
  // CustomKernels need a device description to load the kernels, so we can't
  // return a default config.
  return absl::InvalidArgumentError(
      "CustomKernelBackend doesn't support getting a default config.");
}

absl::StatusOr<std::unique_ptr<HloModule>> CustomKernelBackend::WrapInModule(
    const HloInstruction& hlo_instruction, const BackendConfig& config) {
  return absl::InvalidArgumentError(
      "CustomKernelBackend doesn't support wrapping in a module.");
}

absl::StatusOr<std::unique_ptr<HloModule>> CustomKernelBackend::RunHloPasses(
    std::unique_ptr<HloModule> hlo_module,
    const Compiler::CompileOptions& options) {
  return absl::InvalidArgumentError(
      "CustomKernelBackend doesn't support wrapping in a module.");
}

}  // namespace gpu
}  // namespace xla
