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

#include "xla/backends/cpu/autotuner/xnnpack_backend.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/backends/cpu/xnnpack_config.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

absl::Status CheckIfXnnFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return xla::InvalidArgument(
        "XnnpackBackend only supports fusion instructions. Received %s.",
        HloOpcodeString(instr.opcode()));
  }
  if (!instr.has_backend_config()) {
    return xla::InvalidArgument("Instruction %s does not have backend config.",
                                instr.ToString());
  }

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr.backend_config<BackendConfig>());

  if (!backend_config.has_fusion_config()) {
    return xla::InvalidArgument(
        "Backend config %s does not have an fusion config.",
        backend_config.DebugString());
  }

  if (backend_config.fusion_config().kind() != kXnnFusionKind) {
    return xla::InvalidArgument(
        "Backend kind %s doesn't match expected kind %s.",
        backend_config.fusion_config().kind(), kXnnFusionKind);
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CodegenBackend>> XnnpackBackend::Create(
    Compiler* compiler) {
  return absl::WrapUnique(new XnnpackBackend(compiler));
}

bool XnnpackBackend::IsSupported(const HloInstruction& instr) {
  return CheckIfXnnFusion(instr).ok();
}

absl::StatusOr<std::vector<std::unique_ptr<xla::BackendConfig>>>
XnnpackBackend::GetSupportedConfigs(const HloInstruction& instr) {
  TF_RETURN_IF_ERROR(CheckIfXnnFusion(instr));
  std::vector<std::unique_ptr<xla::BackendConfig>> configs;
  {
    Config config;
    config.set_use_threadpool(true);
    configs.push_back(std::make_unique<Config>(config));
  }

  {
    Config config;
    config.set_use_threadpool(false);
    configs.push_back(std::make_unique<Config>(config));
  }
  return configs;
}
absl::StatusOr<std::unique_ptr<xla::BackendConfig>>
XnnpackBackend::GetDefaultConfig(const HloInstruction& instr) {
  TF_RETURN_IF_ERROR(CheckIfXnnFusion(instr));
  auto config = std::make_unique<Config>();
  config->set_use_threadpool(true);
  return config;
}

absl::Status XnnpackBackend::ApplyConfig(HloInstruction& instr,
                                         const xla::BackendConfig& config) {
  TF_RETURN_IF_ERROR(CheckIfXnnFusion(instr));
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr.backend_config<xla::cpu::BackendConfig>());

  const XnnpackBackend::Config* xnn_config =
      tsl::down_cast<const XnnpackBackend::Config*>(&config);
  TF_RET_CHECK(xnn_config != nullptr);

  *backend_config.mutable_fusion_config()->mutable_xnn_fusion_config() =
      *xnn_config;

  TF_RETURN_IF_ERROR(instr.set_backend_config(backend_config));

  return absl::OkStatus();
}

}  // namespace xla::cpu
