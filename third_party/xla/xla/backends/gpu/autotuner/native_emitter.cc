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

#include "xla/backends/gpu/autotuner/native_emitter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Returns true if the given instruction is a fusion instruction that is
// supported by the native emitter backend.
//
// There is no guarantee that the native emitter backend can actually compile if
// it has a config for another backend, and we currently don't have an easy way
// to check that. Therefore, we only support fusions that are already set up to
// go through the native emitter.
bool NativeEmitterBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto fusion_kind = Cast<HloFusionInstruction>(&instr)->fusion_kind();
  return fusion_kind != HloInstruction::FusionKind::kCustom;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
NativeEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }
  auto config = GetDefaultConfig(instr);
  if (config.ok()) {
    configs.push_back(std::move(config.value()));
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
NativeEmitterBackend::GetDefaultConfig(const HloInstruction& instr) {
  NativeEmitterBackendConfig config;
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(config);
  return any;
}

absl::Status NativeEmitterBackend::ApplyConfig(HloInstruction& instr,
                                               const BackendConfig& config) {
  NativeEmitterBackendConfig native_emitter_fusion_config;
  if (!config.UnpackTo(&native_emitter_fusion_config)) {
    return absl::InvalidArgumentError(
        "Invalid backend config type for NativeEmitterBackendConfig.");
  }
  auto fusion_instr = Cast<HloFusionInstruction>(&instr);
  fusion_instr->set_fusion_kind(HloInstruction::FusionKind::kInput);
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                      instr.backend_config<GpuBackendConfig>());
  *gpu_backend_config.mutable_native_emitter_backend_config() =
      native_emitter_fusion_config;
  TF_RETURN_IF_ERROR(fusion_instr->set_backend_config(gpu_backend_config));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
