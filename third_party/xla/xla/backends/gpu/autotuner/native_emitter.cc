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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/SmallSet.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

llvm::SmallSet<int64_t, 4> ComputeUnrollFactors(
    const HloInstruction& instr, int64_t default_unroll_factor,
    const se::DeviceDescription& device_description) {
  llvm::SmallSet<int64_t, 4> unroll_factors;
  unroll_factors.insert(default_unroll_factor);

  auto analysis = HloFusionAnalysis::Create(instr, device_description);
  int64_t num_elements = ShapeUtil::ElementsIn(analysis.first_result_shape());
  int64_t n_threads_max = analysis.device_info().threads_per_core_limit() *
                          analysis.device_info().core_count();
  if (num_elements >= n_threads_max &&
      !MayCausePerformanceDropIfUnrolled(analysis.fusion())) {
    int64_t max_unroll_factor = MaxUnrollFactor(&analysis);
    unroll_factors.insert(max_unroll_factor);
    if (max_unroll_factor > 1) {
      unroll_factors.insert(max_unroll_factor / 2);
    }
  }
  return unroll_factors;
}

}  // namespace

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
  auto fusion = Cast<HloFusionInstruction>(&instr);
  return fusion->fusion_kind() != HloInstruction::FusionKind::kCustom;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
NativeEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  ASSIGN_OR_RETURN(std::unique_ptr<BackendConfig> default_config_any,
                   GetDefaultConfig(instr));
  NativeEmitterBackendConfig default_config;
  if (!default_config_any->UnpackTo(&default_config)) {
    return absl::InternalError("Failed to unpack default config.");
  }
  // Tune unroll factor for loops.
  if (debug_options().xla_gpu_native_emitter_tune_unroll_factor_for_loops() &&
      default_config.type() == NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP) {
    llvm::SmallSet<int64_t, 4> unroll_factors =
        ComputeUnrollFactors(instr, default_config.unroll_factor(),
                             target_config().device_description);
    for (int64_t unroll_factor : unroll_factors) {
      NativeEmitterBackendConfig config;
      config.set_type(NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP);
      config.set_unroll_factor(unroll_factor);
      auto any = std::make_unique<google::protobuf::Any>();
      any->PackFrom(config);
      configs.push_back(std::move(any));
    }
    return configs;
  }
  configs.push_back(std::move(default_config_any));
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
NativeEmitterBackend::GetDefaultConfig(const HloInstruction& instr) {
  NativeEmitterBackendConfig config;
  if (IsSupported(instr)) {
    if (debug_options().xla_gpu_native_emitter_tune_unroll_factor_for_loops()) {
      se::DeviceDescription device_description =
          target_config().device_description;
      HloFusionAnalysis fusion_analysis =
          HloFusionAnalysis::Create(instr, device_description);
      if (fusion_analysis.emitter_fusion_kind() ==
          HloFusionAnalysis::EmitterFusionKind::kLoop) {
        config.set_type(NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP);
        config.set_unroll_factor(ComputeLoopFusionConfig(fusion_analysis));
      }
    }
  }
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
  HloInstruction::FusionKind emitter_fusion_kind =
      native_emitter_fusion_config.type() ==
              NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP
          ? HloInstruction::FusionKind::kLoop
          : HloInstruction::FusionKind::kInput;
  fusion_instr->set_fusion_kind(emitter_fusion_kind);
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                   instr.backend_config<GpuBackendConfig>());
  *gpu_backend_config.mutable_native_emitter_backend_config() =
      native_emitter_fusion_config;
  RETURN_IF_ERROR(fusion_instr->set_backend_config(gpu_backend_config));
  return absl::OkStatus();
}

}  // namespace xla::gpu
