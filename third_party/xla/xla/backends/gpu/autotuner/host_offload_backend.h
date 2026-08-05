/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_HOST_OFFLOAD_BACKEND_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_HOST_OFFLOAD_BACKEND_H_

#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class HostOffloadBackend final : public GpuCodegenBackend {
 public:
  HostOffloadBackend(const DebugOptions* debug_options, Compiler* compiler,
                     const Compiler::GpuTargetConfig* target_config,
                     stream_executor::StreamExecutor* stream_executor = nullptr,
                     bool uses_last_output_for_scratch = false)
      : GpuCodegenBackend(autotuner::Backend::HOST_OFFLOAD, debug_options,
                          compiler, target_config, stream_executor,
                          uses_last_output_for_scratch) {}

  std::string version() const final { return "0.0.0-experimental"; }
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) final {
    ASSIGN_OR_RETURN(std::unique_ptr<BackendConfig> default_config,
                     GetDefaultConfig(instr));
    std::vector<std::unique_ptr<BackendConfig>> res;
    res.push_back(std::move(default_config));
    return res;
  }

  // Returns a default config for the given HLO instruction.
  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) final {
    return std::make_unique<BackendConfig>();
  }

  // Apply config to the given HLO instruction.
  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) final {
    if (!IsSupported(instr)) {
      return absl::InvalidArgumentError(
          "HostOffloadBackend does not support this instruction.");
    }

    ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                     instr.backend_config<GpuBackendConfig>());
    if (instr.opcode() == HloOpcode::kFusion) {
      FusionBackendConfig& backend_config =
          *gpu_config.mutable_fusion_backend_config();
      backend_config.set_kind("host_offload");
    }
    gpu_config.set_device_type(DeviceType::DEVICE_TYPE_HOST);
    RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));

    LOG(ERROR) << "Apply config HOST OFFLOAD" << instr.opcode();
    return absl::OkStatus();
  }

  // Returns true if the backend can produce numerically wrong results.
  bool CanProduceWrongResults() const final { return false; }

 private:
  bool IsSupported(const HloInstruction& instr) final {
    std::set<HloOpcode> supported{
        HloOpcode::kFusion, HloOpcode::kRngGetAndUpdateState, HloOpcode::kSort,
        HloOpcode::kCall};  //, HloOpcode::kConditional};
    if (supported.find(instr.opcode()) == supported.end()) {
      return false;
    }
    if (instr.opcode() == HloOpcode::kFusion) {
      for (HloInstruction* instr : instr.fused_instructions()) {
        if (instr->opcode() == HloOpcode::kBitcast) {
          return false;
        }
      }
    }
    auto gpu_config = instr.backend_config<GpuBackendConfig>();
    if (!gpu_config.ok()) {
      return false;
    }
    /*if (gpu_config->has_fusion_backend_config()) {
      return true;
      }*/
    return true;
  }
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_HOST_OFFLOAD_BACKEND_H_
