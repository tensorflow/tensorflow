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

#include "xla/backends/cpu/autotuner/llvm_kernel_backend.h"

#include <array>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<CodegenBackend>> LlvmKernelBackend::Create(
    Compiler* compiler) {
  return absl::WrapUnique(new LlvmKernelBackend(compiler));
}

// Adding other instructions to supports requires adding the
// opcode here.
bool LlvmKernelBackend::IsSupported(const HloInstruction& instr) {
  auto is_supported_fusion = [](const HloInstruction& instr) {
    if (instr.opcode() != HloOpcode::kFusion) {
      return false;
    }
    auto* fusion = Cast<HloFusionInstruction>(&instr);
    return fusion->fused_expression_root()->opcode() == HloOpcode::kScatter;
  };
  return instr.opcode() == HloOpcode::kConcatenate ||
         is_supported_fusion(instr);
}

absl::StatusOr<std::vector<std::unique_ptr<xla::BackendConfig>>>
LlvmKernelBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<xla::BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  constexpr std::array<bool, 2> boolean_options = {false, true};
  for (const auto& disable_loop_unrolling : boolean_options) {
    for (const auto& slp_vectorizer_disabled : boolean_options) {
      for (const auto& optimize_for_size : boolean_options) {
        Config config;
        config.set_disable_loop_unrolling(disable_loop_unrolling);
        config.set_slp_vectorizer_disabled(slp_vectorizer_disabled);
        config.set_optimize_for_size(optimize_for_size);
        configs.push_back(std::make_unique<Config>(config));
      }
    }
  }
  return configs;
}

absl::StatusOr<std::unique_ptr<xla::BackendConfig>>
LlvmKernelBackend::GetDefaultConfig(const HloInstruction& instr) {
  auto config = std::make_unique<Config>();
  config->set_disable_loop_unrolling(false);
  config->set_slp_vectorizer_disabled(false);
  config->set_optimize_for_size(false);
  return config;
}

absl::Status LlvmKernelBackend::ApplyConfig(HloInstruction& instr,
                                            const xla::BackendConfig& config) {
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr.backend_config<xla::cpu::BackendConfig>());

  const LlvmKernelBackend::Config* llvm_kernel_config =
      tsl::down_cast<const LlvmKernelBackend::Config*>(&config);
  TF_RET_CHECK(llvm_kernel_config != nullptr);

  *backend_config.mutable_llvm_kernel_options() = *llvm_kernel_config;

  TF_RETURN_IF_ERROR(instr.set_backend_config(backend_config));

  return absl::OkStatus();
}

}  // namespace xla::cpu
