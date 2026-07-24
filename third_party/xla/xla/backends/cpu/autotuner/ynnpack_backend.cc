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

#include "xla/backends/cpu/autotuner/ynnpack_backend.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/gloop/util/status/status_macros.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/transforms/library_fusion_kinds.h"
#include "xla/backends/cpu/ynn_fusion_options.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/util.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<CodegenBackend>> YnnpackBackend::Create(
    Compiler* compiler) {
  return absl::WrapUnique(new YnnpackBackend(compiler));
}

bool YnnpackBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto backend_config_or = instr.backend_config<xla::cpu::BackendConfig>();
  if (!backend_config_or.ok()) {
    return false;
  }

  return backend_config_or->fusion_config().kind() == kYnnFusionKind;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::BackendConfig>>>
YnnpackBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<xla::BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  // Generate configurations to try.
  // YnnFusionOptions has use_threadpool. We try both true and false.
  for (bool use_threadpool : {false, true}) {
    auto config = std::make_unique<xla::BackendConfig>();
    auto* ynn_config = config->mutable_ynn_fusion();
    ynn_config->set_use_threadpool(use_threadpool);
    configs.push_back(std::move(config));
  }
  return configs;
}

// Use threadpool by default.
absl::StatusOr<std::unique_ptr<xla::BackendConfig>>
YnnpackBackend::GetDefaultConfig(const HloInstruction& instr) {
  auto config = std::make_unique<xla::BackendConfig>();
  auto* ynn_config = config->mutable_ynn_fusion();
  ynn_config->set_use_threadpool(true);
  return config;
}

absl::Status YnnpackBackend::ApplyConfig(HloInstruction& instr,
                                         const xla::BackendConfig& config) {
  ABSL_ASSIGN_OR_RETURN(auto backend_config,
                   instr.backend_config<xla::cpu::BackendConfig>());

  if (!config.has_ynn_fusion()) {
    return absl::InvalidArgumentError(
        "Expected YnnFusionOptions config for YnnpackBackend.");
  }
  const xla::cpu::YnnFusionOptions& ynn_options = config.ynn_fusion();

  auto* fusion_config = backend_config.mutable_fusion_config();
  *fusion_config->mutable_ynn_fusion_options() = ynn_options;

  ABSL_RETURN_IF_ERROR(instr.set_backend_config(backend_config));

  return absl::OkStatus();
}

}  // namespace xla::cpu
