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

#include "xla/backends/autotuner/autotuner.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/service/executable.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/fingerprint.h"

namespace xla {

namespace {

tsl::Fprint128 GetFingerprint(const HloInstruction* instr) {
  auto options = HloPrintOptions::Fingerprint();
  options.set_print_backend_config(true);
  options.set_sort_backend_config(true);
  options.set_print_operand_shape(true);

  return tsl::Fingerprint128(instr->ToString(options));
}
}  // namespace

absl::StatusOr<std::unique_ptr<Autotuner>> Autotuner::Create(
    std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
    std::unique_ptr<Profiler> profiler, AutotuneConfig autotune_config) {
  if (codegen_backends.empty()) {
    return absl::InvalidArgumentError("No codegen backends provided");
  }
  return absl::WrapUnique(new Autotuner(std::move(codegen_backends),
                                        std::move(profiler),
                                        std::move(autotune_config)));
}

absl::Status Autotuner::Autotune(HloInstruction* instr) {
  VLOG(1) << "Autotuning HLO: " << instr->ToString();
  TF_ASSIGN_OR_RETURN(auto best_config, GetBestConfig(instr));
  CodegenBackend* best_codegen_backend = best_config.first;
  return best_codegen_backend->ApplyConfig(*instr, *best_config.second);
}

absl::StatusOr<std::pair<CodegenBackend*, std::unique_ptr<BackendConfig>>>
Autotuner::GetBestConfig(HloInstruction* instr) {
  std::unique_ptr<BackendConfig> best_config;
  CodegenBackend* best_codegen_backend = nullptr;
  absl::Duration min_duration = absl::InfiniteDuration();
  for (auto& codegen_backend : codegen_backends_) {
    TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                        codegen_backend->GetSupportedConfigs(*instr));
    VLOG(1) << "Got " << configs.size()
            << " configs from codegen backend: " << codegen_backend->name();
    std::vector<std::unique_ptr<Executable>> executables;
    std::vector<std::unique_ptr<BackendConfig>> valid_configs;
    for (auto& config : configs) {
      VLOG(2) << "Trying to compile config: " << config->DebugString();
      auto executable = codegen_backend->Compile(*instr, *config);
      // TODO b/407495547: Change it to tolerate only specific compilation
      // errors, as opposed to all errors.
      if (!executable.ok() && autotune_config_.skip_failing_configs) {
        VLOG(1) << "Failed to compile: " << executable.status();
        continue;
      }
      executables.push_back(std::move(executable.value()));
      valid_configs.push_back(std::move(config));
    }
    CHECK_EQ(executables.size(), valid_configs.size());
    TF_ASSIGN_OR_RETURN(
        std::vector<ProfileResult> results,
        profiler_->ProfileWithSharedBuffers(std::move(executables)));
    for (int i = 0; i < results.size(); ++i) {
      if (results[i].duration < min_duration) {
        min_duration = results[i].duration;
        best_config = std::move(valid_configs[i]);
        best_codegen_backend = codegen_backend.get();
      }
    }
  }
  if (best_config == nullptr) {
    return absl::InternalError("No config found!");
  }
  CHECK(best_codegen_backend != nullptr);
  return std::make_pair(best_codegen_backend, std::move(best_config));
}

Autotuner::InstructionsByFingerprint Autotuner::GetAutotuningCandidates(
    const HloModule* module, const InstructionFilterFn& should_autotune) {
  InstructionsByFingerprint instrunctions_by_fingerprint;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (should_autotune(*instr)) {
        instrunctions_by_fingerprint[GetFingerprint(instr)].push_back(instr);
      }
    }
  }
  return instrunctions_by_fingerprint;
}

absl::Status Autotuner::Autotune(HloModule* module,
                                 const InstructionFilterFn& should_autotune) {
  InstructionsByFingerprint instrunctions_by_fingerprint =
      GetAutotuningCandidates(module, should_autotune);
  if (instrunctions_by_fingerprint.empty()) {
    VLOG(1) << "No instructions to autotune.";
    return absl::OkStatus();
  }

  VLOG(1) << "Autotuning " << instrunctions_by_fingerprint.size()
          << " unique instructions.";
  for (auto& [_, instructions] : instrunctions_by_fingerprint) {
    CHECK(!instructions.empty());
    TF_ASSIGN_OR_RETURN(auto best_config, GetBestConfig(instructions[0]));
    CodegenBackend* best_codegen_backend = best_config.first;
    for (auto* instr : instructions) {
      TF_RETURN_IF_ERROR(
          best_codegen_backend->ApplyConfig(*instr, *best_config.second));
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
